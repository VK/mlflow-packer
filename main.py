from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict


import os
import mlflow
from mlflow.tracking import MlflowClient
import docker
import configparser

import requests

# load config once
config = configparser.ConfigParser()
config.read('/default.cfg')


def get_mflow_client():
    token = config.get('Databricks', 'TOKEN')
    registry = config.get('Databricks', 'REGISTRY')
    user = config.get('Databricks', 'USER')

    os.environ['DATABRICKS_HOST'] = registry
    os.environ['DATABRICKS_TOKEN'] = token
    os.environ["MLFLOW_TRACKING_TOKEN"] = token
    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
    mlflow.set_tracking_uri(registry)

    return MlflowClient()


def get_repo_tags(repo):
    
    repo = repo.replace("_", "-")
    
    base_url = config.get('Docker', 'HOST')
    token = config.get('Docker', 'TOKEN')
    user = config.get('Docker', 'USER')
    org = config.get('Docker', 'ORG')

    login_url = f"{base_url}/users/login"
    repo_url = f"{base_url}/repositories/{org}/{repo}/tags"

    tok_req = requests.post(
        login_url, json={"username": user, "password": token})
    token = tok_req.json()["token"]
    headers = {"Authorization": f"JWT {token}"}

    res = requests.get(repo_url, headers=headers)
    data = res.json()

    return [el["name"] for el in data["results"]]




def mlflow_build_docker(source, name, env):
    org = config.get('Docker', 'ORG')
    print(f'mlflow models build-docker -m {source} -n {org}/{name} --env-manager {env}')
    os.system(
        f'mlflow models build-docker -m {source} -n {org}/{name} --env-manager {env}'
    )


def docker_push(name):
    base_url = config.get('Docker', 'HOST')
    token = config.get('Docker', 'TOKEN')
    user = config.get('Docker', 'USER')
    org = config.get('Docker', 'ORG')

    client = docker.from_env()
    client.login(username = user, password=token)

    return client.api.push(f"{org}/{name}")



app = FastAPI(
    title="MLflow Packer",
    description="""Build and push mlflow models.""",
    version="0.0.10",
)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url='/docs')



class MlflowList(BaseModel):
    name: str
    latest_versions: Dict[str, str] = None


@app.get("/models",  response_model=List[MlflowList])
async def list_models():
    """
    List the available model versions in mlflow registry:

    - **name**: model name
    - **latest_versions**: set of version and Production, Staging, Archived state
    """


    # extract list of marked tags
    model_tags = [e.strip() for e in config.get("Models", "TAGS", fallback = "").split(",") if e != '']

    # use the mlflow client to get all models
    mlflow_c = get_mflow_client()
    models = [m for m in mlflow_c.list_registered_models() if any(
        [t  in m.tags.keys() for t in model_tags]) or len(model_tags) == 0]

    return JSONResponse([
        {
            "name": m.name,
            "latest_versions": {
                v.version: v.current_stage
                for v in m.latest_versions
             }
            } for m in models])





class DockerList(BaseModel):
    name: str
    versions: List[str]


@app.get("/images", response_model=List[DockerList])
async def list_docker_models():
    """
    List the available model versions in docker regristry:

    - **name**: model name
    - **versions**: list of all versions
    """


    # extract list of marked tags
    model_tags = [e.strip() for e in config.get("Models", "TAGS", fallback = "").split(",") if e != '']

    # use the mlflow client to get all models
    mlflow_c = get_mflow_client()
    models = [m for m in mlflow_c.list_registered_models() if any(
        [t  in m.tags.keys() for t in model_tags]) or len(model_tags) == 0]

    return JSONResponse([
        {
            "name": m.name,
            "versions":  get_repo_tags(m.name)
            } for m in models])






class BuildResponse(BaseModel):
    result: str

@app.get("/build", response_model=BuildResponse)
async def build_docker_model(name: str, version: str, env: str = "local"):
    """
    Build a new model version an push it to the server regitry

    - **name**: model name
    - **version**: the version to build
    - **env**: specify environment manager (local, conda, virtualenv)
    """

    # use the mlflow client to get all models
    mlflow_c = get_mflow_client()
    models = mlflow_c.list_registered_models()

    model = [m for m in models if m.name == name]

    if len(model) == 0:
        return JSONResponse({"result": "Model not found."})

    model = model[0]

    version = [v for v in model.latest_versions if v.version == version]


    if len(version) == 0:
        return JSONResponse({"result": "Version not found."})

    version = version[0]

    new_name = f"{model.name.lower().replace('_', '-')}:{version.version}"
    
    mlflow_build_docker(version.source, new_name, env)
    res = docker_push(new_name)

    return JSONResponse({"result": res})

