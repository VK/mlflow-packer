from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
from inspect import signature


import os
import shutil
import mlflow
from mlflow.tracking import MlflowClient
import docker
import configparser

import requests

# load config once
config = configparser.ConfigParser()
config.read('/default.cfg')
BASE_IMAGE_NAME = "mlflow-packer-base"

initial_wd = os.getcwd()

def get_mflow_client():
    
    if 'DBR_TOKEN' in os.environ:
        # get token from ENVIRONMENT if possible
        token = os.environ['DBR_TOKEN']
    else:
        token = config.get('Databricks', 'TOKEN')

    registry = config.get('Databricks', 'REGISTRY')
    user = config.get('Databricks', 'USER')

    os.environ['DATABRICKS_HOST'] = registry
    os.environ['DATABRICKS_TOKEN'] = token
    os.environ["MLFLOW_TRACKING_TOKEN"] = token
    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
    mlflow.set_tracking_uri(registry)

    client = MlflowClient()
    print("init mlflow client")
    return client


def get_azure_access_token(scope):
    base_url = config.get('Docker', 'HOST')
    user = config.get('Docker', 'USER')  # Typically the ACR admin user or service principal username
    token = config.get('Docker', 'TOKEN')  # ACR admin password or token

    # Get a valid auth token
    auth_url = f"https://{base_url}/oauth2/token"
    auth_data = {
        'grant_type': 'password',
        'service': base_url,
        'scope': scope,
        'username': user,
        'password': token
    }

    auth_res = requests.post(auth_url, data=auth_data)
    if auth_res.ok:
        auth_token = auth_res.json().get('access_token')
        return auth_token
    else:
        raise Exception("Failed to get auth token")

def is_dockerhub():
    """
    Check if the Docker host is Docker Hub.

    Returns:
        bool: True if the Docker host is Docker Hub, False otherwise.
    """
    base_url = config.get('Docker', 'HOST')
    return "docker.io" in base_url

def get_repo_tags(repo):

    repo = repo.replace("_", "-").lower()
    
    base_url = config.get('Docker', 'HOST')
    token = config.get('Docker', 'TOKEN')
    user = config.get('Docker', 'USER')
    org = config.get('Docker', 'ORG')

    # check if base_url suggests docerhub
    if is_dockerhub():

        login_url = f"{base_url}/users/login"
        repo_url = f"{base_url}/repositories/{org}/{repo}/tags"

        tok_req = requests.post(
            login_url, json={"username": user, "password": token})
        token = tok_req.json()["token"]
        headers = {"Authorization": f"JWT {token}"}

        res = requests.get(repo_url, headers=headers)
        if res.ok:
            data = res.json()
            return [el["name"] for el in data["results"]]
        else:
            return []
        
    else:
        # assume azure container registry
        tags_url = f"https://{base_url}/v2/{repo}/tags/list"

        # Create Bearer auth header
        auth_token = get_azure_access_token(f"repository:{repo}:metadata_read")
        headers = {"Authorization": f"Bearer {auth_token}"}

        res = requests.get(tags_url, headers=headers)
        if res.ok:
            data = res.json()
            return data.get('tags', [])
        else:
            return []



def get_hub_image_name(name):
    org = config.get('Docker', 'ORG')
    base_url = config.get('Docker', 'HOST')
    if is_dockerhub():
        return f"{org}/{name}"
    else:
        return f"{base_url}/{name}"


def mlflow_build_docker(source, name, env):
    print(f'mlflow models build-docker -m {source} -n {get_hub_image_name(name)} --env-manager {env}')
    os.system(
        f'mlflow models build-docker -m {source} -n {get_hub_image_name(name)} --env-manager {env}'
    )


def docker_push(name):
    base_url = config.get('Docker', 'HOST')
    token = config.get('Docker', 'TOKEN')
    user = config.get('Docker', 'USER')
    org = config.get('Docker', 'ORG')

    client = docker.from_env()
    if is_dockerhub():
        client.login(username = user, password=token)
    else:
        client.login(username = user, password=token, registry=base_url)

    if is_dockerhub():
        return client.api.push(f"{org}/{name}")
    else:
        return client.api.push(f"{base_url}/{name}")


def docker_pull(name):
    base_url = config.get('Docker', 'HOST')
    token = config.get('Docker', 'TOKEN')
    user = config.get('Docker', 'USER')
    org = config.get('Docker', 'ORG')

    client = docker.from_env()

    if is_dockerhub():
        client.login(username = user, password=token)
    else:
        client.login(username = user, password=token, registry=base_url)

    if is_dockerhub():
        return client.api.pull(f"{org}/{name}")
    else:
        return client.api.pull(f"{base_url}/{name}")


def add_index_url_to_req_file(req_file_name):
    import os
    AZURE_DEV_PAT = os.environ.get('AZURE_DEV_PAT', None)
    if AZURE_DEV_PAT is None:
        print("No AZURE_DEV_PAT found, skipping adding index url to requirements.txt")
        return
    
    index_url = f"https://{AZURE_DEV_PAT}@pkgs.dev.azure.com/OsramDS/_packaging/OsramDS/pypi/simple/"

    with open(req_file_name, 'r') as f:
        original_content = f.read()
    with open(req_file_name, 'w') as f:
        f.write(f"--extra-index-url {index_url}" + '\n')
        f.write(original_content)    



def build_mlflow_packer_base(python_version, tag, req_file_name, modeldir):
    """
    create a base image for to serve a model with all the required dependencies
    """

    org = config.get('Docker', 'ORG')
    base_url = config.get('Docker', 'HOST')
    os.chdir(os.path.dirname(req_file_name))

    if is_dockerhub():
        base_imagename = f"{org}/{BASE_IMAGE_NAME}"
    else:
        base_imagename = f"{base_url}/{BASE_IMAGE_NAME}"

    dockerfile = f"""
FROM python:{python_version}

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt \\
    && pip install uvicorn==0.18.2 protobuf==3.20.* fastapi==0.80.*\\
    && mkdir -p /model

WORKDIR /model
EXPOSE 8080

ENTRYPOINT gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080 --timeout 600
    

"""
    
    with open("baseDockerfile", 'w') as f:
        f.write(dockerfile)
    os.system(
        f'docker build -f baseDockerfile -t {base_imagename}:{tag} .' 
    )


    return docker_push(f'{BASE_IMAGE_NAME}:{tag}')
        
            
        
def build_with_base_image(model, version):
    """build the model server without mlflow, but with a good base base image
    """
    
    import tempfile
    import yaml
    import hashlib

    org = config.get('Docker', 'ORG')
    base_url = config.get('Docker', 'HOST')
    if is_dockerhub():
        base_imagename = f"{org}/{BASE_IMAGE_NAME}"
    else:
        base_imagename = f"{base_url}/{BASE_IMAGE_NAME}"


    os.chdir(initial_wd)
    cwd = os.getcwd()

    with tempfile.TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)
        command = f'mlflow artifacts  download -u {version.source} -d {tmpdirname}'
        print(command)
        os.system(command)

        model_dir = list(os.scandir(tmpdirname))
        if len(model_dir) == 1:
            model_dir = model_dir[0]
        else:
            os.chdir(cwd)
            raise Exception(f"Multiple model dirs downloaded {model_dir}")

        # extract python version
        with open(os.path.join(model_dir, "conda.yaml"), "r") as stream:
            try:
                python_version = [
                    d for d in yaml.safe_load(stream)["dependencies"]
                    if "python" in d
                    ][0]
                python_version = python_version.split("=")[-1]
            except yaml.YAMLError as exc:
                os.chdir(cwd)
                raise Exception("Problem parsing conda.yaml")

        # create requirements hash
        md5_hash = hashlib.md5()
        md5_hash.update(b"8.09.2023")
        req_file_name = os.path.join(model_dir, "requirements.txt") 
        with open(req_file_name,"rb") as f:
            # Read and update hash in chunks of 4K
            for byte_block in iter(lambda: f.read(4096),b""):
                md5_hash.update(byte_block)
        req_hash = md5_hash.hexdigest()

        add_index_url_to_req_file(req_file_name)

        # check if the matching minimal model container is available
        try:
            known_containers = get_repo_tags(base_imagename)
        except:
            known_containers = []

        new_tag = f"{python_version}-{req_hash}"
        
        # compute a new container if needed
        if new_tag not in known_containers:
            res = build_mlflow_packer_base(python_version, new_tag, req_file_name, model_dir)
        else:
            print(f"pull image {base_imagename}:{new_tag}")
            docker_pull(f"{base_imagename}:{new_tag}")

        # inject main.py
        shutil.copyfile("/app/buildtemplate/main.py", os.path.join(model_dir, "main.py"))
        shutil.copyfile("/app/buildtemplate/setup.py", os.path.join(model_dir, "setup.py"))

        os.chdir(os.path.dirname(req_file_name))
        # create dockerfile with the serving
        dockerfile = f"""
        
FROM {base_imagename}:{new_tag}

ENV MODEL_TITLE={model.name.lower().replace('_', '-')}
ENV MODEL_VERSION={version.version}

COPY . /model/
RUN python setup.py

        """
        
        with open("Dockerfile", 'w') as f:
            f.write(dockerfile)

        # build the dockerfile
        new_name = f"{model.name.lower().replace('_', '-')}:{version.version}"
        image_name = get_hub_image_name(new_name)

        os.system(
            f'docker build -f Dockerfile -t {image_name} .' 
        )        

        # publish the container
        res = docker_push(new_name)

    os.chdir(cwd)
    return res




def build_with_tfserving(model, version, tf_version="auto"):
    
    import tempfile
    import hashlib

    org = config.get('Docker', 'ORG')
    os.chdir(initial_wd)
    cwd = os.getcwd()

    with tempfile.TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)
        command = f'mlflow artifacts  download -u {version.source} -d {tmpdirname}'
        print(command)
        os.system(command)

        model_dir = list(os.scandir(tmpdirname))
        if len(model_dir) == 1:
            model_dir = model_dir[0]
        else:
            os.chdir(cwd)
            raise Exception(f"Multiple model dirs downloaded {model_dir}")

       # extract tensorflow version
        if tf_version == "auto":
            with open(os.path.join(model_dir, "requirements.txt"), "r") as file:
                try:
                    tf_version = [
                        d for d in file.readlines()
                        if "tensorflow==" in d
                        ][0]
                    print(tf_version)
                    tf_version = tf_version.split("=")[-1]
                except:
                    os.chdir(cwd)
                    raise Exception("Error parsing requirements for tensorflow")

        dockerfile = f"""
        
FROM tensorflow/serving:{tf_version}

ENV MODEL_NAME {model.name}

COPY {model_dir.name}/data/* /models/{model.name}/01/

        """

        print(dockerfile)

        with open("Dockerfile", 'w') as f:
            f.write(dockerfile)

        # build the dockerfile
        new_name = f"{model.name.lower().replace('_', '-')}:{version.version}-tfserving"
        image_name = get_hub_image_name(new_name)
        os.system(
            f'docker build -f Dockerfile -t {image_name} .' 
        )

        # publish the container
        res = docker_push(new_name)
        
    os.chdir(cwd)
    return res
        

base_path = os.environ.get('BASE_PATH', '/')
if base_path[0] != "/":
    base_path = "/" + base_path

app = FastAPI(
    title="MLflow Packer",
    description="""Build and push mlflow models.""",
    version="0.2.9",
    root_path = base_path
)


@app.get("/", include_in_schema=False)
async def root():
    if base_path == "/":
        return RedirectResponse(url=f'/docs')
    else:
        return RedirectResponse(url=f'{base_path}/docs')



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



@app.get("/image", response_model=List[DockerList])
async def list_docker_models(name: str, ):
    """
    List the available images of a model
    - **name**: model name
    """

    return JSONResponse(get_repo_tags(name.lower()))


class BuildResponse(BaseModel):
    result: str

@app.get("/build", response_model=BuildResponse)
async def build_docker_model(name: str, version: str, env: str = "baseimage", contvers: str = "auto"):
    """
    Build a new model version an push it to the server regitry

    - **name**: model name
    - **version**: the version to build
    - **env**: specify environment manager (local, conda, virtualenv, baseimage, tfserving)
    - **contvers**: container version (e.g 2.14.1)
    """

    # cleanup before start
    os.system("docker system prune -f")

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

    if env == "baseimage":

        res = build_with_base_image(model, version)
        return JSONResponse({"result": res})

    elif env == "tfserving":

        res = build_with_tfserving(model, version, contvers)
        return JSONResponse({"result": res})

    else:

        new_name = f"{model.name.lower().replace('_', '-')}:{version.version}"
        
        mlflow_build_docker(version.source, new_name, env)
        res = docker_push(new_name)

        return JSONResponse({"result": res})


class HealthResponse(BaseModel):
    result: str
def health() -> HealthResponse:
    client = docker.from_env()
    client.ping()
    return HealthResponse(result="OK")
health_response_model = signature(health).return_annotation
app.add_api_route(
    "/health",
    health,
    response_model=health_response_model,
    methods=["GET"],
)
