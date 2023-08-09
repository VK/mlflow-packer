import os
from fastapi import FastAPI, Request, APIRouter
from fastapi.responses import RedirectResponse
import numpy as np
from mlflow.pyfunc import load_model
from pydantic import BaseModel, create_model
from mlflow.types import Schema 
from starlette.middleware.base import BaseHTTPMiddleware
import json
from inspect import signature
import subprocess


base_path = os.environ.get('BASE_PATH', '/')
if base_path[0] != "/":
    base_path = "/" + base_path
app = FastAPI(root_path = base_path)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url=f'{base_path}/docs')

model = load_model(".")


input_example = None
try:
    with open("input_example.json", "r") as f:
        input_example = json.load(f)
except:
    print("No input example")


MLFLOW_SIGNATURE_TO_NUMPY_TYPE_MAP = {
    "boolean": bool,
    "integer": int,
    "long": int,
    "int32": int,
    "int64": int,
    "double": float,
    "float": float,
    "float32": float,
    "float64": float,
    "string": str,
    "binary": bytes,
    "tensor": list
}


def build_input_model(schema: Schema) -> BaseModel:
    fields = {
        item["name"]: (
            MLFLOW_SIGNATURE_TO_NUMPY_TYPE_MAP.get(item["type"]),
            ...,
        )
        for item in schema.to_dict()
    }
    return create_model("Inputs", **fields)  # type: ignore


def build_output_model(schema: Schema) -> BaseModel:
    fields = {
        item["name"]: (
            MLFLOW_SIGNATURE_TO_NUMPY_TYPE_MAP.get(item["type"]),
            ...,
        )
        for item in schema.to_dict()
    }
    return create_model("Outputs", **fields) # type: ignore



class Request(BaseModel):
    inputs: build_input_model(model.metadata.get_input_schema())  # type: ignore

class Response(BaseModel):
    outputs: build_output_model(model.metadata.get_output_schema())  # type: ignore
    message: str = "?"
    model_id: str = model.metadata.run_id

input_types = {item["name"]:item["tensor-spec"]["dtype"] if "tensor-spec" in item else item["type"] for item in model.metadata.get_input_schema().to_dict()}
output_keys = [item["name"] for item in model.metadata.get_output_schema().to_dict()]


def predictor(request: Request) -> Response:
    inputs = {k: np.array(v).astype(input_types[k]) for k, v in request.inputs.dict().items()}

    error = False
    message = "OK"
    data = None

    try:
        data = model.predict(inputs)
    except:
        error = True

    if error:
        print("reload model")
        model = load_model(".")

        try:
            data = model.predict(inputs)
        except Exception as ex:
            message = str(ex)
            pid = os.getpid()
            print(f"Kill worker {pid}")
            # os.kill(pid, 9)
            subprocess.Popen(f"/bin/sleep 1 && /bin/kill -9 {pid} ",  start_new_session=True, shell=True)


    
    if isinstance(data, list):
        outputs = {k: v.tolist() for k, v in zip(output_keys, data)}
    elif isinstance(data, dict):
        outputs = {k: v.tolist() for k, v in data.items()}
    else:
        outputs = {k: []  for k in output_keys}
    return Response(outputs=outputs, message=message)

response_model = signature(predictor).return_annotation

app.add_api_route(
    "/invocations",
    predictor,
    response_model=response_model,
    methods=["POST"],
)


if input_example:
    # add health check endpoint if input_example is available

    class HealthResponse(BaseModel):
        result: str
    def health() -> HealthResponse:
        inputs = Request(inputs=input_example["inputs"])
        predictor(inputs)
        return HealthResponse(result="OK")
    health_response_model = signature(health).return_annotation
    app.add_api_route(
        "/health",
        health,
        response_model=health_response_model,
        methods=["GET"],
    )