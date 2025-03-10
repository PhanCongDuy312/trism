from typing import Any
from trism_here.inout import Inout
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.grpc import InferenceServerClient as GrpcClient
from tritonclient.http import InferenceServerClient as HttpClient
import tritonclient.grpc.aio as aio_grpcclient
import numpy as np
import json


# NOTE: attrdict broken in python 3.10 and not maintained.
# https://github.com/wallento/wavedrompy/issues/32#issuecomment-1306701776
try:
  from attrdict import AttrDict
except ImportError:
  # Monkey patch collections
  import collections
  import collections.abc
  for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))
  from attrdict import AttrDict


def protoclient(grpc: bool = True) -> Any:
  return grpcclient if grpc else httpclient


def serverclient(url: str, grpc: bool = True, async_mode:bool = False, concurrency: int = 10, *args, **kwds) -> GrpcClient | HttpClient:
  if async_mode == False:
    return GrpcClient(url=url, *args, **kwds) if grpc else \
      HttpClient(url=url, concurrency=concurrency, *args, **kwds)
  else:
    return aio_grpcclient.InferenceServerClient(url=url, verbose=False)

def inout(serverclient, model: str, version: str = "", *args, **kwds) -> Any:
  meta = serverclient.get_model_metadata(model, version, *args, **kwds)
  conf = serverclient.get_model_config(model, version, *args, **kwds)
  if isinstance(serverclient, GrpcClient):
    conf = conf.config
  else:
    meta, conf = AttrDict(meta), AttrDict(conf)
  inputs = [Inout(name=inp.name, shape=inp.shape, dtype=inp.datatype) for inp in meta.inputs]
  outputs = [Inout(name=out.name, shape=out.shape, dtype=out.datatype) for out in meta.outputs]
  return inputs, outputs

def create_request(model_name, prompt, stream, request_id, sampling_parameters, exclude_input_in_output, send_parameters_as_tensor=True):
    inputs = []
    prompt_data = np.array([prompt.encode("utf-8")], dtype=np.object_)
    try:
        inputs.append(grpcclient.InferInput("text_input", [1], "BYTES"))
        inputs[-1].set_data_from_numpy(prompt_data)
    except Exception as error:
        print(f"Encountered an error during request creation: {error}")

    stream_data = np.array([stream], dtype=bool)
    inputs.append(grpcclient.InferInput("stream", [1], "BOOL"))
    inputs[-1].set_data_from_numpy(stream_data) 

    if send_parameters_as_tensor:
        sampling_parameters_data = np.array([json.dumps(sampling_parameters).encode("utf-8")], dtype=np.object_)
        inputs.append(grpcclient.InferInput("sampling_parameters", [1], "BYTES"))
        inputs[-1].set_data_from_numpy(sampling_parameters_data)

    inputs.append(grpcclient.InferInput("exclude_input_in_output", [1], "BOOL"))
    inputs[-1].set_data_from_numpy(np.array([exclude_input_in_output], dtype=bool))

    outputs = []
    outputs.append(grpcclient.InferRequestedOutput("text_output"))

    return {
        "model_name": model_name,
        "inputs": inputs,
        "outputs": outputs,
        "request_id": str(request_id),
        "parameters": sampling_parameters,
    }