import numpy as np
from trism import client
from functools import partial
import time
from tritonclient.utils import InferenceServerException
import sys


class TritonModel:

  @property
  def model(self) -> str:
    return self._model
  
  @property
  def version(self) -> str:
    return self._version
  
  @property
  def url(self) -> str:
    return self._url
  
  @property
  def grpc(self) -> bool:
    return self._grpc
  
  @property
  def inputs(self):
    return self._inputs
  
  @property
  def outputs(self):
    return self._outputs

  def __init__(self, model: str, version: int, url: str, grpc: bool = True) -> None:
    self._url = url
    self._grpc = grpc
    self._model = model
    self._version = str(version) if version > 0 else ""
    self._protoclient = client.protoclient(self.grpc)
    self._serverclient = client.serverclient(self.url, self.grpc)
    self._inputs, self._outputs = client.inout(self._serverclient, self.model, self.version)
  
  def run(self, data: list[np.array]):
    inputs = [self.inputs[i].make_input(self._protoclient, data[i]) for i in range(len(self.inputs))]
    outputs = [output.make_output(self._protoclient) for output in self.outputs]
    results = self._serverclient.infer(self.model, inputs, self.version, outputs)
    return {output.name: results.as_numpy(output.name) for output in self.outputs}

  def run_async(self, data: list[np.array], timeout):
    """
    MUST use GRPC for async infer
    timeout: None or int
    """   
    inputs = [self.inputs[i].make_input(self._protoclient, data[i]) for i in range(len(self.inputs))]
    outputs = [output.make_output(self._protoclient) for output in self.outputs]
    result = async_infer(self._serverclient, self.model, self.version, inputs, outputs, timeout)
    return result

def callback(user_data, result, error):
    if error:
        user_data.append(error)
    else:
        user_data.append(result)
 
def async_infer(tritonclient, model, version, inputs, outputs, timeout):
  # List to hold the results of inference.
  user_data = []

  # Inference call
  tritonclient.async_infer(
      model_name=model,
      model_version=version,
      inputs=inputs,
      callback=partial(callback, user_data),
      outputs=outputs,
      client_timeout=timeout,
  )

  # Wait until the results are available in user_data
  time_out = 10
  while (len(user_data) == 0) and time_out > 0:
      time_out = time_out - 1
      time.sleep(1)

  # Display and validate the available results
  if len(user_data) == 1:
      # Check for the errors
      if type(user_data[0]) == InferenceServerException:
          print(user_data[0])
          sys.exit(1)
      print("PASS: Async infer")
      return {output.name: user_data[0].as_numpy(output.name) for output in outputs}

      
      