import numpy as np
from trism import client

import sys
import asyncio
import nest_asyncio
nest_asyncio.apply() 
from tritonclient.utils import *

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

    def __init__(self, model: str, version: int, url: str, grpc: bool = True, async_mode: bool = False, **kwagrs) -> None:
        self._url = url
        self._grpc = grpc
        self.async_mode = async_mode # Async mode just work with grpc only.
        self._model = model
        self._version = str(version) if version > 0 else ""
        self._protoclient = client.protoclient(self.grpc)
        self._serverclient = client.serverclient(self.url, self.grpc, self.async_mode)
        if self.async_mode == False:
            self._inputs, self._outputs = client.inout(self._serverclient, self.model, self.version)
        else:
            self._results_dict = {}
            
            for key, value in kwagrs.items():
                setattr(self, key, value) 

            _ = self.check_runasync_args()

    def run(self, data: list[np.array]):
        inputs = [self.inputs[i].make_input(self._protoclient, data[i]) for i in range(len(self.inputs))]
        outputs = [output.make_output(self._protoclient) for output in self.outputs]
        results = self._serverclient.infer(self.model, inputs, self.version, outputs)
        return {output.name: results.as_numpy(output.name) for output in self.outputs}
    
    
    # When call this function, use "await" to call.
    def run_async(self):
        results = asyncio.run(self.triton_run_async())
        return results


    # Async utils
    def check_runasync_args(self):
        """
        The **kwargs attributes should follow the structure:
        {
            "stream_timeout": <float>,
            "offset": <int>,
            "iterations": <int>,
            "streaming_mode": <bool>,
            "exclude_inputs_in_outputs": <bool>,
            "lora_name": <optional lora name>
        }
        """
        required_attributes = {
            "stream_timeout": (float, type(None)),
            "offset": int,
            "iterations": int,
            "streaming_mode": bool,
            "exclude_inputs_in_outputs":bool,
            "lora_name": str,
            "temperature": float,
            "top_p": float,
            "max_tokens": int
        }

        for attribute, expected_types in required_attributes.items():
            if not hasattr(self, attribute):
                raise ValueError(f"Error: '{attribute}' is missing!")

            # Get the attribute value
            attribute_value = getattr(self, attribute)

            # Check if the attribute's type matches the expected type(s)
            if not isinstance(attribute_value, expected_types):
                raise TypeError(f"Error: '{attribute}' should be of type {expected_types}, but got {type(attribute_value)}!")

        print("All extra attributes are present and have the correct type.")
        
    async def triton_run_async(self):
        sampling_parameters = {
            "temperature": str(self.temperature),
            "top_p": str(self.top_p),
            "max_tokens": str(self.max_tokens)
        }
        exclude_input_in_output = self.exclude_inputs_in_outputs
        prompts = self.input_prompts
        print("Using provided prompts from event input...")

        results_dict, success = await self.process_stream(prompts, sampling_parameters, exclude_input_in_output)

        final_result = ""
        for id in results_dict.keys():
            for result in results_dict[id]:
                final_result += result.decode("utf-8")
            final_result += "\n=========\n\n"
        print("Processed results.")

        print(f"\nResults:\n{final_result}")
        if success:
            print("PASS: vLLM example")
        else:
            print("FAIL: vLLM example")
        return final_result
            
    async def process_stream(self, prompts, sampling_parameters, exclude_input_in_output):
        results_dict = {}
        success = True
        async for response in self.stream_infer(prompts, sampling_parameters, exclude_input_in_output):
            result, error = response
            print("response:", response)
            if error:
                print(f"Encountered error while processing: {error}")
                success = False
            else:
                output = result.as_numpy("text_output")
                for i in output:
                    result_text = result.get_response().id
                    results_dict[result_text].append(i)
        return results_dict, success
        
    async def stream_infer(self, prompts, sampling_parameters, exclude_input_in_output):
        try:
            response_iterator = self._serverclient.stream_infer(
                inputs_iterator=self.async_request_iterator(prompts, sampling_parameters, exclude_input_in_output),
                stream_timeout=self.stream_timeout,
            )

            async for response in response_iterator:
                yield response
        except InferenceServerException as error:
            print(error)
            sys.exit(1)

    async def async_request_iterator(self, prompts, sampling_parameters, exclude_input_in_output):
        try:
            for iter in range(self.iterations):
                for i, prompt in enumerate(prompts):
                    prompt_id = self.offset + (len(prompts) * iter) + i
                    self._results_dict[str(prompt_id)] = []
                    yield client.create_request(
                        self.model,
                        prompt,
                        self.streaming_mode,
                        prompt_id,
                        sampling_parameters,
                        exclude_input_in_output,
                    )
        except Exception as error:
            print(f"Caught an error in the request iterator: {error}")