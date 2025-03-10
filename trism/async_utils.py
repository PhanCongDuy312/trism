import argparse
import asyncio
import json
import sys
import time
import nest_asyncio
nest_asyncio.apply()  # Allow nested event loops
import numpy as np
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import *

async def triton_run_async(self):
    sampling_parameters = {
        "temperature": "0.1",
        "top_p": "0.95",
        "max_tokens": "100",
    }
    exclude_input_in_output = self._flags.exclude_inputs_in_outputs
    if self._flags.lora_name is not None:
        sampling_parameters["lora_name"] = self._flags.lora_name

    prompts = self._flags.input_prompts
    print("Using provided prompts from event input...")

    success = await self.process_stream(prompts, sampling_parameters, exclude_input_in_output)

    final_result = ""
    for id in self._results_dict.keys():
        for result in self._results_dict[id]:
            final_result += result.decode("utf-8")
            
        final_result += "\n=========\n\n"
    print("Processed results.")

    if self._flags.verbose:
        print(f"\nResults:\n{final_result}")
    if success:
        print("PASS: vLLM example")
    else:
        print("FAIL: vLLM example")
    return final_result