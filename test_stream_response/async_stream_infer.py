import argparse
import os
import sys
import random
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype, InferenceServerException
import uuid
import time
import queue
import statistics
from functools import partial
import json
class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()

def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)

def send_request(model_name, inputs, outputs, request_id):
    tritonclient.async_stream_infer(
        model_name=model_name, inputs=inputs, outputs=outputs, request_id=request_id
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client for batch-enabled AGM model")
    parser.add_argument(
        "--request_count",
        action="store",
        help="Specify the number of requests you want to send to the server",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--max_tokens",
        action="store",
        help="Range of output lengths that can be randomly generated. To always generate the same size, pass the same value twice",
        default=[1, 1],
        nargs=2,
        type=int,
    )
    args = parser.parse_args()

    prompt = "Phổi hai bên kém sáng, đám mờ không đều kèm xơ hóa đáy phổi trái, mờ kính rải rác, rốn phổi tăng đậm nhẹ. Bờ vòm hoành phải đều, tù góc sườn hoành trái. Bóng mờ tim không to, các cung tim trong giới hạn sinh lý. Hãy đưa ra kết luận"
    sampling_parameters = {
        "temperature": "0.1",
        "top_p": "0.95",
        "max_tokens": "100",
    }
    
    prompt_data = np.array([prompt.encode("utf-8")], dtype=np.object_)
    stream_data = np.array([True], dtype=bool)
    sampling_parameters_data = np.array([json.dumps(sampling_parameters).encode("utf-8")], dtype=np.object_)
    exclude_input_in_output_data = np.array([True], dtype=bool)
    
    # Create input tensor
    inputs = []
    inputs.append(grpcclient.InferInput("text_input", [1], "BYTES"))
    inputs[-1].set_data_from_numpy(prompt_data)
    inputs.append(grpcclient.InferInput("stream", [1], "BOOL"))
    inputs[-1].set_data_from_numpy(stream_data)
    inputs.append(grpcclient.InferInput("sampling_parameters", [1], "BYTES"))
    inputs[-1].set_data_from_numpy(sampling_parameters_data)      
    inputs.append(grpcclient.InferInput("exclude_input_in_output", [1], "BOOL"))
    inputs[-1].set_data_from_numpy(np.array(exclude_input_in_output_data))   

    user_data = UserData()
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput("text_output"))
    request_ids = []

    with grpcclient.InferenceServerClient(url="localhost:1235") as tritonclient:

        # Establish stream
        tritonclient.start_stream(
            callback=partial(callback, user_data),
            stream_timeout=1000000000,
        )

        request_start_time_map = {}
        request_end_time_map = {}
        total_time_begin = time.time()
        for request_id in range(0, args.request_count):
            request_ids.append(request_id)
            request_start_time_map[request_id] = time.time()
            print(f"request_start_time: {request_start_time_map[request_id]}")
            send_request("deepseek_qwen_awq_1_5b", inputs, outputs, str(request_id))

        received_requests = {}
        finished_requests = 0
        results_dict = {}
        while finished_requests != args.request_count:
            if not user_data._completed_requests.empty():
                try:
                    result = user_data._completed_requests.get(block=False)
                except Exception:
                    break
                if isinstance(result, InferenceServerException):
                    print(f"Received error from Triton server: {result}")
                    break
                response_id = int(result.get_response().id)
                if response_id in received_requests:
                    print("Old request got output")
                    print(result.as_numpy("text_output"))
                else:
                    print("New request got output")
                    print("final_result",result.as_numpy("text_output"))

                # Check if the response tensor is empty to determine the final response
                if result.as_numpy("OUTPUT") is None or result.as_numpy("OUTPUT").size == 0:
                    finished_requests += 1

        print("Client termination")
