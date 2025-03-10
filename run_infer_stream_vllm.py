from trism_here import TritonModel

# Create triton model.
prompt = ["Phổi hai bên kém sáng, đám mờ không đều kèm xơ hóa đáy phổi trái, mờ kính rải rác, rốn phổi tăng đậm nhẹ. Bờ vòm hoành phải đều, tù góc sườn hoành trái. Bóng mờ tim không to, các cung tim trong giới hạn sinh lý. Hãy đưa ra kết luận"]

model = TritonModel(
  model="deepseek_qwen_awq_1_5b",     # Model name.
  version=0,            # Model version.
  url="localhost:1235", # Triton Server URL.
  grpc=True,
  async_mode=True,
  stream_timeout=None,
  offset=0,
  iterations=1,
  streaming_mode=False,
  exclude_inputs_in_outputs=False,
  lora_name=None,
  input_prompts = prompt 
)
result = model.run_async()

print("result ne:",result)