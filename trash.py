from trism import TritonModel

# Create triton model.
model = TritonModel(
  model="my_model",     # Model name.
  version=0,            # Model version.
  url="localhost:8001", # Triton Server URL.
  grpc=True,
  temp=1
)
_ = model.run_async()