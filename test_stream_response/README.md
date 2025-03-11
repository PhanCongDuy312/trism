# Python TRISM
Change model name in model.json to use your model
```bash
docker build . -t async_stream_qwen_base

docker run --gpus all -it --rm -p 1234:8000 -p 1235:8001 -p 1238:8002 async_stream_qwen_base
```
