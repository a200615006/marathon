# marathon
本项目为2025编程马拉松项目

### 基本环境信息

python==3.12
pytorch==2.8.0+cu128

### vLLM

python -m vllm.entrypoints.openai.api_server --model /root/autodl-tmp/Qwen3-Embedding-4B --task embed --trust-remote-code --port 8000 --host 0.0.0.0 --gpu-memory-utilization 0.5 --max-model-len 4096
