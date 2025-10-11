# marathon
本项目为2025编程马拉松项目

### 基本环境信息

python==3.12
pytorch==2.8.0+cu128

### vLLM

python -m vllm.entrypoints.openai.api_server --model /root/autodl-tmp/Qwen3-Embedding-4B --task embed --trust-remote-code --port 8000 --host 0.0.0.0 --gpu-memory-utilization 0.5 --max-model-len 4096

### packages
!pip install llama-index chromadb transformers sentence-transformers 

!pip install llama-index-embeddings-huggingface llama-index-llms-huggingface llama-index-postprocessor-sbert-rerank 

!pip install llama-index-retrievers-bm25 llama-index-vector-stores-chroma

!pip install llama-index-readers-file unstructured[all-docs] pymupdf jieba

!pip install llama-index-vector-stores-milvus llama-index-llms-dashscope llama-index-postprocessor-cohere-rerank cohere llama-index-embeddings-openai
