import os
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from pymilvus import MilvusClient
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError, ConnectionError
import json
import requests
import time
from datetime import datetime
import logging

# -------------------------- 1. 基礎配置（用戶可根據實際環境調整） --------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# RAG核心配置
RAG_CONFIG = {
    "milvus_db_path": "./milvus_data.db",
    "milvus_collection_name": "testchunks",
    "embedding_model_path": "./embedding-model",
    "reranker_model_path": "./reranker-model",
    "es_url": "https://localhost:9200",
    "es_username": "elastic",
    "es_password": "vSCQnhBXoox0sRo7-U1x",
    "es_index_name": "chunk_documents",
    "max_seq_length": 512,
    "rag_top_k": 3,  # 檢索返回的top相關結果數
    "doc_preview_length": 1000,  # 文檔內容預覽長度（固定300字）
    "es_connection_timeout": 5,  # Elasticsearch連接超時時間（秒）
    "es_retry_count": 2  # Elasticsearch連接重試次數
}

#大模型配置
LLM_CONFIG = {
    "api_url": "your-llm-url",
    "api_key": "your-llm-apikey",
    "model_name": "deepseek-ai/DeepSeek-V3.1",
    "max_tokens": 2000,
    "temperature": 0.7,
    "max_retries": 3,
    "retry_delay": 5
}

# -------------------------- 2. 混合RAG檢索系統（新增ES服務檢測） --------------------------
class HybridRAGSearcher:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"RAG檢索系統使用設備: {self.device}")

        self.embedding_tokenizer, self.embedding_model = self._init_embedding_model()
        self.reranker_tokenizer, self.reranker_model = self._init_reranker_model()
        self.milvus_client = self._init_milvus()
        
        # 新增：ES服務檢測與初始化
        self.es_client, self.es_available = self._init_elasticsearch_with_detection()
        
        # 根據ES可用性調整檢索策略
        if self.es_available:
            logger.info("混合RAG檢索系統初始化完成（Milvus + Elasticsearch）")
        else:
            logger.warning("混合RAG檢索系統初始化完成（僅Milvus，Elasticsearch不可用）")

    def _init_embedding_model(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config["embedding_model_path"])
            model = AutoModel.from_pretrained(self.config["embedding_model_path"]).to(self.device)
            model.eval()
            logger.info("BCE嵌入模型加載成功")
            return tokenizer, model
        except Exception as e:
            logger.error(f"嵌入模型加載失敗: {e}")
            raise

    def _init_reranker_model(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config["reranker_model_path"])
            model = AutoModelForSequenceClassification.from_pretrained(self.config["reranker_model_path"]).to(self.device)
            model.eval()
            logger.info("BCE重排序模型加載成功")
            return tokenizer, model
        except Exception as e:
            logger.error(f"重排序模型加載失敗: {e}")
            raise

    def _init_milvus(self):
        try:
            client = MilvusClient(self.config["milvus_db_path"])
            collections = client.list_collections()
            if self.config["milvus_collection_name"] not in collections:
                raise ValueError(f"Milvus集合 '{self.config['milvus_collection_name']}' 不存在")
            logger.info("Milvus連接成功")
            return client
        except Exception as e:
            logger.error(f"Milvus初始化失敗: {e}")
            raise

    def _init_elasticsearch_with_detection(self):
        """
        新增：帶檢測功能的Elasticsearch初始化
        Returns:
            tuple: (es_client, is_available)
        """
        max_retries = self.config.get("es_retry_count", 2)
        timeout = self.config.get("es_connection_timeout", 5)
        
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"正在檢測Elasticsearch服務（第{attempt + 1}/{max_retries + 1}次）...")
                
                # 創建ES客戶端
                client = Elasticsearch(
                    [self.config["es_url"]],
                    basic_auth=(self.config["es_username"], self.config["es_password"]),
                    verify_certs=False,
                    ssl_show_warn=False,
                    request_timeout=timeout,
                    retry_on_timeout=False
                )
                
                # 測試連接
                if client.ping():
                    # 進一步檢查索引是否存在
                    if client.indices.exists(index=self.config["es_index_name"]):
                        logger.info("✅ Elasticsearch服務檢測成功，索引存在")
                        return client, True
                    else:
                        logger.warning(f"⚠️  Elasticsearch服務可達，但索引 '{self.config['es_index_name']}' 不存在")
                        return client, False
                else:
                    logger.warning(f"❌ Elasticsearch ping失敗（第{attempt + 1}次）")
                    
            except (ConnectionError, Exception) as e:
                logger.warning(f"❌ Elasticsearch連接失敗（第{attempt + 1}次）: {str(e)[:100]}...")
                
                # 如果不是最後一次嘗試，等待後重試
                if attempt < max_retries:
                    time.sleep(1)
        
        # 所有嘗試失敗
        logger.warning("🔄 Elasticsearch服務不可用，將僅使用Milvus進行向量檢索")
        return None, False

    def _generate_embedding(self, texts: List[str]) -> np.ndarray:
        with torch.no_grad():
            inputs = self.embedding_tokenizer(
                texts, padding=True, truncation=True, max_length=self.config["max_seq_length"], return_tensors="pt"
            ).to(self.device)
            outputs = self.embedding_model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            return (cls_emb / cls_emb.norm(dim=1, keepdim=True)).cpu().numpy()

    def _vector_search(self, query: str) -> List[Dict]:
        try:
            query_emb = self._generate_embedding([query])[0].tolist()
            
            # 如果ES不可用，增加Milvus檢索數量以補償
            search_limit = self.config["rag_top_k"] * 2
            if not self.es_available:
                search_limit = self.config["rag_top_k"] * 4  # 增加檢索數量
                
            results = self.milvus_client.search(
                collection_name=self.config["milvus_collection_name"],
                anns_field="vector",
                data=[query_emb],
                limit=search_limit,
                output_fields=["text", "folder", "file", "timestamp"]
            )
            return [
                {
                    "id": f"milvus_{hit['id']}",
                    "text": hit["entity"]["text"],
                    "source": f"{hit['entity']['folder']}/{hit['entity']['file']}",
                    "full_source_path": os.path.abspath(f"{hit['entity']['folder']}/{hit['entity']['file']}"),
                    "score": 1.0 / (1.0 + hit["distance"]),
                    "search_type": "向量檢索",
                    "timestamp": hit["entity"].get("timestamp", "")
                }
                for hit in results[0]
            ]
        except Exception as e:
            logger.error(f"向量搜索失敗: {e}")
            return []

    def _keyword_search(self, query: str) -> List[Dict]:
        """修改：只在ES可用時執行關鍵詞搜索"""
        if not self.es_available:
            logger.debug("Elasticsearch不可用，跳過關鍵詞檢索")
            return []
            
        try:
            response = self.es_client.search(
                index=self.config["es_index_name"],
                body={"query": {"match": {"content": query}}, "size": self.config["rag_top_k"] * 2}
            )
            return [
                {
                    "id": f"es_{hit['_id']}",
                    "text": hit["_source"]["content"],
                    "source": f"{hit['_source']['folder']}/{hit['_source']['filename']}",
                    "full_source_path": os.path.abspath(f"{hit['_source']['folder']}/{hit['_source']['filename']}"),
                    "score": hit["_score"] / 100.0,
                    "search_type": "關鍵詞檢索",
                    "timestamp": hit["_source"].get("import_time", "")
                }
                for hit in response["hits"]["hits"]
            ]
        except Exception as e:
            logger.error(f"關鍵詞搜索失敗: {e}")
            # ES搜索失敗時，標記為不可用
            self.es_available = False
            logger.warning("Elasticsearch搜索失敗，後續將僅使用Milvus")
            return []

    def _merge_deduplicate(self, vector_res: List[Dict], keyword_res: List[Dict]) -> List[Dict]:
        seen = set()
        merged = []
        for res in vector_res + keyword_res:
            text_key = res["text"][:100]
            if text_key not in seen:
                seen.add(text_key)
                merged.append(res)
        
        # 根據ES可用性調整結果數量
        result_count = self.config["rag_top_k"] * 3
        if not self.es_available:
            result_count = self.config["rag_top_k"] * 2  # 僅向量檢索時適當減少
            
        return sorted(merged, key=lambda x: x["score"], reverse=True)[:result_count]

    def _rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        if not results:
            return []
        try:
            max_doc_len = self.config["max_seq_length"] - len(self.reranker_tokenizer.tokenize(query)) - 3
            pairs = [[query, res["text"][:max_doc_len]] for res in results]
            
            with torch.no_grad():
                inputs = self.reranker_tokenizer(
                    pairs, padding=True, truncation=True, max_length=self.config["max_seq_length"], return_tensors="pt"
                ).to(self.device)
                scores = torch.sigmoid(self.reranker_model(**inputs).logits.view(-1,)).cpu().tolist()
            
            for res, score in zip(results, scores):
                res["rerank_score"] = score
            return sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:self.config["rag_top_k"]]
        except Exception as e:
            logger.error(f"重排序失敗: {e}")
            return sorted(results, key=lambda x: x["score"], reverse=True)[:self.config["rag_top_k"]]

    def search(self, query: str) -> List[Dict]:
        search_mode = "混合檢索（向量+關鍵詞）" if self.es_available else "純向量檢索"
        logger.info(f"開始RAG檢索（{search_mode}），查詢: {query[:50]}...")
        
        start = datetime.now()
        vector_res = self._vector_search(query)
        keyword_res = self._keyword_search(query)  # 內部會檢查ES可用性
        merged_res = self._merge_deduplicate(vector_res, keyword_res)
        final_res = self._rerank(query, merged_res)
        
        # 補充檢索耗時和檢索模式信息
        retrieval_time = (datetime.now() - start).total_seconds()
        for res in final_res:
            res["retrieval_time"] = retrieval_time
            res["search_mode"] = search_mode
            
        logger.info(f"RAG檢索完成（{search_mode}），獲取 {len(final_res)} 條相關結果，耗時 {retrieval_time:.2f} 秒")
        return final_res

    def get_service_status(self) -> Dict[str, Any]:
        """新增：獲取服務狀態信息"""
        return {
            "milvus_available": True,  # Milvus在初始化時已確保可用
            "elasticsearch_available": self.es_available,
            "search_mode": "混合檢索（向量+關鍵詞）" if self.es_available else "純向量檢索"
        }

    def close(self):
        if self.milvus_client:
            self.milvus_client.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("RAG檢索系統資源已釋放")

# -------------------------- 3. 文檔信息格式化（新增服務狀態展示） --------------------------
def format_detailed_documents(rag_results: List[Dict], preview_length: int = 300) -> str:
    """
    格式化檢索文檔詳情：包含完整路徑（相對+絕對）、檢索類型、分數、300字內容
    """
    if not rag_results:
        return "【未檢索到相關文檔】\n"
    
    # 獲取檢索模式（從第一個結果中提取）
    search_mode = rag_results[0].get("search_mode", "未知檢索模式")
    detailed_str = f"### 詳細檢索文檔（共{len(rag_results)}條，檢索模式：{search_mode}）\n"
    
    for idx, doc in enumerate(rag_results, 1):
        # 處理文檔內容
        doc_text = doc["text"].strip()
        if len(doc_text) <= preview_length:
            preview_text = doc_text
            ellipsis = ""
        else:
            preview_text = doc_text[:preview_length]
            end_symbols = ["。", "！", "？", "；", "】", ")", "}"]
            last_symbol_idx = max([preview_text.rfind(s) for s in end_symbols if s in preview_text], default=-1)
            if last_symbol_idx != -1 and last_symbol_idx > preview_length * 0.7:
                preview_text = preview_text[:last_symbol_idx + 1]
            ellipsis = "..."
        
        # 拼接文檔詳情
        detailed_str += f"""
#### 文檔{idx}
- **來源路徑（相對路徑）**: {doc["source"]}
- **完整路徑（絕對路徑）**: {doc["full_source_path"]}
- **檢索類型**: {doc["search_type"]}
- **相關性分數**: {doc.get("rerank_score", doc["score"]):.4f}（分數越高越相關）
- **文檔時間戳**: {doc["timestamp"] if doc["timestamp"] else "未記錄"}
- **300字內容預覽**: {preview_text}{ellipsis}

{"-"*50}
"""
    return detailed_str

# -------------------------- 4. 大模型調用與結果整合 --------------------------
def format_rag_for_llm(rag_results: List[Dict]) -> str:
    """給大模型的參考文檔（簡潔版）"""
    if not rag_results:
        return "【無相關參考文檔】"
    llm_ref = "【相關參考文檔】\n"
    for i, res in enumerate(rag_results, 1):
        preview = res["text"]
        llm_ref += f"{i}. 來源：{res['source']} | 內容：{preview}\n"
    return llm_ref

def call_deepseek_v31(prompt: str, system_prompt: str) -> str:
    """調用DeepSeek-V3.1大模型生成回答"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_CONFIG['api_key']}"
    }
    payload = {
        "model": LLM_CONFIG["model_name"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": LLM_CONFIG["max_tokens"],
        "temperature": LLM_CONFIG["temperature"],
        "stream": False
    }

    for retry in range(LLM_CONFIG["max_retries"]):
        try:
            response = requests.post(LLM_CONFIG["api_url"], json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            result = response.json()
            if "choices" in result and result["choices"]:
                return result["choices"][0]["message"]["content"].strip()
            else:
                raise ValueError(f"大模型返回格式異常: {result}")
        except Exception as e:
            logger.error(f"大模型調用失敗（第{retry+1}/{LLM_CONFIG['max_retries']}次）: {e}")
            if retry < LLM_CONFIG["max_retries"] - 1:
                time.sleep(LLM_CONFIG["retry_delay"])
    return "抱歉，大模型調用多次失敗，請稍後重試。"

def rag_qa_pipeline(query: str, rag_searcher: HybridRAGSearcher) -> str:
    """端到端RAG問答流程"""
    # 1. RAG檢索獲取結果
    rag_results = rag_searcher.search(query)
    # 2. 生成大模型所需的簡潔參考
    llm_ref = format_rag_for_llm(rag_results)
    # 3. 生成用戶所需的詳細文檔詳情
    detailed_docs = format_detailed_documents(
        rag_results, 
        preview_length=RAG_CONFIG["doc_preview_length"]
    )

    # 4. 大模型Prompt
    system_prompt = """你是基於檢索增強（RAG）的專業問答助手，嚴格遵循：
1. 必須優先使用【相關參考文檔摘要】中的信息回答，每個結論需標註對應文檔編號（如【參考1】）；
2. 若參考文檔信息不足，補充知識時需標註"注：以下內容基於模型自身知識補充"；
3. 回答邏輯清晰，分點說明（適用時），不編造信息，無法回答則直接說明。"""
    
    user_prompt = f"""用戶問題：{query}

{llm_ref}

請基於上述參考文檔，回答用戶問題。"""

    # 5. 調用大模型生成回答
    logger.info("調用DeepSeek-V3.1生成回答...")
    answer = call_deepseek_v31(user_prompt, system_prompt)

    # 6. 獲取服務狀態
    service_status = rag_searcher.get_service_status()

    # 7. 整合最終輸出
    final_output = f"""
# RAG問答結果
## 一、用戶問題
{query}

## 二、AI回答
{answer}

## 三、檢索統計信息
- 檢索到相關文檔數量：{len(rag_results)} 條
- 總檢索耗時：{rag_results[0]["retrieval_time"]:.2f} 秒（含向量檢索、關鍵詞檢索、重排序）
- 檢索模式：{service_status["search_mode"]}
- 服務狀態：Milvus ✅ | Elasticsearch {'✅' if service_status['elasticsearch_available'] else '❌'}

## 四、{detailed_docs}

> 注：詳細文檔中的"完整路徑（絕對路徑）"可直接複製到文件管理器打開，查看文檔全文。
"""
    return final_output

# -------------------------- 5. 主函數（新增服務狀態提示） --------------------------
def main():
    rag_searcher = None
    try:
        rag_searcher = HybridRAGSearcher(RAG_CONFIG)
        service_status = rag_searcher.get_service_status()
        
        print("="*100)
        print("===== 帶詳細文檔展示的RAG問答系統（基於DeepSeek-V3.1） =====")
        print(f"當前檢索模式：{service_status['search_mode']}")
        print(f"服務狀態：Milvus ✅ | Elasticsearch {'✅' if service_status['elasticsearch_available'] else '❌'}")
        if not service_status['elasticsearch_available']:
            print("⚠️  注意：Elasticsearch服務不可用，將僅使用Milvus進行向量檢索")
        print("說明：輸入問題後，將返回AI回答+檢索文檔詳情（含300字內容+文件路徑）")
        print("輸入'退出'或'quit'可結束程序")
        print("="*100)
        
        while True:
            user_query = input("\n請輸入你的問題：").strip()
            if user_query.lower() in ["退出", "quit", "exit"]:
                print("\n感謝使用，程序已退出！")
                break
            if not user_query:
                print("請輸入有效的問題，不能為空！")
                continue
            
            # 執行問答流程並打印結果
            print("\n" + "="*50)
            print(f"正在處理問題：{user_query}")
            print("步驟1/2：RAG檢索相關文檔...")
            print("步驟2/2：調用大模型生成回答...")
            print("="*50)
            
            final_result = rag_qa_pipeline(user_query, rag_searcher)
            print("\n" + "="*100)
            print("最終結果：")
            print(final_result)
            print("="*100)
            
    except Exception as e:
        logger.error(f"系統初始化失敗: {e}")
        print(f"\n錯誤：系統初始化失敗，原因：{str(e)}")
        print("請檢查Milvus服務是否啟動，或配置參數是否正確。")
    finally:
        if rag_searcher:
            rag_searcher.close()

if __name__ == "__main__":
    main()
