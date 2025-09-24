import os
import numpy as np
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from sentence_transformers import SentenceTransformer
import json
from datetime import datetime
import torch


class MilvusLiteRetriever:
    def __init__(self, db_path="./milvus_data.db", model_path="./embedding-model"):
        self.db_path = db_path
        
        # 加載模型到對應設備
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用設備: {self.device}")
        self.model = SentenceTransformer(model_path).to(self.device)
        self.dimension = 768
        
        self.client = MilvusClient(db_path)
        self.collection_name = "testchunks"
        self.vector_field_name = "vector"
            
    def create_collection_if_not_exists(self, collection_name):
        """創建集合（如果不存在）"""
        self.collection_name = collection_name
        collections = self.client.list_collections()
        print('collection lists:', collections)
        
        if collection_name not in collections:
            # 創建字段架構
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="folder", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="file", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=50)
            ]
            
            # 創建集合架構
            schema = CollectionSchema(fields, description="Text chunks collection")
            
            # 創建新集合
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema
            )
            print(f"創建新集合: {collection_name}")
            
            # 創建索引 - Milvus Lite 只支援 FLAT 索引
            self.create_index()
            return False
        else:
            print(f"集合已存在: {collection_name}")
            return True
    
    def create_index(self):
        """創建向量索引 - Milvus Lite 只支援 FLAT 索引"""
        print("正在創建向量索引（使用 FLAT 索引）...")
        
        # 準備索引參數 - Milvus Lite 只支援 FLAT 索引
        index_params = MilvusClient.prepare_index_params()
        
        index_params.add_index(
            field_name=self.vector_field_name,
            index_type="FLAT",  # Milvus Lite 只支援 FLAT 索引
            index_name="vector_index",
            metric_type="L2",
            params={}  # FLAT 索引不需要額外參數
        )
        
        # 創建索引
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params
        )
        print("FLAT 向量索引創建完成")
    
    def load_chunks(self, base_path):
        """加載文本 chunks"""
        chunks = []
        metadata = []
        
        for folder in sorted(os.listdir(base_path)):
            folder_path = os.path.join(base_path, folder)
            if os.path.isdir(folder_path):
                for file in sorted(os.listdir(folder_path)):
                    if file.endswith('.txt'):
                        file_path = os.path.join(folder_path, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                                if content:
                                    chunks.append(content)
                                    metadata.append({
                                        "folder": folder,
                                        "file": file,
                                        "timestamp": datetime.now().isoformat()
                                    })
                        except UnicodeDecodeError:
                            print(f"跳過無法解碼的文件: {file_path}")
                        except Exception as e:
                            print(f"讀取文件 {file_path} 時出錯: {e}")
        
        print(f"成功加載 {len(chunks)} 個文本 chunks")
        return chunks, metadata
    
    def generate_embeddings(self, chunks, batch_size=32):
        """使用 GPU 加速生成嵌入向量"""
        print("正在生成嵌入向量（使用 GPU 加速）...")
        
        # 設置模型為評估模式
        self.model.eval()
        
        # 分批處理以避免內存不足
        embeddings = []
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        with torch.no_grad():  # 禁用梯度計算以節省內存
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i+batch_size]
                
                # 生成嵌入向量
                batch_embeddings = self.model.encode(
                    batch_chunks,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    device=self.device
                )
                
                embeddings.append(batch_embeddings)
                
                if (i // batch_size) % 10 == 0:  # 每10個batch顯示進度
                    print(f"處理進度: {min(i+batch_size, len(chunks))}/{len(chunks)}")
        
        # 合併所有批次的嵌入向量
        embeddings = np.vstack(embeddings)
        print(f"嵌入向量生成完成，形狀: {embeddings.shape}")
        return embeddings
    
    def insert_data(self, chunks, metadata, embeddings):
        """插入數據到 Milvus Lite"""
        print("正在插入數據...")
        
        # 準備數據
        data = []
        for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
            data.append({
                "id": i,
                "vector": embeddings[i].tolist(),
                "text": chunk,
                "folder": meta["folder"],
                "file": meta["file"],
                "timestamp": meta["timestamp"]
            })
        
        # 分批插入
        batch_size = 100
        total_inserted = 0
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            try:
                res = self.client.insert(
                    collection_name=self.collection_name,
                    data=batch
                )
                total_inserted += len(batch)
                print(f"已插入 {total_inserted}/{len(data)} 條數據")
            except Exception as e:
                print(f"插入失敗: {e}")
                break
        
        print(f"數據插入完成，共 {total_inserted} 條數據")
        return total_inserted
    
    def search_similar(self, query_text, top_k=5, filter_condition=None):
        """搜索相似文本"""
        # 生成查詢嵌入
        query_embedding = self.model.encode(
            [query_text], 
            device=self.device,
            convert_to_numpy=True
        )[0].tolist()
        
        # FLAT 索引不需要特殊的搜尋參數
        search_params = {
            "params": {}
        }
        
        # 執行搜索
        res = self.client.search(
            collection_name=self.collection_name,
            anns_field=self.vector_field_name,
            data=[query_embedding],
            filter=filter_condition,
            limit=top_k,
            output_fields=["text", "folder", "file", "timestamp"],
            search_params=search_params
        )
        
        # 整理結果
        results = []
        for hit in res[0]:
            results.append({
                "id": hit["id"],
                "score": hit["distance"],
                "text": hit["entity"]["text"],
                "source": f"{hit['entity']['folder']}/{hit['entity']['file']}",
                "timestamp": hit["entity"]["timestamp"]
            })
        
        return results
    
    def get_collection_stats(self):
        """獲取集合統計信息"""
        try:
            stats = self.client.get_collection_stats(self.collection_name)
            return stats
        except Exception as e:
            print(f"獲取集合統計信息失敗: {e}")
            return {}
    
    def close(self):
        """關閉連接"""
        self.client.close()
        # 清理 GPU 內存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    try:
        # 初始化檢索器
        db_path = "./milvus_data.db"
        retriever = MilvusLiteRetriever("./milvus_data.db", "./embedding-model")
        
        # 檢查集合是否存在
        collection_exists = retriever.create_collection_if_not_exists('testchunks')
        
        if not collection_exists:
            # 第一次運行，需要插入數據
            print("正在加載文本 chunks...")
            chunks, metadata = retriever.load_chunks("./chunks_output")
            
            if chunks:
                # 生成嵌入向量（使用 GPU 加速）
                embeddings = retriever.generate_embeddings(chunks, batch_size=64)
                
                # 插入數據
                retriever.insert_data(chunks, metadata, embeddings)
            else:
                print("未找到任何文本 chunks")
                return
        
        # 顯示集合信息
        stats = retriever.get_collection_stats()
        print(f"集合統計: {stats}")
        
        # 交互式搜索
        while True:
            print("\n" + "="*50)
            query = input("milvus 数据库检索测试！！！請輸入搜索內容（輸入 'quit' 退出）: ").strip()
            
            if query.lower() == 'quit':
                break
            
            if not query:
                continue
            
            # 執行搜索
            results = retriever.search_similar(query, top_k=5)
            
            print(f"\n找到 {len(results)} 個相關結果:")
            for i, result in enumerate(results, 1):
                print(f"{i}. ID: {result['id']}, 相似度: {result['score']:.4f}")
                print(f"   來源: {result['source']}")
                print(f"   時間: {result['timestamp']}")
                print(f"   內容: {result['text'][:100]}...")
                print()
    
    except Exception as e:
        print(f"程序執行出錯: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 確保資源被正確釋放
        if 'retriever' in locals():
            retriever.close()
        print("程序結束")

if __name__ == "__main__":
    main()
