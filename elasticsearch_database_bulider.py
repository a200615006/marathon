from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError, SSLError
import os
import time

def connect_es():
    """连接本地Elasticsearch（修复SSL证书验证问题）"""
    try:
        es = Elasticsearch(
            ["https://localhost:9200"],
            basic_auth=('elastic', 'vSCQnhBXoox0sRo7-U1x'),  # 你的密码
            verify_certs=False,  # 关键：跳过SSL证书验证（测试环境用）
            ssl_show_warn=False  # 关闭SSL警告信息
        )
        # 显式测试连接（比es.ping()更可靠）
        es.info()  # 发送一个实际请求验证连接
        print("✅ 成功连接到Elasticsearch")
        return es
    except SSLError as e:
        print(f"❌ SSL证书错误：{e}（已尝试关闭验证，仍失败）")
        return None
    except Exception as e:
        print(f"❌ 连接错误：{e}")
        return None

def create_index(es, index_name="chunk_documents"):
    """创建索引（若不存在）"""
    if not es.indices.exists(index=index_name):
        try:
            mapping = {
                "mappings": {
                    "properties": {
                        "filename": {"type": "text"},
                        "content": {"type": "text"},
                        "file_path": {"type": "keyword"},
                        "folder": {"type": "keyword"},
                        "import_time": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss"}
                    }
                }
            }
            es.indices.create(index=index_name, body=mapping)
            print(f"✅ 索引 '{index_name}' 创建成功")
        except RequestError as e:
            print(f"❌ 创建索引失败：{e}")

def import_documents(es, root_dir="/home/esuser/chunks_output", index_name="chunk_documents"):
    """导入拷贝后的chunks_output下的所有文档"""
    if not os.path.exists(root_dir):
        print(f"❌ 文件夹 '{root_dir}' 不存在")
        return
    
    total = 0
    success = 0
    fail = 0
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".txt"):
                total += 1
                file_path = os.path.join(dirpath, filename)
                rel_folder = os.path.relpath(dirpath, root_dir)
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    doc = {
                        "filename": filename,
                        "content": content,
                        "file_path": file_path,
                        "folder": rel_folder,
                        "import_time": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    doc_id = f"{rel_folder}/{filename[:-4]}".replace(os.sep, "_")
                    es.index(index=index_name, id=doc_id, body=doc)
                    
                    success += 1
                    print(f"✅ 已导入 {success}/{total}：{filename}")
                    
                except Exception as e:
                    fail += 1
                    print(f"❌ 导入失败 {fail}/{total}：{filename}，错误：{e}")
    
    print(f"\n📊 导入完成：总文件 {total}，成功 {success}，失败 {fail}")

if __name__ == "__main__":
    es = connect_es()
    if es:
        create_index(es)
        import_documents(es)
