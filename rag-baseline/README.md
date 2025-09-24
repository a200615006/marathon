# 项目部署与使用指南

## 傻瓜式指南
1. 赋予脚本权限
```bash
chmod +x deploy_all.sh
chmod +x es_create_user.sh
```

2. 执行
```bash
bash deploy_all.sh
```

3. 若第六步创建ES专用用户失败，手动执行，注意是否有足够权限：
```bash
bash ./es_create_user.sh
```

4. 启动操作（按以下子步骤依次执行）：
   1. 打开新终端，执行命令（输入此前设置的ES用户密码）：
      ```bash
      su - es_user
      ```
   2. 继续执行命令，进入Elasticsearch的bin目录（需替换“项目文件夹路径”为实际路径）：
      ```bash
      cd 项目文件夹路径/elasticsearch-9.1.3/bin
      ```
   3. 执行启动命令，同时**重点记录终端中显示的「password: xxxxxxxx」信息**（后续配置需用到）：
      ```bash
      ./elasticsearch
      ```
   4. 找到并打开`elasticsearch_database_bulider.py`文件，将其中的认证配置修改为步骤3记录的密码：
      ```python
      basic_auth=('elastic', '这里填记录的密码')  # 替换“这里填记录的密码”为实际值
      ```
   5. 切换回原终端（非启动ES的终端），执行命令导入ES数据：
      ```bash
      python elasticsearch_database_bulider.py
      ```
   6. 打开`rag.py`文件，修改两处关键配置（ES密码、大模型API），完成后执行命令启动RAG：
      ```bash
      python rag.py
      ```


## 一、项目结构
```
./
├── data/                # 数据库源文件存放路径
├── data_processor/      # 数据解析代码
├── embedding-model/     # 嵌入模型存储路径（已内置bce模型）
├── reranker-model/      # 重排序模型存储路径（已内置bce模型）
├── documents_output/    # 文本解析后生成的txt文件存储路径
└── chunks_output/       # 文本切分后生成的文本片存储路径
```


## 二、Python环境安装
1. 进入项目主目录，执行以下命令安装依赖：
   ```bash
   pip install -r requirements
   ```
2. 若存在缺失的环境依赖，补充使用 `pip install [依赖包名]` 安装。


## 三、硬件平台注意事项
- **项目构建环境**：Python 3.12、PyTorch 2.7.0、CUDA 12.8，硬件为 18 vCPU（AMD EPYC 9754 128-Core Processor）+ RTX 4090D (24GB) × 1。
- **纯CPU使用提醒**：需注意两点：
  1. Milvus数据库构建嵌入向量时的批处理大小；
  2. 后续嵌入模型、重排序模型运行时的数据内存占用与耗时。


## 四、数据准备
1. **支持文件格式**：pdf、docx、md、xlsx、xls。
2. **数据存储路径**：将作为知识库的文档直接放入 `./data` 文件夹内。


## 五、数据解析与分块
1. 在项目根目录下直接运行主脚本：
   ```bash
   python main.py
   ```
2. **输出结果**：
   - 文本解析生成的txt文件 → 存储至 `./documents_output`；
   - 文本切分生成的文本片 → 存储至 `./chunks_output`。
3. **参数说明**：`main.py` 内包含解析、分块相关的众多参数，均有注释，可直接在代码中查看并调整。


## 六、数据库部署与构建
### 6.1 Milvus 数据库
#### 6.1.1 安装方式
| 版本类型       | 安装命令/文档                                                                 | 说明                                                                 |
|----------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------|
| Milvus Lite（轻量版） | `pip install pymilvus`                                                      | 项目默认使用，无需复杂配置                                           |
| Milvus Standalone（独立版） | 官方文档：[Install Standalone](https://milvus.io/docs/zh/install_standalone-binary.md) | 需离线部署、使用HNSW索引时选择；项目内已提供rpm与deb安装包 |

#### 6.1.2 数据库构建步骤
1. **创建数据库文件**：运行以下Python代码初始化数据库：
   ```python
   from pymilvus import MilvusClient
   client = MilvusClient("./milvus_data.db")  # 数据库文件存储路径
   ```
2. **向量生成与存储**：运行 `milvus_database_builder.py`：
   - 功能：调用 `./embedding-model` 中的模型，对 `./chunks_output` 的文本片生成向量并存储至 `milvus_data.db`。
   - 索引调整：默认使用FLAT索引；若需使用HNSW索引，需先安装Milvus Standalone，再参考 [HNSW文档](https://milvus.io/docs/zh/hnsw.md) 修改代码中 `create_index` 函数的配置。


### 6.2 Elasticsearch 数据库
#### 6.2.1 安装方式
1. 项目内已提供 `elasticsearch-5.6.16.tar.gz`（Linux版本），执行以下命令解压至当前目录：
   ```bash
   tar -xvf elasticsearch-5.6.16.tar.gz
   ```

#### 6.2.2 数据库构建步骤
1. **用户配置**：由于root用户无法启动Elasticsearch，需新建普通用户，并将解压后的 `elasticsearch-5.6.16` 目录迁移至该用户下。
2. **启动服务**：切换至新建用户，在 `elasticsearch-5.6.16/bin` 目录下启动服务（启动后终端会显示默认密码，需记录）。
3. **数据导入**：
   - 修改 `elasticsearch_database_bulider.py` 中 `basic_auth` 的第二个参数（填入步骤2记录的密码）。
   - 执行脚本：`python elasticsearch_database_bulider.py`，将 `./chunks_output` 的文本片导入数据库。
   - 注意：脚本中 `import_documents` 函数的第一个参数需填写文本片存储的**绝对路径**。


## 七、RAG（检索增强生成）使用
1. **核心脚本**：`rag.py`（包含完整RAG流程）。
2. **参数配置**：在 `rag.py` 开头的“基础配置”部分修改以下参数：
   - Elasticsearch密码；
   - 大模型API地址/密钥；
   - 检索相关参数（如TopK、检索阈值等）。
3. **运行流程**：
   - 执行 `python rag.py` 启动RAG功能；
   - 检索逻辑：采用 **Elasticsearch + Milvus 混合检索**，流程如下：
     1. 用嵌入模型对用户问题生成向量；
     2. 混合检索获取初步结果；
     3. 用 `reranker-model` 中的重排序模型对结果优化排序；
     4. 输出最终回答。