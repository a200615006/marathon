#!/bin/bash
echo "===== 第一步：安装Python依赖 ====="
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt || echo "依赖安装部分失败，可后续手动执行：pip install -r requirements"
else
    echo "警告：未找到 requirements 文件，请确认项目文件完整！"
    exit 1
fi

echo -e "\n===== 第二步：数据解析（生成txt和文本片） ====="
python data_parser.py || echo "数据解析失败，请检查 data 文件夹是否放了知识库文档（如pdf/docx）"

echo -e "\n===== 第三步：初始化Milvus数据库 ====="
# 自动执行Milvus数据库创建代码
python -c "from pymilvus import MilvusClient; client = MilvusClient('./milvus_data.db'); print('Milvus数据库初始化成功')" || echo "Milvus初始化失败，请先安装pymilvus：pip install pymilvus"

echo -e "\n===== 第四步：构建Milvus向量库 ====="
if [ -f "milvus_database_builder.py" ]; then
    python milvus_database_builder.py || echo "Milvus向量库构建失败，检查 chunks_output 文件夹是否有内容（数据解析是否成功）"
else
    echo "警告：未找到 milvus_database_builder.py，请确认项目文件完整！"
fi

echo -e "\n===== 第五步：解压Elasticsearch ====="
if [ -f "elasticsearch-9.1.3-linux-x86_64.tar.gz" ]; then
    tar -xvf elasticsearch-9.1.3-linux-x86_64.tar.gz || echo "ES解压失败，检查压缩包是否完整"
else
    echo "警告：未找到 elasticsearch-9.1.3-linux-x86_64.tar.gz，请确认项目文件完整！"
    exit 1
fi

echo -e "\n===== 第六步：创建ES专用用户（需手动输密码） ====="
# 调用ES用户辅助脚本（下方会单独提供）
if [ -f "es_create_user.sh" ]; then
    sudo ./es_create_user.sh || echo "ES用户创建失败，需手动运行：sudo ./es_create_user.sh"
else
    echo "警告：未找到 es_create_user.sh，请先创建该脚本！"
fi

echo -e "\n===== 【重要手动步骤提醒】 ====="
echo -e "\n= 【以下步骤为开机后启动 RAG 的步骤】 ="
echo "1. 请打开新终端，执行：su - es_user（输入刚才设置的ES用户密码）"
echo "2. 再执行：cd 项目文件夹路径/elasticsearch-9.1.3/bin"
echo "3. 再执行：./elasticsearch（启动ES，记录终端里的「password: xxxxxxxx」）"
echo "4. 打开 elasticsearch_database_bulider.py，把「basic_auth=('elastic', '这里填记录的密码')」改对"
echo "5. 回到原终端，执行：python elasticsearch_database_bulider.py（导入ES数据）"
echo "6. 最后改 rag.py 里的「ES密码、大模型API」，执行：python rag.py 启动RAG"
EOF
