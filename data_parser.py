import os
from data_processor.processing import process_documents_with_advanced_options


if __name__ == "__main__":
    
    # --- 設定區 ---
    DATA_FOLDER = './data' #原始数据存放文件夹，格式可以包括 pdf,docx,md,xlsx,xls
    CHUNKS_OUTPUT_FOLDER = './chunks_output'  # 存放chunk文件的文件夾
    DOCUMENTS_OUTPUT_FOLDER = './documents_output'  # 存放全文txt的文件夾
    
    # 確保資料夾存在
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"資料夾 '{DATA_FOLDER}' 已建立，請將您的文件放入其中。")

    # 獲取 data 資料夾中的所有目標文件
    try:
        file_list = os.listdir(DATA_FOLDER)
        target_files = [f for f in file_list if f.lower().endswith(('.pdf', '.docx', '.md', '.xlsx', '.xls'))]
    except FileNotFoundError:
        target_files = []

    if not target_files:
        print(f"在 '{DATA_FOLDER}' 中找不到任何文件。請新增文件後再試。")
    else:
        print(f"找到目標文件: {target_files}")

        # 调用主函数，並傳入所有配置
        process_documents_with_advanced_options(
            target_files=target_files,
            data_folder=DATA_FOLDER,
            chunks_output_folder=CHUNKS_OUTPUT_FOLDER,
            documents_output_folder=DOCUMENTS_OUTPUT_FOLDER,

            # --- 輸出控制選項 ---
            save_full_text=True, #是否存储从原始文件得到的全文 txt
            save_tree_structure=False, #是否存储每个文件形成的树结构的描述文件
            save_chunk_summary=False, #是否存储chunk 整体的总结统计
            filename_prefix="prod_",  #chunk命名前缀
            
            # --- 處理流程選項 ---
            enable_chunking=True, #是否进行 chunk切片
            max_chunk_size=4096, #最大chunk分片长度
            min_overlap=100, #最小chunk重叠长度
            max_overlap=400, #最大chunk重叠长度
            enable_path_merging=True, #是否合并同一个树节点下的分片（推荐）
            export_json=False, #是否输出分块为 json 格式
            create_index=False, #是否创建一个json格式的 chunk索引文件
            analyze_structure=False #是否分析文档树的结构
        )
        print("所有文件處理完畢！")