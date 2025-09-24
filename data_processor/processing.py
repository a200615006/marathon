import os
import re
import traceback
from typing import List

# 從我們自己的模組中導入功能
from .parsers import (
    extract_outline_from_docx_with_tables,
    extract_outline_from_pdf,
    extract_outline_from_markdown,
    extract_outline_from_excel
)
from .document_tree import DocumentTree
from .chunking import merge_chunks_by_path
from .utils import (
    save_chunks_to_files,
    analyze_document_structure,
    export_chunks_to_json,
    create_chunk_index,
    is_table_header # parse_extracted_text 會用到
)


def parse_extracted_text(text_content: str) -> DocumentTree:
    """解析提取的文本內容，構建文檔樹"""
    tree = DocumentTree()
    lines = text_content.split('\n')


    # 檢查是否有任何標題結構
    has_headings = any(line.strip().startswith('Heading') for line in lines)


    # 如果完全沒有標題結構，將所有內容作為單一段落處理
    if not has_headings:
        print("No headings found, treating as single paragraph document")
        all_text = '\n'.join([line for line in lines if line.strip() and not line.startswith('Outline')])

        # 直接添加到根節點
        if tree.nodes[0].text:
            tree.nodes[0].text += "\n" + all_text
        else:
            tree.nodes[0].text = all_text

        return tree


    current_tables = []  # 暫存表格數據
    cache_lines=''

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # print(f"Processing line: {line}")

        # 解析標題
        heading_match = re.match(r'^(\s*)Heading (\d+): (.+)$', line)
        if heading_match:
            if cache_lines:
                # print(f"adding paragraph: {content}")
                tree.add_paragraph(cache_lines, 999, "paragraph")
                cache_lines=''

            indent, level_str, title = heading_match.groups()
            level = int(level_str)

            # 如果有暫存的表格，添加到上一個節點
            if current_tables:
                for table in current_tables:
                    tree.add_table_to_current_node(table)
                current_tables = []

            tree.add_paragraph(title, level, "heading")
            continue

        # 解析段落
        paragraph_match = re.match(r'^Paragraph: (.+)$', line)
        if paragraph_match:
            if cache_lines:
                # print(f"adding paragraph: {content}")
                tree.add_paragraph(cache_lines, 999, "paragraph")
                cache_lines=''

            content = paragraph_match.group(1)
            # print(f"Processing paragraph: {content}")

            # 檢查是否是表格行
            if '|' in content and content.count('|') >= 2:
                # 這是表格數據，暫存起來
                table_row = [cell.strip() for cell in content.split('|')[1:-1]]

                # 如果是新表格的開始
                if not current_tables:
                    current_tables.append({
                        'rows': [table_row],
                        'has_header': is_table_header(table_row),
                        'position': 'after_current_heading'
                    })
                else:
                    # 添加到當前表格
                    current_tables[-1]['rows'].append(table_row)
            else:
                # 如果有暫存的表格，先添加到當前節點
                if current_tables:
                    for table in current_tables:
                        tree.add_table_to_current_node(table)
                    current_tables = []

                # 正文段落
                cache_lines += content + '\n'
            continue
        else:
            if current_tables:
                # 當前行不是表格行，將暫存的表格添加到上一個節點
                for table in current_tables:
                    tree.add_table_to_current_node(table)
                current_tables = []
            else:
                cache_lines += line + '\n'


        # 解析表格標題
        table_match = re.match(r'^Table \d+ \(Page \d+\):$', line)
        if table_match:
            # 表格開始標記，準備收集表格數據
            continue
        
        

    # 處理剩餘的表格
    if current_tables:
        for table in current_tables:
            tree.add_table_to_current_node(table)

    return tree


def process_documents_with_advanced_options(target_files, data_folder,                                                                                      chunks_output_folder=None,
                                           documents_output_folder=None,
                                           save_full_text=True,
                                           save_tree_structure=True,
                                           save_chunk_summary=True,
                                           filename_prefix="",
                                           # 現有選項
                                           enable_chunking=True,
                                           max_chunk_size=4096,
                                           min_overlap=100,
                                           max_overlap=400,
                                           enable_path_merging=False,
                                           export_json=True,
                                           create_index=True,
                                           analyze_structure=True):
                                           # 新增輸出文件夾選項
    """增強版文檔處理函數，包含超细粒度合并和選擇性保存選項"""

    # 設置輸出文件夾，如果未指定則使用默認
    if documents_output_folder is None:
        documents_output_folder = output_folder
    if chunks_output_folder is None:
        chunks_output_folder = output_folder
    
    os.makedirs(documents_output_folder, exist_ok=True)
    os.makedirs(chunks_output_folder, exist_ok=True)

    for file_name in target_files:
        file_path = os.path.join(data_folder, file_name)

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        try:
            print(f"Processing: {file_name}")

            if file_name.lower().endswith('.docx'):
                extracted_content = extract_outline_from_docx_with_tables(file_path)
            elif file_name.lower().endswith('.pdf'):
                extracted_content = extract_outline_from_pdf(file_path)
            elif file_name.lower().endswith('.md'):
                extracted_content = extract_outline_from_markdown(file_path)
            elif file_name.lower().endswith(('.xlsx', '.xls')):
                extracted_content = extract_outline_from_excel(file_path)
            else:
                print(f"Unsupported file type: {file_name}")
                continue

            doc_tree = parse_extracted_text(extracted_content)
            base_filename = file_name.rsplit('.', 1)[0]

            if analyze_structure:
                analyze_document_structure(doc_tree)

            # 根據選項保存基礎文件到文檔輸出文件夾
            if save_full_text:
                output_filename = base_filename + '.txt'
                output_path = os.path.join(documents_output_folder, output_filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(extracted_content)
                print(f"Saved full text to {output_path}")

            if save_tree_structure:
                tree_filename = base_filename + '_tree.txt'
                tree_path = os.path.join(documents_output_folder, tree_filename)
                with open(tree_path, 'w', encoding='utf-8') as f:
                    f.write("Document Tree Structure:\n")
                    f.write("=" * 50 + "\n")
                    f.write(doc_tree.get_tree_structure())
                print(f"Saved tree structure to {tree_path}")

            if enable_chunking:
                print("Creating chunks...")
                chunks = doc_tree.create_all_chunks(
                    max_chunk_size=max_chunk_size,
                    min_overlap=min_overlap,
                    max_overlap=max_overlap
                )

                if enable_path_merging and chunks:
                    print("Applying path-based merging...")
                    original_count = len(chunks)
                    chunks = merge_chunks_by_path(chunks, max_chunk_size, max_overlap)
                    print(f"Path merging reduced chunks from {original_count} to {len(chunks)}")

                if chunks:
                    # 傳入前綴以保存分塊文件到chunk輸出文件夾
                    chunks_folder = save_chunks_to_files(chunks, chunks_output_folder, base_filename, filename_prefix)

                    # 選擇性保存 chunk summary
                    if not save_chunk_summary:
                        summary_path = os.path.join(chunks_folder, "chunks_summary.txt")
                        if os.path.exists(summary_path):
                            os.remove(summary_path)
                            print("Chunk summary file removed as per settings.")

                    if export_json:
                        json_path = os.path.join(chunks_output_folder, f"{base_filename}_chunks.json")
                        export_chunks_to_json(chunks, json_path)

                    if create_index:
                        chunk_index = create_chunk_index(chunks)
                        index_path = os.path.join(chunks_output_folder, f"{base_filename}_chunk_index.json")
                        import json
                        with open(index_path, 'w', encoding='utf-8') as f:
                            json.dump(chunk_index, f, ensure_ascii=False, indent=2)
                        print(f"Created chunk index: {index_path}")

                    print(f"Successfully processed {file_name} with {len(chunks)} chunks")
                else:
                    print("No chunks created")

            print()

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            import traceback
            traceback.print_exc()
