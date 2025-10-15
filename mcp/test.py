import pandas as pd
import requests
import requests


def load_test_data():
    file_path = "./sample_A.xlsx"
    df = pd.read_excel(file_path)
    print(df.columns)
    id_list = df["id"].tolist()
    category_list = df["category"].tolist()
    question_list = df["question"].tolist()
    content_list = df["content"].tolist()
    answer_list = df["answer"].tolist()
    label_list = df["label"].tolist()

    for i in range(len(id_list)):
        id = id_list[i]
        category = category_list[i]
        question = question_list[i]
        content = content_list[i]
        label = label_list[i]

        req_object = {
            "query": question
            # "segments": "1",
            # "paper": "exam",
            # "id": id,
            # "category": category,
            # "question": question,
            # "content": content
        }

        test_http_post(req_object)


def test_http_post(req_object):
    url = "http://localhost:10002/query"

    data = req_object
    response = requests.post(url, json=req_object)

    # response = requests.get(url, params= params)
    print(response.text)


if __name__ == "__main__":
    # 启动FastAPI服务器
    load_test_data()