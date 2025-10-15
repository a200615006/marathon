import requests
import json


def test_exam_api():
    """测试答题API"""
    url = "http://localhost:10001/api/exam"

    # 测试选择题
    choice_question = {
        "segments": "初赛",
        "paper": "B",
        "id": 3,
        "question": "下列哪项情形会导致投标人不得参与本项目投标？",
        "category": "选择题",
        "content": "选项：\\nA) 具有合法有效的注册证明文件\\nB) 能提供增值税专用发票\\nC) 被'信用中国'列入失信被执行人名单\\nD) 同意遵守招标人保密要求"
    }

    # 测试问答题
    sql_question = {
        "segments": "初赛",
        "paper": "B",
        "id": 4,
        "category": "问答题",
        "question": "请写出查询商户类型为'ONLINE'的前5个商户的商户ID、商户名称和营业执照号的SQL语句。请通过SQL查询语句（SELECT语句）获取结果，最终仅返回结果值"
    }

    print("测试选择题:")
    response1 = requests.post(url, json=choice_question)
    print(f"状态码: {response1.status_code}")
    print(f"响应: {json.dumps(response1.json(), indent=2, ensure_ascii=False)}")

    print("\n测试问答题:")
    response2 = requests.post(url, json=sql_question)
    print(f"状态码: {response2.status_code}")
    print(f"响应: {json.dumps(response2.json(), indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    test_exam_api()