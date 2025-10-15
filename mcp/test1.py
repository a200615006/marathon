import pandas as pd
import requests
import time
from typing import Dict, Any, List
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_test_data() -> None:
    """
    加载测试数据并向API发送请求

    提示词说明：
    1. 从Excel文件读取测试数据
    2. 验证数据完整性
    3. 分批发送请求避免服务器过载
    4. 记录执行结果和错误信息
    """
    try:
        file_path = "./sample_A.xlsx"
        logger.info(f"开始加载测试数据文件: {file_path}")

        # 读取Excel文件
        df = pd.read_excel(file_path)
        logger.info(f"数据加载成功，共 {len(df)} 条记录")

        # 显示列信息
        print("数据列信息:")
        print(df.columns.tolist())
        print("\n数据前几行:")
        print(df.head())

        # 验证必要的列是否存在
        required_columns = ["id", "category", "question", "content", "answer", "label"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"缺少必要的列: {missing_columns}")
            return

        # 准备测试数据
        test_cases = []
        for i in range(len(df)):
            test_case = {
                "id": df.iloc[i]["id"],
                "category": df.iloc[i]["category"],
                "question": df.iloc[i]["question"],
                "content": df.iloc[i]["content"]
            }
            test_cases.append(test_case)

        # 分批发送请求（避免服务器压力过大）
        batch_size = 5  # 每批发送5个请求
        delay_between_batches = 1  # 批次间延迟1秒

        successful_requests = 0
        failed_requests = 0

        for i in range(0, len(test_cases), batch_size):
            batch = test_cases[i:i + batch_size]
            logger.info(f"处理批次 {i // batch_size + 1}/{(len(test_cases) - 1) // batch_size + 1}")

            for j, test_case in enumerate(batch):
                case_num = i + j + 1
                logger.info(f"发送请求 {case_num}/{len(test_cases)}: ID={test_case['id']}")

                success = test_http_post(test_case, case_num)
                if success:
                    successful_requests += 1
                else:
                    failed_requests += 1

                # 单个请求间短暂延迟
                if j < len(batch) - 1:
                    time.sleep(0.2)

            # 批次间延迟
            if i + batch_size < len(test_cases):
                time.sleep(delay_between_batches)

        # 输出测试总结
        logger.info(f"测试完成! 成功: {successful_requests}, 失败: {failed_requests}, 总计: {len(test_cases)}")

    except FileNotFoundError:
        logger.error(f"文件未找到: {file_path}")
    except Exception as e:
        logger.error(f"加载测试数据时发生错误: {e}")


def test_http_post(req_object: Dict[str, Any], case_number: int = None) -> bool:
    """
    发送HTTP POST请求到考试API

    提示词说明：
    1. 构建完整的请求数据
    2. 处理网络异常和超时
    3. 验证响应状态
    4. 记录详细的请求和响应信息
    """
    url = "http://localhost:10002/api/exam"

    try:
        # 设置请求超时时间
        timeout = 30

        # 添加请求标识（可选）
        if case_number:
            req_object["test_case_number"] = case_number

        logger.debug(f"发送请求到 {url}")
        logger.debug(f"请求数据: {req_object}")

        # 发送POST请求
        response = requests.post(
            url,
            json=req_object,
            timeout=timeout,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "TestClient/1.0"
            }
        )

        # 检查响应状态
        if response.status_code == 200:
            logger.info(f"请求成功 (案例 {case_number}): {response.text[:100]}...")
            return True
        else:
            logger.warning(f"请求失败 (案例 {case_number}), 状态码: {response.status_code}, 响应: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        logger.error(f"连接错误 (案例 {case_number}): 无法连接到服务器 {url}")
        return False
    except requests.exceptions.Timeout:
        logger.error(f"请求超时 (案例 {case_number}): 超过 {timeout} 秒")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"请求异常 (案例 {case_number}): {e}")
        return False
    except Exception as e:
        logger.error(f"未知错误 (案例 {case_number}): {e}")
        return False


def validate_server_connection() -> bool:
    """
    验证服务器连接是否正常
    """
    url = "http://localhost:10002/api/exam"
    try:
        # 发送一个简单的测试请求
        test_data = {
            "segments": "1",
            "paper": "exam",
            "id": 1,
            "category": "test",
            "question": "连接测试",
            "content": "这是一个连接测试请求",
            "query": "今天星期几"
        }
        response = requests.post(url, json=test_data, timeout=10)
        return response.status_code == 200
    except:
        return False


if __name__ == "__main__":
    """
    主程序执行提示词：
    1. 首先验证服务器连接
    2. 加载并处理测试数据
    3. 提供执行进度反馈
    4. 生成测试报告
    """
    logger.info("=== 开始执行测试程序 ===")

    # 验证服务器连接
    logger.info("验证服务器连接...")
    if validate_server_connection():
        logger.info("服务器连接正常，开始加载测试数据")
        load_test_data()
    else:
        logger.error("无法连接到服务器，请确保FastAPI服务正在运行在 http://localhost:10002")
        logger.info("请使用以下命令启动服务: uvicorn main:app --host 0.0.0.0 --port 10002")

    logger.info("=== 测试程序执行结束 ===")
