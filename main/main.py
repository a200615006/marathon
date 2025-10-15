from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import uvicorn
from typing import Optional, List, Any
import json
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="创新大赛答题 API 服务",
    description="处理选择题和问答题的 HTTP 服务",
    version="1.0.0"
)


# 请求和响应数据模型
class ChoiceQuestionRequest(BaseModel):
    segments: str
    paper: str
    id: int
    question: str
    category: str
    content: Optional[str] = None


class ChoiceQuestionResponse(BaseModel):
    segments: str
    paper: str
    id: int
    answer: str


class QAQuestionRequest(BaseModel):
    segments: str
    paper: str
    id: int
    category: str
    question: str


class QAQuestionResponse(BaseModel):
    segments: str
    paper: str
    id: int
    answer: str


def process_choice_question(question: str, content: str) -> str:
    """
    处理选择题
    根据问题内容和选项分析正确答案
    """
    logger.info(f"处理选择题: {question}")
    logger.info(f"选项内容: {content}")

    # 这里应该是您的AI模型推理逻辑
    # 以下为示例逻辑，您需要替换为实际的模型推理

    # 示例1: 根据关键词判断
    question_lower = question.lower()
    content_lower = content.lower()

    # 示例逻辑 - 实际使用时请替换为您的模型推理
    if "失信" in question_lower or "信用中国" in content_lower:
        return "C"
    elif "违法" in question_lower or "违规" in question_lower:
        return "C"
    elif "无效" in question_lower or "不得参与" in question_lower:
        # 分析哪个选项会导致不能投标
        if "失信" in content_lower or "被执行人" in content_lower:
            return "C"
        elif "合法" in content_lower or "有效" in content_lower:
            return "A"  # 这个通常是允许的
    elif "sql" in question_lower or "查询" in question_lower:
        # 如果是SQL相关选择题
        if "错误" in question_lower or "不正确" in question_lower:
            # 找出错误的选项
            if "select *" in content_lower and "limit" not in content_lower:
                return "A"  # 示例
        else:
            # 找出正确的选项
            if "limit 5" in content_lower or "top 5" in content_lower:
                return "C"  # 示例

    # 默认返回逻辑 - 实际应该基于模型推理
    options = ["A", "B", "C", "D"]
    # 这里可以添加更复杂的逻辑来分析哪个选项最可能是正确答案

    # 简单示例：根据问题长度选择选项
    return options[len(question) % len(options)]


def process_qa_question(question: str) -> str:
    """
    处理SQL问答题
    根据问题生成相应的SQL查询语句或结果
    """
    logger.info(f"处理SQL问答题: {question}")

    # 这里应该是您的AI模型推理逻辑
    # 以下为示例逻辑，您需要替换为实际的模型推理

    question_lower = question.lower()

    # 示例SQL查询生成 - 实际使用时请替换为您的模型推理
    if "商户类型" in question and "online" in question_lower:
        # 生成查询前5个ONLINE类型商户的SQL
        sql_result = [
            ["M00005", "商户05", "BL00000005"],
            ["M00006", "商户06", "BL00000006"],
            ["M00008", "商户08", "BL00000008"],
            ["M00012", "商户12", "BL00000012"],
            ["M00016", "商户16", "BL00000016"]
        ]
        return json.dumps(sql_result, ensure_ascii=False)

    elif "商户" in question and "前5" in question:
        # 通用商户查询
        sql_result = [
            ["M00001", "商户01", "BL00000001"],
            ["M00002", "商户02", "BL00000002"],
            ["M00003", "商户03", "BL00000003"],
            ["M00004", "商户04", "BL00000004"],
            ["M00005", "商户05", "BL00000005"]
        ]
        return json.dumps(sql_result, ensure_ascii=False)

    elif "员工" in question or "employ" in question_lower:
        # 员工信息查询
        sql_result = [
            ["E001", "张三", "部门A"],
            ["E002", "李四", "部门B"],
            ["E003", "王五", "部门A"],
            ["E004", "赵六", "部门C"],
            ["E005", "钱七", "部门B"]
        ]
        return json.dumps(sql_result, ensure_ascii=False)

    else:
        # 默认返回空结果
        return "[]"


@app.post("/api/exam", response_model=ChoiceQuestionResponse)
async def exam_api(request: ChoiceQuestionRequest):
    """
    主答题接口
    接收问题并返回答案
    """
    try:
        logger.info(f"收到请求 - segments: {request.segments}, paper: {request.paper}, ID: {request.id}, category: {request.category}")
        logger.info(f"question: {request.question}，content: {request.content}")

        # 根据问题类型调用不同的处理函数
        if request.category == "选择题":
            answer = process_choice_question(request.question, request.content or "")
        else:
            answer = process_qa_question(request.question)

        # 构建响应
        response = ChoiceQuestionResponse(
            segments=request.segments,
            paper=request.paper,
            id=request.id,
            answer=answer
        )

        logger.info(f"返回答案: {answer}")
        return response

    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理请求时发生错误: {str(e)}")


@app.get("/")
async def root():
    """根端点，服务健康检查"""
    return {
        "message": "竞赛答题 API 服务运行中",
        "status": "healthy",
        "endpoint": "/api/exam"
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}


if __name__ == "__main__":
    # 启动服务在 10000 端口
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=10001,
        workers=1
    )