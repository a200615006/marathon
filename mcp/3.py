from contextlib import asynccontextmanager

import os
import json
from datetime import datetime, date
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from advancedMCPHttpToolManager import AdvancedMCPHttpToolManager
from req_resp_obj import ToolResponse, ToolRequest, QueryResponse, UserQuery, ChoiceQuestionResponse, ChoiceQuestionRequest, QAQuestionRequest, QAQuestionResponse
import logging


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期管理器"""
    # 启动时初始化
    global tool_manager
    tool_manager = AdvancedMCPHttpToolManager(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_URL"),
        tools_directory="mcp_tools",
        max_iterations=5,
        headers={
            "X-App-Id": os.getenv("your_app_id"),
            "X-App-Key": os.getenv("your_app_key"),
            "Content-Type": "application/json"
        }
    )
    print("🚀 MCP工具管理器初始化完成")

    yield  # 应用运行期间

    # 关闭时清理（如果需要）
    print("🛑 应用关闭")


# FastAPI 应用
app = FastAPI(
    title="创新大赛答题 API 服务",
    description="处理选择题和问答题的 HTTP 服务",
    version="1.0.0",
    lifespan=lifespan  # 使用 lifespan 事件处理器
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    if not tool_manager:
        raise HTTPException(status_code=500, detail="工具管理器未初始化")

    try:
        result = tool_manager.process_user_query(question,  content or "")
        print(result)
        print(result.response)
        return result.response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")

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


@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "MCP工具服务API - 支持本地和HTTP工具",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/tools")
async def get_available_tools():
    """获取可用工具列表"""
    if not tool_manager:
        raise HTTPException(status_code=500, detail="工具管理器未初始化")

    tools = tool_manager.get_available_tools()
    return {
        "success": True,
        "tools": tools,
        "total_tools": len(tools)
    }


@app.post("/tools/call", response_model=ToolResponse)
async def call_tool(request: ToolRequest):
    """直接调用指定工具"""
    if not tool_manager:
        raise HTTPException(status_code=500, detail="工具管理器未初始化")

    try:
        result = tool_manager.call_tool(request.tool_name, request.arguments)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"工具调用失败: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def process_query(request: UserQuery):
    """处理用户查询，可能涉及多个工具调用"""
    if not tool_manager:
        raise HTTPException(status_code=500, detail="工具管理器未初始化")

    try:
        result = tool_manager.process_user_query(request.query, "")
        print(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")


@app.post("/api/exam", response_model=ChoiceQuestionResponse)
async def exam(request: ChoiceQuestionRequest):
    """
    主答题接口
    接收问题并返回答案
    """
    try:
        logger.info(
            f"收到请求 - segments: {request.segments}, paper: {request.paper}, ID: {request.id}, category: {request.category}")
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


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "tools_loaded": len(tool_manager.tools) if tool_manager else 0
    }


if __name__ == "__main__":
    # 启动FastAPI服务器
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=10002,
        log_level="info"
    )