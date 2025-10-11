import openai
import json
import os
import requests
import time
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI


class AdvancedMCPHttpToolManager:
    def __init__(self, api_key: str, base_url: str = None, tools_directory: str = "mcp_tools", max_iterations: int = 5):
        """
        高级MCP HTTP工具管理器，支持多次调用和多个工具

        Args:
            api_key: OpenAI API密钥
            base_url: OpenAI API基础URL
            tools_directory: MCP工具描述文件目录
            max_iterations: 最大迭代次数，防止无限循环
        """
        client_args = {"api_key": api_key}
        if base_url:
            client_args["base_url"] = base_url
            client_args["timeout"] = 60.0

        self.client = OpenAI(**client_args)
        self.tools_directory = tools_directory
        self.max_iterations = max_iterations
        self.tools = self.load_tools_from_files()
        self.conversation_history = []
        self.call_log = []  # 记录所有工具调用

        print(json.dumps(self.tools, indent=4, ensure_ascii=False))

    def load_tools_from_files(self) -> List[Dict[str, Any]]:
        """从文本文件加载MCP工具描述"""
        tools = []

        for filename in os.listdir(self.tools_directory):
            if filename.endswith(('.txt', '.json')):
                file_path = os.path.join(self.tools_directory, filename)
                try:
                    tool_config = self.parse_tool_file(file_path)
                    if tool_config:
                        tools.append(tool_config)
                        print(f"✅ 已加载工具: {tool_config['function']['name']}")
                except Exception as e:
                    print(f"❌ 加载工具文件 {filename} 时出错: {e}")

        print(f"📊 总共加载了 {len(tools)} 个工具")
        return tools

    def parse_tool_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """解析单个工具描述文件"""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()

        # 支持JSON格式
        if content.startswith('{'):
            try:
                tool_data = json.loads(content)
                # 确保有http_config
                if 'http_config' not in tool_data:
                    tool_data['http_config'] = {
                        'url': tool_data.get('url', ''),
                        'method': tool_data.get('method', 'GET')
                    }
                return tool_data
            except json.JSONDecodeError as e:
                print(f"JSON解析错误 {file_path}: {e}")
                return None

        # 解析文本格式
        return self.parse_text_format(content, os.path.basename(file_path))

    def parse_text_format(self, content: str, filename: str) -> Dict[str, Any]:
        """解析文本格式的工具描述"""
        lines = content.split('\n')
        tool_info = {
            "name": "",
            "description": "",
            "http_config": {},
            "parameters": {"type": "object", "properties": {}, "required": []}
        }

        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测章节标题
            if line.lower().startswith('name:'):
                tool_info['name'] = line[5:].strip()
            elif line.lower().startswith('description:'):
                tool_info['description'] = line[12:].strip()
            elif line.lower().startswith('http_url:'):
                tool_info['http_config']['url'] = line[9:].strip()
            elif line.lower().startswith('http_method:'):
                tool_info['http_config']['method'] = line[12:].strip().upper()
            elif line.lower().startswith('parameters:'):
                current_section = 'parameters'
            elif line.lower().startswith('required:'):
                required_params = line[9:].strip().split(',')
                tool_info['parameters']['required'] = [p.strip() for p in required_params if p.strip()]
            elif ':' in line and current_section == 'parameters':
                param_name, param_desc = line.split(':', 1)
                param_name = param_name.strip()
                param_desc = param_desc.strip()

                param_type = "string"
                if any(word in param_desc.lower() for word in ['number', 'int', 'float', 'integer']):
                    param_type = "number"
                elif any(word in param_desc.lower() for word in ['boolean', 'bool']):
                    param_type = "boolean"
                elif any(word in param_desc.lower() for word in ['array', 'list']):
                    param_type = "array"

                tool_info['parameters']['properties'][param_name] = {
                    "type": param_type,
                    "description": param_desc
                }

        if not tool_info['name']:
            tool_info['name'] = os.path.splitext(filename)[0]

        if 'method' not in tool_info['http_config']:
            tool_info['http_config']['method'] = 'GET'

        return {
            "type": "function",
            "function": {
                "name": tool_info['name'],
                "description": tool_info['description'],
                "parameters": tool_info['parameters']
            },
            "http_config": tool_info['http_config']
        }

    def call_http_service(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """调用本地HTTP服务"""
        tool_config = None
        for tool in self.tools:
            if tool['function']['name'] == tool_name and 'http_config' in tool:
                tool_config = tool
                break

        if not tool_config:
            return f"错误: 未找到工具 '{tool_name}' 的配置"

        http_config = tool_config['http_config']
        url = http_config.get('url')
        method = http_config.get('method', 'GET')

        if not url:
            return f"错误: 工具 '{tool_name}' 未配置HTTP URL"

        try:
            print(f"🌐 调用HTTP服务: {method} {url}")
            print(f"📤 请求参数: {arguments}")

            # 记录调用开始
            call_start = time.time()

            # 根据HTTP方法调用服务
            if method == 'GET':
                response = requests.get(url, params=arguments, timeout=30)
            elif method == 'POST':
                response = requests.post(url, json=arguments, timeout=30)
            elif method == 'PUT':
                response = requests.put(url, json=arguments, timeout=30)
            elif method == 'DELETE':
                response = requests.delete(url, params=arguments, timeout=30)
            else:
                return f"错误: 不支持的HTTP方法 {method}"

            call_duration = time.time() - call_start

            # 记录调用日志
            call_log = {
                "tool": tool_name,
                "arguments": arguments,
                "status_code": response.status_code,
                "duration": round(call_duration, 2),
                "timestamp": time.time()
            }
            self.call_log.append(call_log)

            if response.status_code == 200:
                result = response.json() if 'application/json' in response.headers.get('content-type',
                                                                                       '') else response.text
                print(f"✅ HTTP调用成功 (耗时: {call_duration:.2f}s)")
                return str(result)
            else:
                error_msg = f"HTTP错误 {response.status_code}: {response.text}"
                print(f"❌ {error_msg}")
                return error_msg

        except requests.exceptions.Timeout:
            error_msg = f"HTTP请求超时: {url}"
            print(f"⏰ {error_msg}")
            return error_msg
        except requests.exceptions.ConnectionError:
            error_msg = f"无法连接到HTTP服务: {url}"
            print(f"🔌 {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"HTTP调用异常: {str(e)}"
            print(f"🚨 {error_msg}")
            return error_msg

    def process_user_query(self, user_query: str) -> str:
        """
        处理用户查询，支持多次工具调用和多个工具

        返回:
            str: 最终回复内容
        """
        messages = self.conversation_history + [{"role": "user", "content": user_query}]
        iteration_count = 0

        print(f"\n🔍 开始处理查询: {user_query}")

        while iteration_count < self.max_iterations:
            iteration_count += 1
            print(f"\n🔄 第 {iteration_count} 轮处理")

            try:
                # 准备工具列表（移除http_config）
                available_tools = [{k: v for k, v in tool.items() if k != 'http_config'} for tool in self.tools]

                response = self.client.chat.completions.create(
                    model="qwen-plus",
                    messages=messages,
                    tools=available_tools if available_tools else None,
                    tool_choice="auto" if available_tools else "none",
                    timeout=30.0
                )

                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls

                # 如果没有工具调用，直接返回结果
                if not tool_calls:
                    final_reply = response_message.content
                    print(f"💬 模型选择直接回复 (第{iteration_count}轮)")

                    # 更新对话历史
                    self.conversation_history.extend([
                        {"role": "user", "content": user_query},
                        {"role": "assistant", "content": final_reply}
                    ])

                    return final_reply

                # 处理工具调用
                print(f"🔧 模型决定调用 {len(tool_calls)} 个工具")
                messages.append(response_message)

                # 执行所有工具调用
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    print(f"🛠️ 调用工具 [{function_name}]: {function_args}")

                    # 调用HTTP服务
                    tool_result = self.call_http_service(function_name, function_args)

                    # 将工具结果添加到消息中
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    })

                # 检查是否应该继续迭代
                if iteration_count >= self.max_iterations:
                    print("⚠️ 达到最大迭代次数，生成最终回复")
                    break

            except Exception as e:
                error_msg = f"第 {iteration_count} 轮处理时出错: {e}"
                print(f"🚨 {error_msg}")
                messages.append({"role": "system", "content": f"处理错误: {e}"})
                break

        # 生成最终回复
        try:
            final_response = self.client.chat.completions.create(
                model="qwen-plus",
                messages=messages,
                timeout=30.0
            )

            final_content = final_response.choices[0].message.content
            print(f"✅ 处理完成，共进行 {iteration_count} 轮，调用 {len(self.call_log)} 次工具")

            # 更新对话历史
            self.conversation_history.extend([
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": final_content}
            ])

            return final_content

        except Exception as e:
            return f"生成最终回复时出错: {e}"

    def get_call_statistics(self) -> Dict[str, Any]:
        """获取工具调用统计信息"""
        if not self.call_log:
            return {"total_calls": 0, "tools_used": []}

        tool_usage = {}
        for call in self.call_log:
            tool_name = call["tool"]
            if tool_name in tool_usage:
                tool_usage[tool_name] += 1
            else:
                tool_usage[tool_name] = 1

        return {
            "total_calls": len(self.call_log),
            "unique_tools": len(tool_usage),
            "tools_used": tool_usage,
            "average_duration": round(sum(call["duration"] for call in self.call_log) / len(self.call_log), 2)
        }


# 增强的模拟HTTP服务
class EnhancedMockHttpServer:
    """支持多次调用的模拟HTTP服务"""

    @staticmethod
    def run_mock_server():
        from flask import Flask, request, jsonify

        app = Flask(__name__)

        # 模拟数据库
        users_db = {
            "1": {"id": "1", "name": "张三", "email": "zhangsan@example.com", "city": "北京"},
            "2": {"id": "2", "name": "李四", "email": "lisi@example.com", "city": "上海"},
            "3": {"id": "3", "name": "王五", "email": "wangwu@example.com", "city": "广州"}
        }

        orders_db = {
            "1": [
                {"order_id": "ORD001", "product_id": "P100", "quantity": 2, "status": "completed"},
                {"order_id": "ORD002", "product_id": "P200", "quantity": 1, "status": "pending"}
            ],
            "2": [
                {"order_id": "ORD003", "product_id": "P100", "quantity": 1, "status": "completed"}
            ]
        }

        products_db = {
            "P100": {"id": "P100", "name": "智能手机", "price": 2999, "category": "电子产品"},
            "P200": {"id": "P200", "name": "笔记本电脑", "price": 5999, "category": "电子产品"},
            "P300": {"id": "P300", "name": "书籍", "price": 59, "category": "图书"}
        }

        weather_db = {
            "北京": {"temperature": 22, "condition": "晴朗", "humidity": 65},
            "上海": {"temperature": 25, "condition": "多云", "humidity": 70},
            "广州": {"temperature": 28, "condition": "小雨", "humidity": 80},
            "纽约": {"temperature": 18, "condition": "晴朗", "humidity": 60}
        }

        @app.route('/api/users', methods=['GET'])
        def get_user():
            user_id = request.args.get('user_id')
            user = users_db.get(user_id)
            if user:
                return jsonify(user)
            else:
                return jsonify({"error": "用户不存在"}), 404

        @app.route('/api/orders', methods=['GET'])
        def get_orders():
            user_id = request.args.get('user_id')
            orders = orders_db.get(user_id, [])
            return jsonify({
                "user_id": user_id,
                "orders": orders,
                "total_orders": len(orders)
            })

        @app.route('/api/products', methods=['GET'])
        def get_product():
            product_id = request.args.get('product_id')
            product = products_db.get(product_id)
            if product:
                return jsonify(product)
            else:
                return jsonify({"error": "产品不存在"}), 404

        @app.route('/api/weather', methods=['GET'])
        def get_weather():
            cities_param = request.args.get('cities', '')
            cities = [city.strip() for city in cities_param.split(',') if city.strip()]

            weather_data = {}
            for city in cities:
                if city in weather_db:
                    weather_data[city] = weather_db[city]
                else:
                    weather_data[city] = {"error": "城市数据不存在"}

            return jsonify(weather_data)

        @app.route('/api/analyze', methods=['POST'])
        def analyze_data():
            data = request.json
            analysis_type = data.get('analysis_type', 'summary')

            if analysis_type == 'comparison':
                result = {"type": "comparison", "summary": "数据对比分析完成", "insights": ["发现趋势差异", "识别关键指标"]}
            elif analysis_type == 'summary':
                result = {"type": "summary", "key_points": ["数据摘要1", "数据摘要2"], "conclusion": "分析完成"}
            else:
                result = {"type": analysis_type, "status": "analyzed", "message": "分析处理完成"}

            return jsonify(result)

        print("🚀 启动增强模拟HTTP服务器在 http://localhost:8000")
        app.run(port=8000, debug=False)


# 使用示例和测试
def main():
    # 初始化高级MCP工具管理器
    tool_manager = AdvancedMCPHttpToolManager(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_URL"),  # 如果需要代理，请设置base_url
        tools_directory="mcp_tools",
        max_iterations=5
    )

    # 测试复杂查询（需要多次调用多个工具）
    test_queries = [
        # 简单查询 - 可能直接回答
        # "你好，请介绍一下你自己",
        #
        # # 需要单次工具调用
        # "查询用户1的基本信息",
        #
        # # 需要多次工具调用（先查用户，再查订单）
        # "请分析用户1的购买行为和订单情况",
        #
        # # 需要多个工具调用（用户信息 + 订单 + 产品详情）
        "请给我用户2的完整信息，包括他的订单和购买的产品详情"
        #
        # # 需要多次调用的复杂分析
        # "比较北京、上海、广州三个城市的天气情况"

        # 链式调用
        # "先查询用户1的信息，然后根据他的订单查看产品详情，最后做个总结",
        
        # "查看上海的天气，如果是晴天的话查询用户1的信息，否则查询用户2的信息,总结输出",

        # "今天上海的天气分析总结一下"
    ]

    print("🤖 高级MCP HTTP工具管理器已就绪")
    print("=" * 60)

    for i, query in enumerate(test_queries, 1):
        print(f"\n🧪 测试用例 {i}/{len(test_queries)}")
        print(f"👤 用户提问: {query}")
        print("-" * 50)

        # 清空调用日志（可选）
        tool_manager.call_log = []

        response = tool_manager.process_user_query(query)
        print(f"🤖 AI回复: {response}")

        # 显示调用统计
        stats = tool_manager.get_call_statistics()
        if stats["total_calls"] > 0:
            print(f"📊 工具调用统计: {stats}")

        print("=" * 60)


def run_mock_server():
    """运行模拟HTTP服务器"""
    EnhancedMockHttpServer.run_mock_server()


if __name__ == "__main__":
    # 如果要运行模拟服务器，取消下面的注释
    #run_mock_server()

    # 运行主程序
    main()