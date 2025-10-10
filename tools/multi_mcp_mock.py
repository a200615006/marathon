import openai
import json
import os
import requests
import time
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI

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

            print(weather_data)
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

            print(result)
            return jsonify(result)

        print("🚀 启动增强模拟HTTP服务器在 http://localhost:8000")
        app.run(port=8000, debug=False)


def run_mock_server():
    """运行模拟HTTP服务器"""
    EnhancedMockHttpServer.run_mock_server()


if __name__ == "__main__":
    # 如果要运行模拟服务器，取消下面的注释
    run_mock_server()
