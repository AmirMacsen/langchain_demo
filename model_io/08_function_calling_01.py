import os

import requests
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import JsonOutputKeyToolsParser

load_dotenv()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "当你想查询指定城市的天气时非常有用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市或县区，比如北京市、杭州市、余杭区等。",
                    }
                },
                "required": ["location"]
            }
        }
    }
]


def get_current_weather(area_code: str) -> str:
    """
    获取指定城市的天气信息。

    :param area_code: 比如北京的code。
    :return: 天气信息。
    """
    app_code = os.environ.get("WEATHER_APP_CODE")

    # Step 1.构建请求
    url = "https://getweather.market.alicloudapi.com/lundear/weather1d"

    # Step 2.设置查询参数
    params = {
        "areaCode": area_code
    }

    headers = {
        'Authorization': 'APPCODE ' + app_code
    }

    # Step 3.发送GET请求
    response = requests.get(url, params=params, headers=headers)
    # Step 4.解析响应
    data = response.json()

    return data

llm = ChatTongyi(api_key=os.environ["DASHSCOPE_API_KEY"])

bind_tools = llm.bind_tools(tools)

chains = (bind_tools
          | JsonOutputKeyToolsParser(key_name='get_current_weather', first_tool_only=True)
          | get_current_weather)
invoke = chains.invoke("今天北京的天气怎么样？")
print(invoke)
