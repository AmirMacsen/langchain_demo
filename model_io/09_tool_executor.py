import os

import requests
from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool


load_dotenv()

@tool
def get_current_weather(areaCn: str) -> str:
    """
    获取指定城市的天气信息。

    :param areaCn: 比如北京。
    :return: 天气信息。
    """
    app_code = os.environ.get("WEATHER_APP_CODE")

    # Step 1.构建请求
    url = "https://getweather.market.alicloudapi.com/lundear/weather1d"

    # Step 2.设置查询参数
    params = {
        "areaCn": areaCn
    }

    headers = {
        'Authorization': 'APPCODE ' + app_code
    }

    # Step 3.发送GET请求
    response = requests.get(url, params=params, headers=headers)
    # Step 4.解析响应
    data = response.json()

    return data

@tool
def get_stock_price(stock_code: str) -> str:
    """
    获取指定股票的实时价格。

    :param stock_code: 比如股票的code。
    :return: 股票价格。
    """
    return f"股票 {stock_code} 的实时价格是 10.00"


prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个人工智能助手，你的名字是小明。请输出与用户输入有关的答案，禁止输出其他无关信息"),
    ("human", "用户问题：{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")  # 预留中间步骤占位符
])

agent = create_tool_calling_agent(tools=[get_current_weather, get_stock_price],
                                  prompt=prompt,
                                  llm=ChatTongyi(api_key=os.environ["DASHSCOPE_API_KEY"]))

executor = AgentExecutor(agent=agent, tools=[get_current_weather, get_stock_price], max_iterations=3,
                         handle_parsing_errors=True)
resp = executor.invoke({"input": "今天深圳的天气怎么样？"})
print(resp.get("output"))


