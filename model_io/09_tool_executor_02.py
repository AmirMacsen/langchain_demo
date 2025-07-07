import os
from typing import ClassVar

import requests
from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool, BaseTool

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


# class PreprocessedWeatherTool(BaseTool):
#     name: ClassVar[str] = "get_current_weather"  # 添加类型注解
#     description: ClassVar[str] = "获取标准化处理后的天气信息"  # 添加类型注解
#     original_tool: BaseTool  # 这个字段已经有类型注解
#
#     def __init__(self, vector_store):
#         super().__init__()
#         self.vector_store = vector_store
#     def _run(self, areaCn: str, **kwargs) -> str:
#         # 执行参数预处理
#         normalized_area = self.preprocess_input(areaCn)
#         # 调用原始工具
#         return self.original_tool.run({"areaCn": normalized_area})
#
#     def preprocess_input(self, user_input: str) -> str:
#         """从Excel加载的预处理逻辑（示例）"""
#         data =
#
#         return mapping_table.get(user_input.strip(), user_input)


def read_excel():
    import pandas as pd
    excel = pd.read_excel("weather_code.xlsx")
    return excel.to_dict(orient="records")

def to_vector(documents: list):
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=os.environ["DASHSCOPE_API_KEY"]
    )
    return Chroma.from_texts(documents, embeddings, collection_name="area_code",
                             persist_directory="area_code_vector_store")


vector_store = to_vector([str(item) for item in read_excel()])

search = vector_store.similarity_search("北京", k=2)
print(search)

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


