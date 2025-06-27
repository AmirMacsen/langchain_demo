import os

from langchain_community.chat_models import ChatTongyi
from langchain_community.llms.tongyi import Tongyi
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


def test_prompt_template():
    template = """
        [system] 你是一个有帮助的AI机器人，你的名字是{name}。
        [human] 你好，最近怎么样？
        [ai] 我很好，谢谢！
        [human] {user_input}
        """
    prompt = PromptTemplate(template=template, input_variables=["name", "user_input"])

    model = ChatTongyi(
        api_key=os.environ["DASHSCOPE_API_KEY"]
    )
    resp = model.invoke(prompt.format(name="小张", user_input="你叫什么名字？"))
    print("normal: ", resp.response_metadata.get("token_usage"))


def test_chat_template():
    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个有帮助的AI机器人，你的名字是{name}。"),
            ("human", "你好，最近怎么样？"),
            ("ai", "我很好，谢谢！"),
            ("human", "{user_input}"),
        ]
    )
    model = ChatTongyi(
        api_key=os.environ["DASHSCOPE_API_KEY"]
    )
    resp = model.invoke(chat_template.format_messages(name="小张", user_input="你叫什么名字？"))
    print("chat: ", resp.response_metadata.get("token_usage"))


if __name__ == '__main__':
    test_prompt_template()
    test_chat_template()
