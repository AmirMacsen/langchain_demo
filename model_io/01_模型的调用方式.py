import asyncio
import os
import time
from langchain_core.messages import HumanMessage

### 调用chat model
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.language_models import BaseChatModel

### invoke的方式
def test_invoke():
    model = ChatTongyi(
        api_key=os.environ["DASHSCOPE_API_KEY"]
    )
    invoke = model.invoke([HumanMessage(content="你好")])
    print("invoke: ", invoke.content)

async def  test_ainvoke():
    model = ChatTongyi(
        api_key=os.environ["DASHSCOPE_API_KEY"]
    )
    invoke = await model.ainvoke([HumanMessage(content="你好")])
    print("ainvoke: ", invoke.content)

### stream的方式
def test_stream():
    model = ChatTongyi(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        streaming=True
    )
    print("steam: ", end="")
    for chunk in model.stream([HumanMessage(content="你好")]):
        time.sleep(0.5)
        print(chunk.content, sep="", end="", flush=True)

    print()


async def test_astream():
    model = ChatTongyi(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        streaming=True
    )
    print("steam: ", end="")
    async for chunk in model.astream([HumanMessage(content="你好")]):
        time.sleep(0.5)
        print(chunk.content, sep="", end="", flush=True)

    print()

def test_batch():
    model = ChatTongyi(
        api_key=os.environ["DASHSCOPE_API_KEY"]
    )
    batch = model.batch(
        [[HumanMessage(content="你在干什么")], [HumanMessage(content="机器学习现在的发展现状和展望")], ])
    print("batch: ", batch)

if __name__ == '__main__':
    test_invoke()
    asyncio.run(test_ainvoke())
    asyncio.run(test_astream())
    test_stream()
    test_batch()

