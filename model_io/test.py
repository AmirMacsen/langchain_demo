import os
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable
from langchain_community.chat_models import ChatTongyi
from pydantic import BaseModel, Field
from typing import List, Optional

# 定义结构化字段模型
class PatientRecord(BaseModel):
    name: str = Field(description="患者姓名")
    age: int = Field(description="患者年龄，必须为正整数")
    symptoms: List[str] = Field(description="患者症状列表，每个症状独立")
    diagnosis: str = Field(description="诊断结果")
    is_urgent: Optional[bool] = Field(description="是否紧急", default=False)

# Json 输出格式解析器
parser = JsonOutputParser(pydantic_object=PatientRecord)
format_instructions = parser.get_format_instructions()

# 示例样本（你可以继续扩展）
examples = [
    {
        "input": "陈丽，女，28岁，孕36周，突发头痛、视物模糊，血压180/110mmHg，诊断为子痫前期。",
        "output": {
            "name": "陈丽",
            "age": 28,
            "symptoms": ["头痛", "视物模糊"],
            "diagnosis": "子痫前期",
            "is_urgent": True
        }
    },
    {
        "input": "李强，男，45岁，主诉胸闷、气短，心电图提示心肌缺血，诊断为不稳定型心绞痛，需紧急处理。",
        "output": {
            "name": "李强",
            "age": 45,
            "symptoms": ["胸闷", "气短"],
            "diagnosis": "不稳定型心绞痛",
            "is_urgent": True
        }
    }
]

# 示例模板
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="输入：{input}\n输出：{output}"
)

# 自定义提示词模板（使用你提供的结构）
few_shot_prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    prefix=(
        "你是医学病历信息抽取专家。\n"
        "请仅根据 **输入文本** 提取如下字段：\n\n"
        "- name（患者姓名）\n"
        "- age（年龄，正整数）\n"
        "- symptoms（症状，按列表格式列出）\n"
        "- diagnosis（诊断结果）\n"
        "- is_urgent（是否紧急，true/false）\n\n"
        "请严格按照以下格式输出：\n{{ format_instructions }}\n\n"
        "以下是一些示例："
    ),
    suffix="现在请从下面的文本中提取字段：\n\n输入：{{ medical_text }}\n输出：",
    input_variables=["medical_text"],
    partial_variables={"format_instructions": format_instructions},
    template_format="jinja2"
)

# 模型实例
llm = ChatTongyi(api_key=os.environ["DASHSCOPE_API_KEY"])

# 构建链
chain: Runnable = few_shot_prompt | llm | parser

# 测试一条输入
test_input = "王芳，女，32岁，头晕、心悸，诊断为甲状腺功能亢进。"
result = chain.invoke({"medical_text": test_input})

print(result)
