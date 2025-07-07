import os
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from pydantic import BaseModel,Field,field_validator

from model_io.few_shot_examples import struct_output


class PatientRecord(BaseModel):
    name: str = Field(description="患者姓名")
    age: int = Field(description="患者年龄, 必须是整数")
    symptoms: List[str] = Field(description="患者症状列表，每个症状独立拆分")
    diagnosis: str = Field(description="患者诊断结果")
    is_urgent: Optional[bool] = Field(description="患者是否需要加急处理", default=False)

    @field_validator("age")
    def age_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("年龄必须大于0，且为整数")
        return v


parser = JsonOutputParser(pydantic_object=PatientRecord)
format_instructions = parser.get_format_instructions()

embeddings_model = DashScopeEmbeddings(
    model="text-embedding-v1", dashscope_api_key=os.environ["DASHSCOPE_API_KEY"]
)

documents = [" ".join(example.values()) for example in struct_output]
vector_store = Chroma.from_texts(documents, embeddings_model, metadatas=struct_output)

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="输入：{input}\n输出：{output}",
)

selector = SemanticSimilarityExampleSelector(
    vectorstore=vector_store,
    k=1,
    example_keys=["input", "output"],
    input_keys=["medical_text"],
)

few_shot_prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    example_selector=selector,
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

model = ChatTongyi(
    model="qwen-turbo",
    api_key=os.environ["DASHSCOPE_API_KEY"],
    top_p=0.9,  # 增加多样性
    streaming=True,
)

chains = few_shot_prompt | model
parse_chain = parser

medical_texts = [
    "赵敏，女，三十岁，主诉头晕和恶心，诊断为低血压。",                # 需修复：三十岁→30
    "无名氏，年龄未知，主诉腹痛，诊断为急性肠胃炎"                     # 异常值测试
]

for medical_text in medical_texts:
    output = chains.invoke({"medical_text": medical_text})
    print(output)
    print(parse_chain.invoke(output))
