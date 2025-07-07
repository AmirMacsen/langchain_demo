import os
from typing import List, Dict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

load_dotenv()

def read_excel():
    import pandas as pd
    excel = pd.read_excel("weather_code.xlsx")
    return excel.to_dict(orient="records")


def standard_excel_data(excel_data: List[Dict]) -> List[Dict]:
    """
    标准化Excel数据，生成多语言检索结构
    Args:
        excel_data: 原始Excel数据，每行包含areaCn/areaEn/cityCn等字段
    Returns:
        标准化后的数据列表
    """
    standardized = []
    for item in excel_data:
        # 主数据构建
        std_item = {
            "standard_name": item["areaCn"],
            "code": item["areaCode"],
            "belongs_to": f"{item.get('provCn', '')} {item.get('provEn', '')}".strip(),
            "aliases": [
                item["areaCn"],  # 中文标准名
                item["areaEn"],  # 英文名
                item.get("cityCn", ""),  # 市级中文
                item.get("cityEn", "")  # 市级英文
            ]
        }

        # 生成检索文本（包含所有语言信息）
        std_item["retrieval_text"] = (
            f"标准名称：{std_item['standard_name']}，"
            f"英文名：{std_item['aliases'][1]}，"
            f"编码：{std_item['code']}，"
            f"所属：{std_item['belongs_to']}"
        )
        standardized.append(std_item)
    return standardized


def store_data(standardized_data: List[Dict], persist_dir: str = "./area_code_vector_store"):
    """
    将标准化数据存入Chroma
    """
    # 准备数据
    texts = [item["retrieval_text"] for item in standardized_data]
    metadatas = [{
        "standard_name": item["standard_name"],
        "code": item["code"],
        "province_cn": item["belongs_to"].split()[0],
        "lang_zh": item["aliases"][0],  # 中文版本
        "lang_en": item["aliases"][1]    # 英文版本
    } for item in standardized_data]

    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=os.environ["DASHSCOPE_API_KEY"]
    )
    # 创建向量库
    db = Chroma.from_texts(texts=texts, metadatas=metadatas, embedding=embeddings,
                                   collection_name="geo_locations", persist_directory=persist_dir)


# excel_data = read_excel()
# standardized_data = standard_excel_data(excel_data)
# store_data(standardized_data)

loaddb = Chroma(
    persist_directory="./area_code_vector_store",
    embedding_function=DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=os.environ["DASHSCOPE_API_KEY"]
    ),
    collection_name="geo_locations"
)

print(loaddb.similarity_search("上海", k=2))