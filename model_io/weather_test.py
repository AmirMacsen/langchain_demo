import os

from dotenv import load_dotenv

load_dotenv()

import requests

app_code = os.environ.get("WEATHER_APP_CODE")


# Step 1.构建请求
url = "https://getweather.market.alicloudapi.com/lundear/weather1d"

# Step 2.设置查询参数
params = {
    "areaCode": "110000"
}


headers = {
    'Authorization': 'APPCODE ' + app_code
}

# Step 3.发送GET请求
response = requests.get(url, params=params, headers=headers)

# Step 4.解析响应
data = response.json()
print(data)
