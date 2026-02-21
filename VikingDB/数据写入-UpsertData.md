<span id="bc399241"></span>
# 概述
接口用于在指定的数据集 Collection 内写入数据。指定写入的数据是一个map，允许单次插入一条数据或者多条数据，单次最多可插入100条数据。
:::warning
在执行upsert操作选择上传数据的fields字段时，**跟创建的数据集类型有关**。
**若数据集带vector的字段（选择“已有向量数据”）：​**请仅上传“vector”字段，不要上传“text”或 “image”字段。两者不能同时上传。
**若数据集带vectorize字段（选择“从向量化开始”）：​**请根据需求上传text、image字段中的一个或一起上传，不要上传“vector”字段；
若需要上传image图多模态类型字段，请先将图片上传至TOS，并将图片的tos存储路径传入"image"字段。
:::
<span id="d6789021"></span>
# 接口升级说明

* 对应的V1接口为：https://www.volcengine.com/docs/84313/1254533
* 使用区别：


| | | | \
| |V2接口 |V1接口 |
|---|---|---|
| | | | \
|写入数据对应的参数 |data |fields |
| | | | \
|返回模型消耗token量 |自动返回 |需手动设置 |
| | | | \
|单次写入的数据量限制 |如果数据集是带vectorize的，不超过1，时延更小；不带vectorize的，不超过100。 |不超过100。 |


* 注：QPS限流是以数据条数计算，V1与V2接口的限流行为完全相同。

<span id="7cd8565a"></span>
# 请求接口
:::tip
请求向量数据库 VikingDB 的 OpenAPI 接口时，可以使用 **ak、sk** **构造签名**进行鉴权。请参见[数据面API调用流程](/docs/84313/1791125)，复制调用示例并填入必要信息
:::

   
   | | | | \
   |URI | /api/vikingdb/data/upsert |统一资源标识符 |
   |---|---|---|
   | | | | \
   |方法 |POST |客户端对向量数据库服务器请求的操作类型 |
   | | | | \
   |请求头 |Content-Type: application/json |请求消息类型 |
   |^^| | | \
   | |Authorization: HMAC-SHA256 *** |鉴权 |



<span id="feabb056"></span>
# 请求体参数

| | | | | | \
|参数名 |类型 |必选 |默认值 |备注 |
|---|---|---|---|---|
| | | | | | \
|resource_id |string |2选1 |- |资源id |
| | |^^| | | \
|collection_name |string |\
| | | |- |collection名称 |
| | | | | | \
|data |\
| |array<map> |\
| | |是 |\
| | | |- |单次写入的数据数目不超过100。 |\
| | | | |每条数据作为一个map，其中key为字段名，value为字段值，不同字段类型的字段值格式见下表。 |\
| | | | |注意： |\
| | | | | |\
| | | | |* 不允许写入不存在的字段名。 |\
| | | | |* 如果缺失某字段，则用默认值填充。若字段类型无默认值（如text），则会请求失败。 |\
| | | | |* 若为请求参数非法（4xx类型），则会全部失败。 |
|^^|^^|^^|^^|^^| \
| | | | | |
| | | | | | \
|ttl |int |\
| | |否 |\
| | | |0 |正整数，负数无效 |\
| | | | |当数据不过期时，默认为0。 |\
| | | | |数据过期时间，单位为秒。设置为86400，则1天后数据自动删除。 |\
| | | | |数据ttl删除，不会立刻更新到索引。 |
| | | | | | \
|async |bool |\
| | |否 |\
| | | |false |异步写入开关 |\
| | | | | |\
| | | | |* 异步写入限流阈值为同步写入的10倍 |\
| | | | |* 若数据集带vectorize字段（选择“从向量化开始”），不支持异步写入。 |\
| | | | |* 异步写入的数据不会同步实时的写入collection，滞后时间为分钟级别。可通过接口 FetchDataInCollection来确认数据是否已经写入collection |\
| | | | |* 异步写入的数据不会触发索引的流式更新，索引同步时间为小时级别。可通过接口 FetchDataInIndex接口确认数据是否同步至index |

<span id="7526a476"></span>
# data参数字段值格式
:::tip
注意：数据插入时主键不能为0
:::

| | | | \
|字段类型 |格式 |说明 |
|---|---|---|
| | | | \
|int64 |整型数值 |整数 |
| | | | \
|float32 |浮点数值 |浮点数 |
| | | | \
|string |字符串 |字符串。内容限制256byte |
| | | | \
|bool |true/false |布尔类型 |
| | | | \
|list<string> |字符串数组 |字符串数组 |
| | | | \
|list<int64> |整型数组 |整数数组 |
| | | | \
|vector |* 向量（浮点数数组） |\
| |* float32/float64压缩为bytes后的base64编码 |稠密向量 |\
| | | |
| | | | \
|sparse_vector |\
| | 输入格式<token_id ,token_weight>的字典列表，来表征稀疏稀疏向量的非零位下标及其对应的值, 其中 token_id 是 string 类型, token_weight 是float 类型 |稀疏向量 |\
| | | |
| | | | \
|text |字符串 |若为向量化字段，则值不能为空。（若否，可以为空） |
| | | | \
|image |字符串 |若为向量化字段，则值不能为空。（若否，可以为空） |\
| | | |\
| | |* 图片tos链接 `tos://{bucket}/{object}` |\
| | |* http/https格式链接 |
| | | | \
|video |map |{ |\
| | |"value": `tos://{bucket}/{object}`，http/https格式url链接，该字段必填 |\
| | |"fps": 0.2 （取值0.2-5，选填） |\
| | |} |
| | | | \
|date_time |string |分钟级别： |\
| | |`yyyy-MM-ddTHH:mmZ`或`yyyy-MM-ddTHH:mm±HH:mm` |\
| | |秒级别： |\
| | |`yyyy-MM-ddTHH:mm:ssZ`或`yyyy-MM-ddTHH:mm:ss±HH:mm` |\
| | |毫秒级别： |\
| | |`yyyy-MM-ddTHH:mm:ss.SSSZ`或`yyyy-MM-ddTHH:mm:ss.SSS±HH:mm` |\
| | |例如："2025-08-12T11:33:56+08:00" |
| | | | \
|geo_point |string |地理坐标`longitude,latitude`，其中`longitude`取值(-180,180)，`latitude`取值(-90,90) |\
| | |例如："116.408108,39.915023" |

<span id="228449b6"></span>
## 创建时间类型字段(date_time)
只能填写以下格式的其中一种，全部遵循RFC3339标准（https://datatracker.ietf.org/doc/html/rfc3339）

| | | | | \
| |格式(string) |示例 |说明 |
|---|---|---|---|
| | | | | \
|分钟级别 |`yyyy-MM-ddTHH:mmZ`（utc时间）或 |\
| |`yyyy-MM-ddTHH:mm±HH:mm`（指定时区） |* `2025-08-12T04:34Z` |\
| | |* `2025-08-12T12:34+08:00` |\
| | | |\
| | | |左侧示例都会解析为北京时间`2025-08-12`的`12:34:00`。 |\
| | | | |
| | | | | \
|秒级别 |`yyyy-MM-ddTHH:mm:ssZ`（utc时间）或 |\
| |`yyyy-MM-ddTHH:mm:ss±HH:mm`（指定时区） |\
| | |* `2025-08-12T04:34:56Z` |\
| | |* `2025-08-12T12:34:56+08:00` |\
| | | |\
| | | |左侧示例都会解析为北京时间`2025-08-12`的`12:34:56`。 |\
| | | | |\
| | | | |
| | | | | \
|毫秒级别 |`yyyy-MM-ddTHH:mm:ss.SSSZ`（utc时间）或 |\
| |`yyyy-MM-ddTHH:mm:ss.SSS±HH:mm`（带时区） |* `2025-08-12T04:34:56.147Z` |\
| | |* `2025-08-12T12:34:56.147+08:00` |\
| | | |\
| | | |\
| | | |左侧示例都会解析为北京时间`2025-08-12`的`12:34:56.147`秒。 |\
| | | | |

<span id="d2b5a6ef"></span>
# 响应体参数
公共响应体参数部分见
[数据面API调用流程](/docs/84313/1791125)

| | | | | | \
|参数名 |类型 |子参数 | |说明 |
|---|---|---|---|---|
| | | | | | \
|result |map |token_usage |map |包括prompt_tokens、completion_tokens、image_tokens、total_tokens信息 |

<span id="c15439b7"></span>
# 请求响应示例
<span id="13064d12"></span>
## 1.写入带直接向量字段的数据集

* 请求参数

```JSON
req_path = "/api/vikingdb/data/upsert"
req_body = {
    "collection_name": "test_coll",
    "data": [
        {
            "f_id": "000135", #以下参数为选填，请根据coll创建字段填写
            "f_vector": [0.1, 0.33, -0.88, 0.66],
            "f_city": "北京"
        }
    ]
}
```


* 响应参数

```JSON
{
    "code": "Success",
    "message": "The API call was executed successfully.",
    "request_id": "02175438839168500000000000000000000ffff0a003ee4fc3499",
    "result": null
}
```


<span id="19c23f4b"></span>
## 2.写入带向量化字段的数据集
<span id="4b88a819"></span>
### 图文示例

* 请求参数

```JSON
req_path = "/api/vikingdb/data/upsert"
req_body = {
    "collection_name": "test_coll_with_vectorize",
    "data": [
        {
            "f_id": "000135",
            "f_city": "北京",
            "f_text": "这是一件历史悠久的文物，具有1300年的历史",
            "f_image": "tos://my_bucket/good_000135.jpg"
        }
    ]
}
```


* 响应参数

```JSON
{
    "code": "Success",
    "message": "The API call was executed successfully.",
    "request_id": "02175438839168500000000000000000000ffff0a003ee4fc3499",
    "result": {
        "token_usage": {
            "doubao-embedding-vision__250615": {
                "prompt_tokens": 1325,
                "completion_tokens": 0,
                "image_tokens": 1312,
                "total_tokens": 1325
            }
        }
    }
}
```


<span id="5089f484"></span>
### 视频示例

* 请求参数

```JSON
req_path = "/api/vikingdb/data/upsert"
req_body = {
    "collection_name": "test_coll_with_vectorize",
    "data": [
        {
            "f_id": "000135",
            "f_city": "北京",
            "f_text": "这是一件历史悠久的文物，具有1300年的历史",
            "f_image": "tos://my_bucket/good_000135.jpg",
            "f_video": {
                "value": "tos://my_bucket/good_000135.mp4",
                "fps": 2.0,
             }
        }
    ]
}
```


* 响应参数

```JSON
{
    "code": "Success",
    "message": "The API call was executed successfully.",
    "request_id": "02175438839168500000000000000000000ffff0a003ee4fc3499",
    "result": {
        "token_usage": {
            "doubao-embedding-vision__250615": {
                "prompt_tokens": 1325,
                "completion_tokens": 0,
                "image_tokens": 1312,
                "total_tokens": 1325
            }
        }
    }
}
```


<span id="35a3b083"></span>
## 3. 写入时间、地理字段
<span id="c7b28394"></span>
### 写入时间类型示例
```JSON
req_path = "/api/vikingdb/data/upsert"
req_body = {
    "collection_name": "test",
    "data": [
        {"f_id": 1, "f_time": "2025-08-12T12:34:56+08:00", "f_vector":[...]},
        {"f_id": 2, "f_time": "2025-08-12T11:33:56+08:00", "f_vector":[...]},
        {"f_id": 3, "f_time": "2025-08-12T04:32:56+08:00", "f_vector":[...]}
    ]
}
```

<span id="adbc05e9"></span>
### 写入地理位置类型示例
```JSON
req_path = "/api/vikingdb/data/upsert"
req_body = {
    "collection_name": "test",
    "data": [
        {"f_id": 1, "f_geo_point": "116.403874,39.914885", "f_vector":[...]},
        {"f_id": 2, "f_geo_point": "116.412138,39.914912", "f_vector":[...]},
        {"f_id": 3, "f_geo_point": "116.408108,39.915023", "f_vector":[...]}
    ]
}
```


<span id="88de0952"></span>
# 请求模版
```Plain Text
      
"""
pip3 install volcengine
"""
import os

from volcengine.auth.SignerV4 import SignerV4
from volcengine.Credentials import Credentials
from volcengine.base.Request import Request
import requests, json

class ClientForDataApi:
    def __init__(self, ak, sk, host):
        self.ak = ak
        self.sk = sk
        self.host = host

    def prepare_request(self, method, path, params=None, data=None):
        r = Request()
        r.set_shema("https")
        r.set_method(method)
        r.set_connection_timeout(10)
        r.set_socket_timeout(10)
        mheaders = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Host': self.host,
        }
        r.set_headers(mheaders)
        if params:
            r.set_query(params)
        r.set_host(self.host)
        r.set_path(path)
        if data is not None:
            r.set_body(json.dumps(data))
        credentials = Credentials(self.ak, self.sk, 'vikingdb', 'cn-beijing')
        SignerV4.sign(r, credentials)
        return r
        
    def do_req(self, req_method, req_path, req_params, req_body):
        req = self.prepare_request(method=req_method, path=req_path, params=req_params, data=req_body)
        return requests.request(method=req.method, url="http://{}{}".format(self.host, req.path),
                                  headers=req.headers, data=req.body, timeout=10000)

if __name__ == '__main__':
    client = ClientForDataApi(
        ak = "*",#替换为您的ak
        sk = "*",#替换为您的sk
        host = "api-vikingdb.vikingdb.cn-beijing.volces.com",#替换为您所在的域名
    )
    req_method = "POST"
    req_params = None
    req_path = "/api/vikingdb/data/upsert"
    req_body = {
        "collection_name": "test_coll_with_vectorize",#替换为您的collection名
        "data": [
            {
                "f_id": "000135",
                "f_city": "北京",
                "f_text": "这是一件历史悠久的文物，具有1300年的历史",
                "f_image": "tos://my_bucket/good_000135.jpg"
            }
        ]
    }
    result = client.do_req(req_method=req_method, req_path=req_path, req_params=req_params, req_body=req_body)
    print("req http status code: ", result.status_code)
    print("req result: \n", result.text)
```