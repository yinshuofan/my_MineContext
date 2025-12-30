<span id="4792cf6d"></span>
# 概述
接口用于根据主键在指定的 Index 查询单条或多条数据，单次最多可查询100条数据。
<span id="05ca1861"></span>
# 接口升级说明

* 对应的V1接口为：https://www.volcengine.com/docs/84313/1254532
* 使用区别：


| | | | \
| |V2接口 |V1接口 |
|---|---|---|
| | | | \
|指定主键的参数名 |ids |primary_keys |
| | | | \
|响应字段 |主键值会单独列出，便于定位数据。 |所有字段均在同一级参数中 |

<span id="f4b9fec3"></span>
# 请求接口
:::tip
请求向量数据库 VikingDB 的 OpenAPI 接口时，可以使用 **ak、sk** **构造签名**进行鉴权。请参见[数据面API调用流程](/docs/84313/1791125)，复制调用示例并填入必要信息
:::

| | | | \
|URI |/api/vikingdb/data/fetch_in_collection |统一资源标识符 |
|---|---|---|
| | | | \
|方法 |POST |客户端对向量数据库服务器请求的操作类型 |
| | | | \
|请求头 |Content-Type: application/json |请求消息类型 |
|^^| | | \
| |Authorization: HMAC-SHA256 *** |鉴权 |


<span id="4fd4cf67"></span>
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
|ids |list<int64>或 |\
| |list<string> |是 |- |* 点查数据的主键列表。最多100条 |
|^^|^^|^^| |^^| \
| | | |- | |


<span id="e9315842"></span>
# 响应体参数
公共响应体参数部分见
[数据面API调用流程](/docs/84313/1791125)

| | | | | | \
|参数名 |类型 |子参数 | |说明 |
|---|---|---|---|---|
| | | | | | \
|result |map |fetch |list<FetchResult> |查询到的数据列表，FetchResult结构见下。 |
|^^|^^| | | | \
| | |ids_not_exist |\
| | | |list<string>或list<int64> |\
| | | | |不存在的主键列表 |


<span id="c798137b"></span>
## FetchResult

| | | | \
|参数名 |类型 |备注 |
|---|---|---|
| | | | \
|id |string/int64 |主键值 |
| | | | \
|fields |map |key为字段名，value为字段值 |

<span id="afd1ef36"></span>
# 请求响应示例

* 请求参数

```JSON
req_path = "/api/vikingdb/data/fetch_in_collection"
req_body = {
    "collection_name": "test_coll",
    "ids": [
        "uid_001",
        "uid_002",
        "uid_005"
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
        "fetch":[
            {
                "id": "uid_001",
                "fields": {
                    "f_good_id": "uid_001", "f_text": "这是一件唐朝的文物，距今1300年。"
                }
            },
            {
                "id": "uid_002",
                "fields": {
                    "f_good_id": "uid_002", "f_text": "这是一件清朝的文物，距今300年。"
                }
            },
        ],
        "ids_not_exist": [
            "uid_005"
        ]
    }
}
```


<span id="456783e1"></span>
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
    req_path = "/api/vikingdb/data/fetch_in_collection"
    req_body = {
        "collection_name": "test_coll",
        "ids": [
            "uid_001",
            "uid_002",
            "uid_005"
        ]
    }
    result = client.do_req(req_method=req_method, req_path=req_path, req_params=req_params, req_body=req_body)
    print("req http status code: ", result.status_code)
    print("req result: \n", result.text)
```