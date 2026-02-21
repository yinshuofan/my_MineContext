<span id="578f3ca1"></span>
# 概述
随机检索是一种在未指定查询内容的情况下，从数据集中随机返回若干条记录的检索方式。随机检索同样支持过滤和对检索结果的后处理，可用于对比召回效果、数据过滤等场景。

<span id="bb4b8363"></span>
# 请求接口
:::tip
请求向量数据库 VikingDB 的 OpenAPI 接口时，可以使用 **ak、sk** 或**构造签名**进行鉴权。
:::

| | | | \
|URI |/api/vikingdb/data/search/random |统一资源标识符 |
|---|---|---|
| | | | \
|方法 |POST |客户端对向量数据库服务器请求的操作类型 |
| | | | \
|请求头 |Content-Type: application/json |请求消息类型 |
|^^| | | \
| |Authorization: HMAC-SHA256 *** |鉴权 |

<span id="45b41c86"></span>
# 请求体参数
无特殊参数。更多信息请参见[检索公共参数](/docs/84313/1791133)。
<span id="3388a9cd"></span>
# 请求响应示例

* 请求参数

```JSON
req_path = "/api/vikingdb/data/search/random"
req_body = {
    "collection_name": "test_coll",
    "index_name": "idx_1",
    "limit": 2
}
```


* 响应参数

```JSON
{
    "code": "Success",
    "message": "The API call was executed successfully.",
    "request_id": "02175438839168500000000000000000000ffff0a003ee4fc3499",
    "result": {
        "data": [
            {
                "id": "uid_001",
                "fields": {
                    "f_good_id": "uid_001", "f_price": 999000
                },
                "score": 1.000000,
                "ann_score": 1.000000
            },
            {
                "id": "uid_002",
                "fields": {
                    "f_good_id": "uid_002", "f_price": 309000
                },
                "score": 0.88232,
                "ann_score": 0.88232,
            }
        ],
        "total_return_count": 2
    }
}
```


<span id="1b3ae972"></span>
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
    req_path = "/api/vikingdb/data/search/random"
    req_body = {
        "collection_name": "test_coll",
        "index_name": "idx_1",
        "limit": 2
    }
    result = client.do_req(req_method=req_method, req_path=req_path, req_params=req_params, req_body=req_body)
    print("req http status code: ", result.status_code)
    print("req result: \n", result.text)
```