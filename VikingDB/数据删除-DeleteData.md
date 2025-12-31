<span id="8cb8d0a9"></span>
# 概述
接口用于在指定的 Collection 删除数据，删除数据后需要时间更新到 Index，即一段时间内 Index 中仍可检索到数据。
<span id="be060813"></span>
# 接口升级说明

* 对应的V1接口为：https://www.volcengine.com/docs/84313/1254539
* 使用区别：


| | | | \
| |V2接口 |V1接口 |
|---|---|---|
| | | | \
|指定主键的参数名 |ids |primary_keys |

<span id="041f2915"></span>
# 请求接口
:::tip
请求向量数据库 VikingDB 的 OpenAPI 接口时，可以使用 **ak、sk** **构造签名**进行鉴权。请参见[数据面API调用流程](/docs/84313/1791125)，复制调用示例并填入必要信息
:::

| | | | \
|URI |/api/vikingdb/data/delete |统一资源标识符 |
|---|---|---|
| | | | \
|方法 |POST |客户端对向量数据库服务器请求的操作类型 |
| | | | \
|请求头 |Content-Type: application/json |请求消息类型 |
|^^| | | \
| |Authorization: HMAC-SHA256 *** |鉴权 |


<span id="dc99e7bf"></span>
# 请求体参数

| | | | | \
|参数名 |类型 |必选 |备注 |
|---|---|---|---|
| | | | | \
|resource_id |string |2选1 |资源id |
| | |^^| | \
|collection_name |string |\
| | | |collection名称 |
| | | | | \
|ids |list |2选1 |删除数据的主键列表（主键为int64或string）。最多100条。 |\
| | | |注意： |\
| | | | |\
| | | |* 若为请求参数非法（4xx类型），则会全部失败。 |
|^^|^^|^^|^^| \
| | | | |
| | |^^| | \
|del_all |\
| |bool | |为true时，删除所有数据；默认为false。 |\
| | | |此接口删除所有数据，并不能立刻同步到索引，因此，在一段时间内（5分钟左右），索引内仍可检索到数据 |


<span id="2bae2cc1"></span>
# 响应体参数
参考[数据面API调用流程](/docs/84313/1791125)

<span id="fc383d49"></span>
# 请求响应示例

* 请求参数

```JSON
req_path = "/api/vikingdb/data/delete"
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
    "result": null
}
```


<span id="7ba127f6"></span>
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
    req_path = "/api/vikingdb/data/delete"
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