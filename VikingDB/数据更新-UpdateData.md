<span id="28cfff14"></span>
# 概述
接口用于为已存在数据的部分字段进行更新。支持 text、标量字段、vector 字段的更新。
<span id="5afc9656"></span>
# 接口升级说明

* 对应的V1接口为：https://www.volcengine.com/docs/84313/1400258
* 使用区别：


| | | | \
| |V2接口 |V1接口 |
|---|---|---|
| | | | \
|写入数据对应的参数 |data |fields |
| | | | \
|返回模型消耗token量 |自动返回 |需手动设置 |
| | | | \
|单次写入的数据量限制 |如果数据集是带vectorize的，不超过1，时延更低；不带vectorize的，不超过100。 |不超过100。 |


* 注：QPS限流是以数据条数计算，V1与V2接口的限流行为完全相同。

<span id="901343e5"></span>
# 请求接口
:::tip
请求向量数据库 VikingDB 的 OpenAPI 接口时，可以使用 **ak、sk** **构造签名**进行鉴权。请参见[数据面API调用流程](/docs/84313/1791125)，复制调用示例并填入必要信息
:::

| | | | \
|URI | /api/vikingdb/data/update |统一资源标识符 |
|---|---|---|
| | | | \
|方法 |POST |客户端对向量数据库服务器请求的操作类型 |
| | | | \
|请求头 |Content-Type: application/json |请求消息类型 |
|^^| | | \
| |Authorization: HMAC-SHA256 *** |鉴权 |


<span id="35e97e10"></span>
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

:::tip
注意：数据插入时主键不能为0
:::

* data可以只包括部分字段。（但必须包含主键）
* 暂不支持异步写入

<span id="1978d160"></span>
# 响应体参数
公共响应体参数部分见
[数据面API调用流程](/docs/84313/1791125)

<span id="a77d4475"></span>
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
    req_path = "/api/vikingdb/data/update"
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