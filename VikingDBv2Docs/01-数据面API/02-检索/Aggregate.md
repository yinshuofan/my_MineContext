<span id="cc192755"></span>
# 概述
聚合统计能指定字段进行分组聚合，并可添加过滤操作，最终得到相应的聚合统计结果，辅助了解数据分布等情况。
<span id="0ffeaac6"></span>
# 接口升级说明

* 对应的V1接口为：https://www.volcengine.com/docs/84313/1399339
* 使用区别：

| |V2接口 |V1接口 |
|---|---|---|
|子索引 |性能和易用性更强的分片研发中，敬请期待。 |partition参数指定 |

<span id="4eb07c15"></span>
# 请求接口
:::tip
请求向量数据库 VikingDB 的 OpenAPI 接口时，可以使用 **ak、sk** **构造签名**进行鉴权。请参见[数据面API调用流程](/docs/84313/1791125)，复制调用示例并填入必要信息
:::

| | | | \
|URI |/api/vikingdb/data/agg |统一资源标识符 |
|---|---|---|
| | | | \
|方法 |POST |客户端对向量数据库服务器请求的操作类型 |
| | | | \
|请求头 |Content-Type: application/json |请求消息类型 |
|^^| | | \
| |Authorization: HMAC-SHA256 *** |鉴权 |


<span id="2008a598"></span>
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
|index_name |string |是 |索引名称 |
| | | | | \
|filter |map |否 |过滤条件，格式见下文。默认为空，不做过滤 |
| | | | | \
|op |string |\
| | |是 |目前仅支持count。使用count算子时，索引中必须至少存在一个string、int64或bool类型的标量索引字段。 |\
| | | | |
| | | | | \
|field |string |否 |对指定字段名进行聚合。字段类型支持string，int64，bool且必须为标量索引字段。 |
| | | | | \
|cond |\
| |map |\
| | |否 |类似SQL里group by的having 子句。仅当field字段存在时，才生效。对于count算子，支持gt，表示仅返回大于阈值的结果项。 |

<span id="c0bbc6ae"></span>
# 响应体参数
公共响应体参数部分见
[数据面API调用流程](/docs/84313/1791125)

| | | | | | \
|参数名 |类型 |子参数 | |说明 |
|---|---|---|---|---|
| | | | | | \
|result |map |agg |map |聚合结果 |
|^^| | | | | \
| | |op |string |算子。目前仅count |
|^^| | | | | \
| | |field |string |请求时设置的field |


<span id="20e7bb92"></span>
# 请求响应示例
<span id="a8a7ffaf"></span>
## 1.查看索引数据总数

* 请求参数

```JSON
req_path = "/api/vikingdb/data/agg"
req_body = {
    "collection_name": "test_coll",
    "index_name": "idx_1",
    "op": "count",
}
```


* 响应参数

```JSON
{
    "code": "Success",
    "message": "The API call was executed successfully.",
    "request_id": "02175438839168500000000000000000000ffff0a003ee4fc3499",
    "result": {
        "op": "count",
        "field": "__NONE__",
        "agg": {
            "__TOTAL__": 4000000
        }
    }
}
```


<span id="c425df67"></span>
## 2.按字段聚合统计数量

* 请求参数

```JSON
req_path = "/api/vikingdb/data/agg"
req_body = {
    "collection_name": "test_coll",
    "index_name": "idx_1",
    "op": "count",
    "field": "city",
    "cond": {
        "gt": 900
    }
}
```


* 响应参数

```JSON
{
    "code": "Success",
    "message": "The API call was executed successfully.",
    "request_id": "02175438839168500000000000000000000ffff0a003ee4fc3499",
    "result": {
        "field": "city",
        "op": "count",
        "agg": {
            "beijing": 1001,
            "shanghai": 999,
        }
    }
}
```


<span id="16ee459a"></span>
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
    req_path = "/api/vikingdb/data/agg"
    req_body = {
        "collection_name": "test_coll",
        "index_name": "idx_1",
        "op": "count",
        "field": "city",
        "cond": {
            "gt": 900
        }
    }
    result = client.do_req(req_method=req_method, req_path=req_path, req_params=req_params, req_body=req_body)
    print("req http status code: ", result.status_code)
    print("req result: \n", result.text)
```
