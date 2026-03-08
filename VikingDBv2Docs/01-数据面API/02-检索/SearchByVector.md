<span id="a9003a18"></span>
# 概述
接口用于实现向量检索。向量检索是一种基于向量空间模型的检索方法，通过计算向量之间的相似度进行检索。在一个给定向量数据集中，向量检索按照某种度量方式（比如内积、欧式距离），对向量构建的一种时间和空间上比较高效的数据结构，能够高效地检索出与目标向量相似的 K 个向量。
:::tip
适用于创建向量库时选择"已有向量数据" ：当导入的数据是向量时，可以通过此接口输入向量进行检索。
:::
<span id="6bd2eb79"></span>
# 请求接口
:::tip
请求向量数据库 VikingDB 的 OpenAPI 接口时，可以使用 **ak、sk** **构造签名**进行鉴权。请参见[数据面API调用流程](/docs/84313/1791125)，复制调用示例并填入必要信息
:::

|URI |/api/vikingdb/data/search/vector |统一资源标识符 |
|---|---|---|
| | | | \
|方法 |POST |++客户++端对向量数据库服务器请求的操作类型 |
| | | | \
|请求头 |Content-Type: application/json |请求消息类型 |
|^^| | | \
| |Authorization: HMAC-SHA256 *** |鉴权 |

<span id="275a0ede"></span>
# 
<span id="7936b699"></span>
# 请求体参数
仅列出本接口特有的参数。更多信息请参见[检索公共参数](/docs/84313/1791133)。

| | | | | \
|参数名 |必选 |类型 |备注 |
|---|---|---|---|
| | | | | \
|dense_vector |是 |list<float32> |检索的稠密向量。 |
| | | | | \
|sparse_vector |否 |map<string,float32> |检索的稀疏向量。 |
| | | | | \
|tensor_rerank |否 |map |张量重排子结构，详见：[Tensor Rerank 结构](/docs/84313/1791133#064b549e) |


<span id="4780d3ff"></span>
### 请求响应示例
<span id="28ab5469"></span>
#### 1.稠密向量检索

* 请求参数

```JSON
req_path = "/api/vikingdb/data/search/vector"
req_body = {
    "collection_name": "test_coll",
    "index_name": "idx_1",
    "dense_vector": [0.1243, -0.344345, 0.43232, ......],
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
                "score": 9.899999618530273,
                "ann_score": 9.899999618530273
            },
            {
                "id": "uid_002",
                "fields": {
                    "f_good_id": "uid_002", "f_price": 309000
                },
                "score": 8.324234999961,
                "ann_score": 8.324234999961,
            }
        ],
        "total_return_count": 2
    }
}
```


<span id="c4e653c3"></span>
#### 2.稠密+稀疏向量检索

* 请求参数

```JSON
req_path = "/api/vikingdb/data/search/vector"
req_body = {
    "collection_name": "test_coll",
    "index_name": "idx_1",
    "dense_vector": [0.1243, -0.344345, 0.43232, ......],
    "sparse_vector": {"宋": 0.1, "官窑": 0.5}
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
                "score": 10.899999618530273,
                "ann_score": 10.899999618530273
            },
            {
                "id": "uid_002",
                "fields": {
                    "f_good_id": "uid_002", "f_price": 309000
                },
                "score": 4.324234999961,
                "ann_score": 4.324234999961,
            }
        ],
        "total_return_count": 2
    }
}
```


<span id="e55da838"></span>
#### 2.稠密向量检索+张量重排

* 请求参数

```Plain Text
req_path = "/api/vikingdb/data/search/vector"
req_body = {
    "collection_name": collection_name,
    "index_name": index_name,
    "dense_vector": [0.1243, -0.344345, 0.43232, ......],
    "limit": 5,
    "output_fields": ["id", "name"],
    "tensor_rerank" : {
        "tensor" : [
            [0.1,0.2,...],
            [0.3,0.4,...],
            ...
            [0.8,0.9,...]
        ],
        "input_limit" : 50
    }
}
```


* 响应参数

```JSON
{
  "code": "Success",
  "message": "The API call was executed successfully.",
  "request_id": "02176373685223400000000000000000000ffff0a00730a97ec44",
  "result": {
    "data": [
      {
        "id": 35,
        "fields": {
          "id": 35,
          "name": "PGKgzIpXyr"
        },
        "score": 142.28414916992188,
        "ann_score": 0.7548813819885254
      },
      {
        "id": 40,
        "fields": {
          "id": 40,
          "name": "XhGVoRiLRn"
        },
        "score": 140.0298309326172,
        "ann_score": 0.7593233585357666
      },
      {
        "id": 87,
        "fields": {
          "id": 87,
          "name": "xoiNIrsiQS"
        },
        "score": 139.07171630859375,
        "ann_score": 0.7538418173789978
      },
      {
        "id": 81,
        "fields": {
          "id": 81,
          "name": "HkEKJIWWtL"
        },
        "score": 138.91444396972656,
        "ann_score": 0.7579569220542908
      },
      {
        "id": 64,
        "fields": {
          "id": 64,
          "name": "tPpbFuOTmC"
        },
        "score": 138.3961181640625,
        "ann_score": 0.7552835941314697
      }
    ],
    "total_return_count": 5
  }
}
```

<span id="c06f8475"></span>
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
    req_path = "/api/vikingdb/data/search/vector"
    req_body = {
        "collection_name": "test_coll",
        "index_name": "idx_1",
        "dense_vector": [0.1243, -0.344345, 0.43232, ......],#查询的向量
        "limit": 2
    }
        result = client.do_req(req_method=req_method, req_path=req_path, req_params=req_params, req_body=req_body)
    print("req http status code: ", result.status_code)
    print("req result: \n", result.text)
```