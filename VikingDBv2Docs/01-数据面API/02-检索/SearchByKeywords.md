
<span id="24ec8d14"></span>
# 概述
多关键词检索能力将传统的关键词精确匹配与基于向量化的语义检索进行结合，实现精准召回包含一个或多个指定关键词的内容的同时，还能通过向量技术理解查询背后的语义，找到内容相关但可能不含精确关键词的结果。多关键词检索能力适用于带有text字段向量化配置（vectorize参数）的索引，支持多个关键词的检索。同时，针对弱语义、强关键词及短语匹配场景，我们新增了全文检索功能，在创建数据集时，选择**配置“FullText”参数**，即可使用全文检索。该功能支持为最多两个文本字段开启分词处理与关键词检索。全文检索能够弥补纯语义向量检索的局限，与现有的向量检索及过滤能力无缝组合，帮助您在“语义+关键词”的混合召回模式下获得更精准的排序结果。
<span id="2840f777"></span>
# 请求接口
:::tip
请求向量数据库 VikingDB 的 OpenAPI 接口时，可以使用 **ak、sk** **构造签名**进行鉴权。请参见[数据面API调用流程](/docs/84313/1791125)，复制调用示例并填入必要信息
:::

|URI |/api/vikingdb/data/search/keywords |统一资源标识符 |
|---|---|---|
| | | | \
|方法 |POST |客户端对向量数据库服务器请求的操作类型 |
| | | | \
|请求头 |Content-Type: application/json |请求消息类型 |
|^^| | | \
| |Authorization: HMAC-SHA256 *** |鉴权 |


<span id="7f767259"></span>
# 请求体参数
仅列出本接口特有的参数。更多信息请参见[检索公共参数](/docs/84313/1791133)。
使用关键词检索接口，需要满足的前提条件为以下二者至少其一：

1. 数据集带Vectorize配置对text字段做向量化。
2. 数据集带FullText配置。


| | | | | \
|参数名 |必选 |类型 |备注 |
|---|---|---|---|
| | | | | \
|mode |\
| |否 |\
| | |string |\
| | | |提供2种模式： |\
| | | | |\
| | | |1. semantic_then_match。数据集需要带Vectorize配置参数。 |\
| | | |2. bm25。前置条件：数据集需要带FullText配置参数。 |\
| | | | |\
| | | |如果不填此参数，会自动根据数据集的配置，选择模式： |\
| | | | |\
| | | |* 如果带FullText，则用bm25模式 |\
| | | |* 如果带Vectorize，则用semantic_then_match模式 |\
| | | |* 如果两者都带，则默认优先使用bm25模式。 |\
| | | |* 如果两者都不带，则无法使用本接口。 |\
| | | | |\
| | | | |
| | | | | \
|keywords |是 |\
| |（二选一） |\
| | |list<string> |关键词列表，列表元素1-10个，元素不允许为空字符串。元素总长度之和不超过512字节。 |
| |^^| | | \
|query |\
| | |string |\
| | | |检索语句，总长度不超过512字节。 |\
| | | |若为semantic搜索，则多个关键词用" "分隔即可，后台按" "分词，最多提取前10个关键词。 |\
| | | |若为bm25搜索，则使用analyzer处理解析。实际检索时，最多提取前10个关键词。 |\
| | | | |
| | | | | \
|case_sensitive |\
| |否 |bool |只针对semantic_then_match搜索时生效。 |\
| | | |是否大小写严格。默认false。 |\
| | | |bm25搜索时会使用analyzer处理。 |
| | | | | \
|fields |\
| |否 |\
| | |list<string> |只针对bm25搜索时生效。 |\
| | | |如果配置了多个全文检索字段，则这里可以指定在哪些字段里进行检索。 |
| | | | | \
|bm25_k1 |\
| |否 |float |只针对bm25搜索时生效。 |\
| | | |bm25的k1参数。默认1.25。用于调节词频对相关性分数的影响。要求大于`0`，通常取值在 `1.2` 到 `2.0` 之间。`k_1`越大，高频词越能显著提升分数；反之越小，词频越容易饱和，即，高频词的得分增长有限。 |
| | | | | \
|bm25_b |否 |float |只针对bm25搜索时生效。 |\
| | | |bm25的b参数。默认0.75。用于调节文档长度对相关性分数的影响。要求在`0`-`1`之间。`b`越大，则长文档的分数被压缩的越厉害；反之，`b`越小，文档长度的影响越有限，接近0则长文档和短文档平等对待。 |

实际检索时，最多提取前10个关键词。关键词内容的总长度不超过512字节。
<span id="5d62b48a"></span>
# 响应体参数
见[检索公共参数](/docs/84313/1791133)。

<span id="c04ea3ac"></span>
# 请求响应示例

* 请求参数

```JSON
req_path = "/api/vikingdb/data/search/keywords"
req_body = {
    "collection_name": "test_coll_with_vectorize",
    "index_name": "idx_1",
    "keywords": ["火山", "向量", "检索", "亿"],
    "output_fields": [
        "f_text"
    ],
    "limit": 5
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
                    "f_text": "支持百亿级向量检索规模"
                },
                "score": 3.605649671017216,
                "ann_score": 0.34465837478637695
            },
            {
                "id": "uid_002",
                "fields": {
                    "f_text": "向量相似度检索是一种基于向量空间模型的检索方法"
                },
                "score": 2.583801264623392,
                "ann_score": 0.269525408744812,
            },
            {
                "id": "uid_003",
                "fields": {
                    "f_text": "向量是指在数学中具有一定大小和方向的量，文本、图片、音视频等非结构化数据"
                },
                "score": 0.269525408744812,
                "ann_score": 0.31852447986602783
            },
            {
                "id": "uid_004",
                "fields": {
                    "f_text": "该数据库内置多种火山引擎自研索引算法"
                },
                "score": 1.5971799596522382,
                "ann_score": 0.3151528239250183
            },
            {
                "id": "uid_005",
                "fields": {
                    "f_text": "可广泛应用于智能问答、智能搜索、推荐系统和数据去重等领域"
                },
                "score": 0.5667128028680034,
                "ann_score": 0.21270805597305298
            }
        ],
        "total_return_count": 5,
        "token_usage": {
            "doubao-embedding-vision__250328": {
                "prompt_tokens":53,
                "completion_tokens":0,
                "image_tokens":0,
                "total_tokens":53
            }
        }
    }
}
```


<span id="0ef3b357"></span>
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
    req_path = "/api/vikingdb/data/search/keywords"
    req_body = {
        "collection_name": "test_coll_with_vectorize",
        "index_name": "idx_1",
        "keywords": ["火山", "向量", "检索", "亿"],
        "output_fields": [
            "f_text"
        ],
        "limit": 5
    }
    result = client.do_req(req_method=req_method, req_path=req_path, req_params=req_params, req_body=req_body)
    print("req http status code: ", result.status_code)
    print("req result: \n", result.text)
```