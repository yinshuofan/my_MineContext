
<span id="f2dafdaa"></span>
# 概述
多模态数据检索是指向量数据库支持直接通过图文等多模态数据类型进行检索，且支持模态的组合，如文搜图，图搜图，图搜文+图，文搜视频，图搜视频等。此外，多模态检索支持对检索结果进行重排的能力，以获得更佳精准的召回结果。
:::tip
适用于创建向量库时选择"需要向量化"：当导入的数据是原始数据时，可以通过此接口输入文本、图片等进行检索。
:::
<span id="f07090fb"></span>
# 请求接口
:::tip
请求向量数据库 VikingDB 的 OpenAPI 接口时，可以使用 **ak、sk** **构造签名**进行鉴权。请参见[数据面API调用流程](/docs/84313/1791125)，复制调用示例并填入必要信息
:::

| | | | \
|URI |/api/vikingdb/data/search/multi_modal |统一资源标识符 |
|---|---|---|
| | | | \
|方法 |POST |客户端对向量数据库服务器请求的操作类型 |
| | | | \
|请求头 |Content-Type: application/json |请求消息类型 |
|^^| | | \
| |Authorization: HMAC-SHA256 *** |鉴权 |

:::tip
场景 1：QPS（CPU 占用）超限

* 检索接口 QPS 与 CU 资源绑定，超出当前 CU 对应的 QPS 上限会触发限流（常规配置下 QPS 约 100）。
* 可通过自主编辑索引扩容 CU 资源解决。

场景 2：Token（TPM）超限

* 多模态检索场景下，存在 embedding TPM（Token 每分钟处理量）限制，通常默认为33000 tokens / s，具体可见[向量库配额说明](/docs/84313/1478243)
* 需联系平台技术人员协助扩容。
:::
<span id="ea7eebb8"></span>
# 请求体参数
仅列出本接口特有的用于多模态检索和重排参数。更多信息请参见[检索公共参数](/docs/84313/1791133)。 

| | | | | \
|参数名 |必选 |类型 |备注 |
|---|---|---|---|
| | | | | \
|text |至少选1 |string |检索的文本内容 |\
| | | |默认utf-8编码 |
| |^^| | | \
|image | |string |* 图片tos链接。`tos://{bucket}/{object}` |\
| | | |* http/https格式链接 |
| |^^| | | \
|video | |map |{ |\
| | | |"value": tos链接，http/https格式链接 （该字段必填） |\
| | | |"fps": 2.0 (0.2-5，该字段选填) |\
| | | |} |
| | | | | \
|instruction |否 |map |instruction 配置 |\
| | | |通过设置 instructions 引导模型更准确地聚焦输入内容的关键信息，以提升向量表示精度，详细参考[Instruction](/docs/84313/1791135#483fdff4)： |\
| | | |注：配置 Instruction 后，检索分数可能存在波动，实际情况以检索效果为准 |
| | | | | \
|rerank |否 |map |rerank 重排参数：在召回后，可选rerank对数据进行重排操作 |\
| | | |详见[rerank子参数列表](/docs/84313/1791135#cded1d61) |
| | | | | \
|tensor_rerank |否 |map |张量重排参数：详见[tensor_rerank子参数列表](/docs/84313/1791135#5edefbdd) |

<span id="cded1d61"></span>
### rerank子参数列表

| | | | | \
|参数名 |必选 |类型 |备注 |
|---|---|---|---|
| | | | |
|model_name |是 |string |rerank模型名称。支持的模型列表见[重排序-Rerank](https://bytedance.larkoffice.com/wiki/Yiavwf8KciotHHkHtZycLBLWncd) |
| | | | |
|model_version |是 |string |rerank模型版本。 |
| | | | |
|input_limit |否 |int |召回后进入rerank阶段的数据量。默认取search中limit参数值。但不得超过100。要求：search.limit<=search.rerank.input_limit<=100。 |
| | | |执行search时，则先取rerank_input_limit召回量进行重排，最终返回重排结果中的前search.limit个数据。 |
| | | | |
|instruction |否 |string |rerank指令。默认不填。在特定使用场景下，填写指令可提高rerank效果。下面给出一些参考示例： |
| | | | |
| | | |* Whether the Document answers the Query or matches the content retrieval intent |
| | | |* Whether the Document and Query represent duplicate questions |
| | | |* 查找与Query图片完全相同的图片，Query图片可能经过了PS处理，包含缩放、裁剪和水印，请忽略PS处理痕迹，找到PS处理前完全相同的原图 |
| | | |* 请根据Query中的视频描述，判断Document中的视频截图序列是否与描述内容匹配。Document包含视频的关键帧截图，按时间顺序排列。请综合考虑视频的场景、人物、动作、物品、情感表达等要素，判断两者是否描述同一个视频内容。 |
| | | | |
|score_threshold |否 |float |可选，按rerank分数阈值过滤。小于score_threshold的数据将不会返回。默认值为0 |
| | | |0.5 |
| | | |0.01-0.1 |
| | | | |
|fail_strategy |否 |string |当rerank调用失败时的处理策略： |
| | | | |
| | | |* `fallback`：默认，降级返回向量检索的排序结果。并在返回结果中"rerank_error"字段中显示错误。 |
| | | |* `fail_fast`：快速失败。任何一条数据rerank失败，会导致请求失败。 |
| | | | |
| | | | |
| | | | |
|timeout_ms |否 |int |rerank超时时间。超时则算作rerank失败，进入fail_strategy机制。 |

<span id="5edefbdd"></span>
### tensor_rerank子参数列表

| | | | | \
|参数名 |类型 |必选 |说明 |
|---|---|---|---|
| | | | | \
|input_limit |int |是 |进入张量重排的候选数量，范围：[1, 1000]，默认值：100 |

<span id="483fdff4"></span>
## Instruction

| | | | | | | \
|参数 |一级参数 |类型 |是否必填 |默认值 |描述 |
|---|---|---|---|---|---|
| | | | | | | \
|instruction | |Map |否 | |**Instruction 配置** |\
| | | | | |通过设置 instructions 引导模型更准确地聚焦输入内容的关键信息，以提升向量表示精度 |\
| | | | | |若不填该字段则等价于 AutoFill = true |
|^^| | | | | | \
| |auto_fill |Boolean |是 |true |**自动填充 Instruction 内容** |\
| | | | | |写入数据时，会自动根据模态信息填充 Instruction 内容。如果模型不支持Instruction，填写后不报错但不生效。各模型的填充规则见下表[模型列表](/docs/84313/1791135#4adb0511)。 |

<span id="4adb0511"></span>
## 模型列表

| | | | | | | | | | | | \
|模型名称 |模型版本 |支持向量化类型 |默认稠密向量维度 |可选稠密向量维度 |文本截断长度 |支持稀疏向量 |支持张量 | 可支持instruction |\
| | | | | | | | | |支持入库Instruction |\
| | | | | | | | | |auto fill 内容 |支持检索Instruction |\
| | | | | | | | | | |auto fill 内容 |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | \
|bge-large-zh |(default) |text |1024 |1024 |512 |否 |否 |是 |null |\
| | | | | | | | | | |`为这个句子生成表示以用于检索相关文章：` |\
| | | | | | | | | | |拼接到 text 最前面 |
| | | | | | | | | | | | \
|bge-m3 |(default) |text |1024 |1024 |8192 |是 |否 |否 |null |null |
| | | | | | | | | | | | \
|bge-visualized-m3 |\
| |(default) |text、image及其组合 |1024 |1024 |8192 |否 |否 |否 |null |null |
| | | | | | | | | | | | \
|doubao-embedding |*240715* |text |2048 |\
| | | | |512, 1024, 2048 |4096 |否 |否 |是 |\
| | | | | | | | | |null |`为这个句子生成表示以用于检索相关文章：` |\
| | | | | | | | | | |拼接到 text 最前面 |
| | | | | | | | | | | | \
|doubao-embedding-large |*240915* |text |2048 |512, 1024, 2048, 4096 |4096 |否 |否 |是 |null |`为这个句子生成表示以用于检索相关文章：` |\
| | | | | | | | | | |拼接到 text 最前面 |
| | | | | | | | | | | | \
|doubao-embedding-vision |*250328* |text、image及其组合 |2048 |2048, 1024 |\
| | | | | |8192 |否 |否 |是 |null |`根据这个问题，找到能回答这个问题的相应文本或图片：` |\
| | | | | | | | | | |拼接到 text 最前面 |
| | | | | | | | | | | | \
|doubao-embedding-vision |*250615* |兼容*241215*和*250328*的用法*。​*另外，支持full_modal_seq（文/图/视频序列） |2048 |\
| | | | |2048, 1024 |\
| | | | | |128k |否 |否 |是 |null |`根据这个问题，找到能回答这个问题的相应文本或图片：` |\
| | | | | | | | | | |拼接到 text 最前面 |
| | | | | | | | | | | | \
|doubao-embedding-vision |251215 |兼容*241215*和*250328*的用法，支持full_modal_seq（文/图/视频序列）。另外支持张量 |2048 |2048, 1024 |128k |是 |是 |是 |`Instruction:Compress the {} into one word.\nQuery:` |\
| | | | | | | | | |占位符里根据数据集模态信息（如text、text and image、text and image and video ...）自动填充 |`Target_modality: {}.\nInstruction:根据这个问题，找到能回答这个问题的相应文本或图片\nQuery:` |\
| | | | | | | | | | |占位符里根据数据集模态信息（如text、text and image、text and image and video ...）自动填充 |


<span id="5291ffac"></span>
# 请求响应示例
<span id="390e1fb7"></span>
## 1.文本检索

* 请求参数

```JSON
req_path = "/api/vikingdb/data/search/multi_modal"
req_body = {
    "collection_name": "test_coll_with_vectorize",
    "index_name": "idx_1",
    "text": "向量是指在数学中具有一定大小和方向的量，文本、图片、音视频等非结构化数据",
    "instruction": {
        "auto_fill": true
    },
    "output_fields": [
        "f_text"
    ],
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
                    "f_text": "向量是指在数学中具有一定大小和方向的量"
                },
                "score": 9.899999618530273,
                "ann_score": 9.899999618530273
            },
            {
                "id": "uid_002",
                "fields": {
                    "f_text": "向量是高中数学里的一个重要概念"
                },
                "score": 8.324234999961,
                "ann_score": 8.324234999961,
            }
        ],
        "total_return_count": 2,
        "real_text_query": "根据这个问题，找到能回答这个问题的相应文本或图片：向量是指在数学中具有一定大小和方向的量，文本、图片、音视频等非结构化数据",
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


<span id="90e646da"></span>
## 2.文本+图片检索

* 请求参数

```JSON
req_path = "/api/vikingdb/data/search/multi_modal"
req_body = {
    "collection_name": "test_coll_with_vectorize",
    "index_name": "idx_1",
    "text": "向量是指在数学中具有一定大小和方向的量，文本、图片、音视频等非结构化数据",
    "image": "tos://my_bucket/vector_icon.jpg",
    "output_fields": [
        "f_text", "f_image"
    ],
    "instruction": {
        "auto_fill": true
    },
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
                    "f_text": "向量是指在数学中具有一定大小和方向的量", "f_image": "tos://my_bucket/vector_1.jpg"
                },
                "score": 9.899999618530273,
                "ann_score": 9.899999618530273
            },
            {
                "id": "uid_002",
                "fields": {
                    "f_text": "向量是高中数学里的一个重要概念", "f_image": "tos://my_bucket/vector_2.jpg"
                },
                "score": 8.324234999961,
                "ann_score": 8.324234999961,
            }
        ],
        "total_return_count": 2,
        "real_text_query": "根据这个问题，找到能回答这个问题的相应文本或图片：向量是指在数学中具有一定大小和方向的量，文本、图片、音视频等非结构化数据",
        "token_usage": {
            "doubao-embedding-vision__250328": {
                "prompt_tokens":1335,
                "completion_tokens":0,
                "image_tokens":1231,
                "total_tokens":1335
            }
        }
    }
}
```


<span id="32f10776"></span>
## 3.视频检索

* 视频token计入image_tokens

```SQL
req_path = "/api/vikingdb/data/search/multi_modal"
req_body = {
        "collection_name": "jiangyuan_video_collection",
        "index_name": "jiangyuan_video_index",
        "text": "猫",
        "video": 
            {   
                "value": "tos://data-import/2101858484/2025_08_19_10_51_40Oheg6Pua9jaD33cVl2GhcKyjumEM7aXy/cat_video.mp4",
                "fps": 1.0,
            },
        "output_fields": [
                "f_id", "f_video"
            ],
        "instruction": {
            "auto_fill": false
        },
        "limit": 2
 }
```

```JSON
{
    "code": "Success",
    "message": "The API call was executed successfully.",
    "request_id": "02175574557162300000000000000000000ffff0a007af7628242",
    "result": {
        "data": [
            {
                "id": "4",
                "fields": {
                    "f_id": "4",
                    "f_video": {
                         "value": "tos://my_bucket/xxxx1.mp4",
                         "fps": 1.0,
                     }
                },
                "score": 0.9932262897491455,
                "ann_score": 0.9932262897491455
            },
            {
                "id": "1",
                "fields": {
                    "f_id": "1",
                    "f_video": {
                         "value": "tos://my_bucket/xxxx2.mp4",
                         "fps": 1.0,
                     }
                },
                "score": 0.41645175218582153,
                "ann_score": 0.41645175218582153
            }
        ],
        "total_return_count": 2,
        "real_text_query": "猫",
        "token_usage": {
            "doubao-embedding-vision__250615": {
                "prompt_tokens": 18146,
                "completion_tokens": 0,
                "image_tokens": 17892,
                "total_tokens": 18146
            }
        }
    }
}
```

<span id="1f6545d7"></span>
## 4.带rerank的图文检索

* 请求参数

```JSON
{
    "collection_name": "test_coll_with_vectorize",
    "index_name": "idx_1",
    "text": "向量是指在数学中具有一定大小和方向的量，文本、图片、音视频等非结构化数据",
    "image": "tos://my_bucket/vector_icon.jpg",
    "output_fields": [
        "f_text", "f_image"
    ],
    "instruction": {
        "auto_fill": true
    },
    "limit": 2,
    "rerank": {
        "model_name": "doubao-seed-rerank",
        "model_version": "251028",
        "instruction": "Whether the Document and Query represent duplicate questions"
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
        "data": [
            {
                "id": "uid_001",
                "fields": {
                    "f_text": "向量是指在数学中具有一定大小和方向的量", "f_image": "tos://my_bucket/vector_1.jpg"
                },
                "score": 0.51,
                "ann_score": 9.899999618530273
            },
            {
                "id": "uid_002",
                "fields": {
                    "f_text": "向量是高中数学里的一个重要概念", "f_image": "tos://my_bucket/vector_2.jpg"
                },
                "score": 0.11,
                "ann_score": 8.324234999961
            }
        ],
        "total_return_count": 2,
        "real_text_query": "根据这个问题，找到能回答这个问题的相应文本或图片：向量是指在数学中具有一定大小和方向的量，文本、图片、音视频等非结构化数据",
        "token_usage": {
            "doubao-embedding-vision__250328": {
                "prompt_tokens":1335,
                "completion_tokens":0,
                "image_tokens":1231,
                "total_tokens":1335
            },
            "doubao-seed-rerank__251028": {
                "prompt_tokens":1335,
                "completion_tokens":0,
                "image_tokens":1231,
                "total_tokens":1335
            }
        }
    }
}
```

<span id="1fa23534"></span>
## 5.带tensor_rerank的图文检索

* 请求参数

```Plain Text
{
    "collection_name": collection_name,
    "index_name": index_name,
    "limit": 3,
    "output_fields": ["id", "name", "dense_text_vector", "sparse_text_vector", "tensor_text_vector"],
    "text": "sky is very blue",
    "instruction": {
        "auto_fill": false
    },
    "tensor_rerank" : {
        "input_limit" : 50,
    },
    "advance": {
        "dense_weight": 0.7
    }
}
```


* 响应参数

```Plain Text
req http status code:  200
req result: 
 {
  "code": "Success",
  "message": "The API call was executed successfully.",
  "request_id": "02176373667508300000000000000000000ffff0a00730abcf2fa",
  "result": {
    "data": [
      {
        "id": 85,
        "fields": {
          "dense_text_vector": "OUmJbeyyyplieGsPHTft",
          "id": 85,
          "name": "FxkdlaxJEe",
          "sparse_text_vector": "ImUZkmTZLxXVLJEwEIDr",
          "tensor_text_vector": "KIEdrgNtIjKmAPdgoNdS"
        },
        "score": 225083.109375,
        "ann_score": 0.18097847700119019
      },
      {
        "id": 40,
        "fields": {
          "dense_text_vector": "BvUISeyzoEBxduVUPfhO",
          "id": 40,
          "name": "hItTJqQtbJ",
          "sparse_text_vector": "NlNAvNyOJDXZFHiJLyeD",
          "tensor_text_vector": "IIVuNrxYInoebjVGTLCf"
        },
        "score": 224653.296875,
        "ann_score": 0.18653953075408936
      },
      {
        "id": 90,
        "fields": {
          "dense_text_vector": "EGEoGsKBLexTzLXfcFAz",
          "id": 90,
          "name": "jLwxYLXUzv",
          "sparse_text_vector": "KAgXCPHqBQYyXUlpTKkk",
          "tensor_text_vector": "PggVnXHcaNucUPSWnSRy"
        },
        "score": 224428.125,
        "ann_score": 0.21451199054718018
      }
    ],
    "total_return_count": 3,
    "real_text_query": "sky is very blue",
    "token_usage": {
      "doubao-embedding-vision__250615": {
        "prompt_tokens": 29,
        "completion_tokens": 0,
        "image_tokens": 0,
        "total_tokens": 29
      }
    }
  }
}
```

<span id="5c8de2d7"></span>
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
    req_path = "/api/vikingdb/data/search/multi_modal"
    req_body = {
        "collection_name": "test_coll_with_vectorize",
        "index_name": "idx_1",
        "text": "向量是指在数学中具有一定大小和方向的量，文本、图片、音视频等非结构化数据",
        "instruction": {
            "auto_fill": true
        },
        "output_fields": [
            "f_text"
        ],
        "limit": 2
    }
    result = client.do_req(req_method=req_method, req_path=req_path, req_params=req_params, req_body=req_body)
    print("req http status code: ", result.status_code)
    print("req result: \n", result.text)
```