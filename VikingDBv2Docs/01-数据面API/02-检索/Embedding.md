
<span id="c4fcc8b1"></span>
# 接口升级说明

* 对应的旧接口为：https://www.volcengine.com/docs/84313/1254554
* 使用区别：

| |新接口 |旧接口 |
|---|---|---|
|模型组合 |稠密和稀疏模型可以独立指定。支持稠密+稀疏、仅稠密、或仅稀疏embedding计算。 |稠密和稀疏模型通过同一个model_name参数指定。 |
|单次写入的数据量限制 |不超过10，时延更低。 |不超过100 |
|返回消耗的模型token用量 |是 |需要额外设置参数才能返回 |


* 注：限流是以token用量计算，V1与V2接口的限流行为完全相同。

<span id="0e1d9636"></span>
# 请求接口
:::tip
* 当前 Embedding 服务支持将文本/图片/视频生成向量。
* 当前对 Embedding 模型设置了 TPM（Tokens Per Minute，每分钟 tokens 数量）的调用限制，每个账号（含主账号下的所有子账号，合并计算）的 TPM 不超过 120000/模型。
* 图片生成向量：
   * 图片大小：建议图片大小不要超过1MB，因embedding v2接口的请求限制为4M，当图片超过1MB时，我们建议用户压缩图片后再次请求，防止接口截断；
   * 图片压缩尺寸推荐：经过我们的实验，将图片的长和宽分别缩放到自身的0.30-0.35倍，可以得到与原图embedding较为相近的结果。其中，0.30-0.35倍 是缩放的拐点，比例再低的话精度劣化会比较明显，缩放比例可以在拐点以上。
   * 当前图片 embedding 限制每秒上传15张图，如果超出限制请及时联系客服扩大限流。
* 视频生成向量：
   * 视频限制：单视频文件需在 50MB 以内（建议30M以内），支持MP4、AVI、MOV格式，暂不支持对视频文件中的音频信息进行理解。
   * 视频支持指定FPS：支持控制从视频中抽取图像的帧率，支持配置0.2-5。
   * 视频embedding请求时长示例：传入30M的视频进行embedding（传入参数是tos路径，视频大小30M，38秒钟，1920 * 1080的尺寸）：fps = 1：耗时8秒；fps = 5：耗时25秒；fps = 0.2：耗时4秒
* 张量：
   * 限制：当前若需要配置张量字段，需将张量化字段名、张量化模型名、张量化模型版本的配置与稠密向量化模型名、稠密向量化模型版本的配置保持一致，且模型固定为"doubao-embedding-vision"，模型版本固定为"251215"
   * 张量阶数：目前固定张量维度阶数为2，可视作向量矩阵
   * 张量维度：格式为[M,N]
      * M：向量矩阵的行数，取值[2,64]间的整数
      * N：向量矩阵的列数，取值[4,2048]间的整数，且当N>4时，N必须是8的倍数
   * 适用场景：
* 上量方式：
   * 上量前需提前通知火山引擎VikingDB相关对接人员，以便做好资源预留准备。
   * 按照梯度爬坡逐步提升用量，具体爬坡策略可参考：起始流量200wTPM，每5分钟涨200wTPM
   * 图片数据的token消耗参考：300KB图片大约消耗1000token。实际token以具体数据为准。
   :::
   <span id="a9cec328"></span>
# 鉴权
:::tip
请求向量数据库 VikingDB 的 OpenAPI 接口时，可以使用 **ak、sk** **构造签名**进行鉴权。请参见[数据面API调用流程](/docs/84313/1791125)，复制调用示例并填入必要信息
:::

| | | | \
|URI |/api/vikingdb/embedding |统一资源标识符 |
|---|---|---|
| | | | \
|方法 |POST |客户端对向量数据库服务器请求的操作类型 |
| | | | \
|请求头 |Content-Type: application/json |请求消息类型 |
|^^| | | \
| |Authorization: HMAC-SHA256 *** |鉴权 |


<span id="3c99f9e1"></span>
# 请求体参数

| | | | | | | | \
|参数 |类型 |是否必选 |子参数 |类型 |是否必选 |说明 |
|---|---|---|---|---|---|---|
| | | | | | | | \
|dense_model |\
| |map |\
| | |3 者至少选 1 |\
| | | |name |\
| | | | |string |是 |\
| | | | | | |模型名 |
|^^|^^|^^| | | | | \
| | | |version |string |否，但豆包模型必选 |模型版本 |
|^^|^^|^^| | | | | \
| | | |dim |int |否 |维度。不填则使用该模型版本的默认维度。 |
|^^|^^|^^| | | | | \
| | | |instruction |map |否 |Instruction 配置 |\
| | | | | | |通过设置 instructions 引导模型更准确地聚焦输入内容的关键信息，以提升向量表示精度，详细参考[instruction 结构](/docs/84313/1791161#7d96226d) |
| | |^^| | | | | \
|sparse_model |map |\
| | | |name |\
| | | | |string |是 |\
| | | | | | |模型名 |
|^^|^^|^^| | | | | \
| | | |version |string |否 |模型版本 |
|^^|^^|^^| | | | | \
| | | |instruction |map |否 |Instruction 配置 |\
| | | | | | |通过设置 instructions 引导模型更准确地聚焦输入内容的关键信息，以提升向量表示精度，详细参考[instruction 结构](/docs/84313/1791161#7d96226d) |
| | |^^| | | | | \
|tensor_model |map | |name |string |是 |模型名称 |
|^^|^^|^^| | | | | \
| | | |version |string |是 |模型版本 |
|^^|^^|^^| | | | | \
| | | |ndim |int |否 |张量阶数，不填则使用该模型版本的默认阶数。目前只能为 2 |
|^^|^^|^^| | | | | \
| | | |shape |int[] |否 |张量维度，不填则使用该模型版本的默认维度。目前只能为 [64, 2048] |
|^^|^^| | | | | | \
| | | |instruction |map |否 |Instruction 配置 |\
| | | | | | |通过设置 instructions 引导模型更准确地聚焦输入内容的关键信息，以提升向量表示精度，详细参考[instruction 结构](/docs/84313/1791161#7d96226d) |
| | | | | | | | \
|data |list<Data> |是 | | | |数据。详细字段见下。列表长度最大 10。如果数据类型是full_modal_seq则长度为1 |

<span id="3c17dc42"></span>
## Data结构

| | | | | \
|参数 |类型 |是否必选 |说明 |
|---|---|---|---|
| | | | | \
|text |string |至少选一个，也可  text、image和video的组合 |\
| | | |文本字符串内容。过长则会截断，各模型的截断阈值见下。 |
| | |^^| | \
|image |\
| |string | |* 图片tos链接。`tos://{bucket}/{object}` |\
| | | |* http/https格式链接 |
| | |^^| | \
|video |map | |{ |\
| | | |"value": `tos://{bucket}/{object}`，http/https格式url链接，该字段必填 |\
| | | |"fps": 0.2 （取值0.2-5，选填） |\
| | | |} |
| | | | | \
|full_modal_seq |list |若选择full_modal_seq，则不能出现上述text等三个参数 |FullModalData结构见下 |


<span id="a2b2a5df"></span>
## MediaData的格式规范
例如image图片、video视频可以为字符串

| | | \
|二选一 |* 同region内的tos资源地址。`tos://{bucket}/{object_key}` |
|^^| | \
| |* 可公开访问的http/https链接。`http://`或`https://` |
|---|---|


<span id="bcb86bbd"></span>
## FullModalData

| | | | | \
|三选一 |字段名 |类型 |备注 |
|^^| | | | \
| |text |\
| | |string |纯文本 |
|^^| | | | \
| |image |string |若无特殊配置参数，使用string类型填入图片资源地址，参考MediaData规范； |
|^^| | | | \
| |video |\
| | |map |\
| | | |若无特殊配置参数，可使用map类型，子参数包括： |\
| | | | |\
| | | |* value：使用string类型填入视频资源地址，参考MediaData规范。 |\
| | | |* fps：表示抽帧的频率。不设置则默认为1，范围为0.2-5.0。不过，服务端默认至少抽取16帧。越大，则抽帧更多，同时消耗的token也越多、时延越高。 |
|---|---|---|---|

<span id="7d96226d"></span>
## instruction 结构

| | | | | | | \
|参数 |一级参数 |类型 |是否必填 |默认值 |描述 |
|---|---|---|---|---|---|
| | | | | | | \
|instruction | |Map |否 | |**Instruction 配置** |\
| | | | | |通过设置 instructions 引导模型更准确地聚焦输入内容的关键信息，以提升向量表示精度 |\
| | | | | |若不填写该字段，则默认不使用 Instruction 配置 |
|^^| | | | | | \
| |content |string |否 | |**填写 Instruction 内容** |\
| | | | | |写入数据时，会自动根据模态信息填充 Instruction 内容。如果模型不支持Instruction，填写后报错。各模型的支持情况见下表[模型列表](/docs/84313/1791161#c6ab6e2a)。 |\
| | | | | |注：若填写该字段，填写内容必须非空，填写内容参考[【向量库】Instruction 参数设置 ](/docs/84313/2178995) |


<span id="c6ab6e2a"></span>
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
| | | | | | | | | |占位符里根据数据集模态信息（如text、text and image、text and image and video ...）填充 |`Target_modality: {}.\nInstruction:根据这个问题，找到能回答这个问题的相应文本或图片\nQuery:` |\
| | | | | | | | | | |占位符里根据数据集模态信息（如text、text and image、text and image and video ...）填充 |

<span id="fd5be7b6"></span>
# 响应体参数
公共响应体参数部分见
[数据面API调用流程](/docs/84313/1791125)

| | | | | | \
|参数名 |类型 |子参数 | |说明 |
|---|---|---|---|---|
| | | | | | \
|request_id |string | | |请求id |
| | | | | | \
|code |int | | |状态码。成功则为0 |
| | | | | | \
|message |string | | |成功则为success |
| | | | | | \
|result |map |data |list<EmbeddingResult> |数据列表 |
| | | | | | \
| | |token_usage |map |按模型粒度的token统计。 |\
| | | | |包括prompt_tokens、completion_tokens、image_tokens、total_tokens信息 |

其中，EmbeddingResult结构：

| | | | \
|参数名 |类型 |说明 |
|---|---|---|
| | | | \
|dense |list<float> |稠密向量结果 |
| | | | \
|sparse |map<string,float> |稀疏向量结果 |


<span id="0e6ff777"></span>
# 请求响应示例
<span id="21f616ad"></span>
## 1.文本 to 稠密向量+稀疏向量

* 请求参数

```JSON
req_path = "/api/vikingdb/embedding"
req_body = {
    "dense_model": {
        "name": "doubao-embedding-large",
        "version": "240915",
        "dim": 1024,
    },
    "sparse_model": {
        "name": "bge-m3",
        "version": "default",
    },
    "data": [
        {
            "text": "天很蓝。"
        },
        {
            "text": "海很深。"
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
        "data": [
            {
                "dense": [0.05149054509186956,0.034275332726816786, ......],
                "sparse": {
                    "天": 0.263671875,
                    "很": 0.18603515625,
                    "蓝": 0.3046875
                }
            },
            {
                "dense": [0.0076801463728645375,0.034275332726816786, ......],
                "sparse": {
                    "很": 0.2010498046875,
                    "海": 0.32958984375,
                    "深": 0.32373046875
                }
            }
        ],
        "token_usage": {
            "bge-m3__default": {
                "prompt_tokens": 14,
                "completion_tokens": 0,
                "image_tokens": 0,
                "total_tokens": 14
            },
            "doubao-embedding-large__240915": {
                "prompt_tokens": 9,
                "completion_tokens": 0,
                "image_tokens": 0,
                "total_tokens": 9
            }
        }
    }
}
```


<span id="2a8976ae"></span>
## 2.文本+图片 to 稠密向量

* 请求参数

```JSON
req_path = "/api/vikingdb/embedding"
req_body = {
    "dense_model": {
        "name": "bge-visualized-m3",
        "version": "default",
        "dim": 1024
    },
    "data": [
        {"text": "天很蓝"},
        {"image": "tos://my_bucket/sky_1.jpeg"}
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
        "data": [
            {
                "dense": [0.013657506555318832,0.034275332726816786, ......]
            }
        ],
        "token_usage": {
            "bge-visualized-m3__default": {
                "prompt_tokens": 926,
                "completion_tokens": 0,
                "image_tokens": 926,
                "total_tokens": 926
            }
        }
    }
}
```


<span id="4f544fcb"></span>
## 3.文本+图片+视频 to 稠密向量

* 请求参数

```JSON
req_path = "/api/vikingdb/embedding"
req_body = {
    "dense_model": {
        "name": "doubao-embedding-vision",
        "version": "250615",
        "dim": 1024,
    },
    "data": [
        {
            "full_modal_seq":[
                {
                    "text": "haha haha"
                },
                {
                    "image": "tos://my_bucket/images/dogs/5.jpeg"
                },
                {
                    "video": "tos://my_bucket/videos/1.mp4"
                },
                {
                    "video":
                    //如果需要对视频有自定义处理，可将video值定义为json，以设置高级参数：
                    //value表示视频内容；
                    //fps表示抽帧的频率。不设置则默认为1，范围为0.2-5.0。不过，服务端默认至少抽取16帧。越大，则抽帧更多，同时消耗的token也越多、时延越高。
                        {
                            "value": "tos://my_bucket/videos/2.mp4",
                            "fps": 0.4, 
                        }
                }
            ]
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
        "data": [
            {
                "dense": [0.05149054509186956,0.034275332726816786, ......]
            }
        ],
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

<span id="9e25e766"></span>
## 4.图片 to 稠密向量（tos 导入）

* 请求参数

```Go
req_path = "/api/vikingdb/embedding"
req_body = {
    "dense_model": {
        "name": "doubao-embedding-vision",
        "version": "250615",
        "dim": 2048,
     },
    "data": [
        {
            "image": {"value":"tos://yuan-tos/cat.jpeg","x-tos-process": "image/resize,l_600|image/indexcrop,x_600,i_0"},
        } 
    ]
}
```


* 响应参数

```JSON
{
    "code":"Success",
    "message":"The API call was executed successfully.",
    "request_id":"xxx",
    "result":
        {
            "data":[
                {
                    "dense":[...],
                }
            ],
            "token_usage":{"doubao-embedding-vision__250615":{"prompt_tokens":472,"completion_tokens":0,"image_tokens":459,"total_tokens":472}}
        }
}   
```

<span id="8d6578d7"></span>
## 5.文本 to 稠密向量+稀疏向量（doubao-embedding-vision）
doubao-embedding-vision 0615模型稀疏向量输出结构为多个{index:value}
index：类型为string，代表token对应的行号
value：token对应的embedding值

* 请求参数

```JSON
req_path = "/api/vikingdb/embedding"
req_body = {
    "dense_model": {
        "name": "doubao-embedding-vision",
        "version": "250615",
        "dim": 2048,
     },
    "sparse_model": {
        "name": "doubao-embedding-vision",
        "version": "250615",
     },
    "data": [
        {
            "text": "从前有座山,山里有座庙", 
        } 
    ]
}
```


* 响应参数

```JSON
{
    "code":"Success",
    "message":"The API call was executed successfully.",
    "request_id":"02175739508084500000000000000000000ffff0a005a4847c517",
    "result":
        {
            "data":[
                {
                    "dense":[...],
                    "sparse"::{"139":0.2431640625,"14310":0.375,"1590":0.35546875,"3129":0.47265625,"37460":0.44140625,"514":0.294921875,"95188":0.408203125}
                }
            ],
            "token_usage":{"doubao-embedding-vision__250615":{"prompt_tokens":23,"completion_tokens":0,"image_tokens":0,"total_tokens":23}}
        }
}   
```

<span id="c7262a2f"></span>
## 6.文本+图片 to 稠密向量+张量
注意：仅doubao-embedding-vision-251215支持张量生成
**普通模式：​**针对每条多模态数据进行向量化，返回每条多模态数据对应的embedding值

* 请求参数

```JSON
{
    "dense_model": {
        "name": "doubao-embedding-vision",
        "version": "251215",
        "dim": 1024,
    },
    "sparse_model": {
        "name": "doubao-embedding-vision",
        "version": "251215",
    },
    "tensor_model": {
        # 这个模型支持 dense/sparse/tensor
        "name": "doubao-embedding-vision",
        "version": "251215",
        // 目前只允许二阶
        "ndim": 2,
        "shape": [64, 2048],
    },
    "data": [
        // 每个 {} 都会各自做 embedding
        {
            "text": "天很蓝"*100,
            "image": "tos://rd-test/test_1.jpg"
        },
        {
            "text": "海很深",
            "image": "tos://rd-test/test_2.jpg"
        }
    ]
}

```


* 响应参数

```JSON
{
  "code": "Success",
  "message": "The API call was executed successfully.",
  "request_id": "202511171606139CC148D9AEEF3402E6AC",
  "result": {
    "data": [
      {
        "dense": [
          -0.058069152178866705,
          -0.013388165641238714,
          ......
          -0.0248406928765152,
          0.013388165641238714
        ],
        "tensor": [
          [
            -0.018070077523589134,
            0.04924188181757927,
            ......
            0.05638158693909645,
            -0.0352160781621933
          ],
          [
            0.1655759960412979,
            -0.08179780840873718,
            ......
            0.02615116909146309,
            -0.08150720596313477
          ],
          [
            0.08374457061290741,
            -0.09642229229211807,
            ......
            -0.0688898041844368,
            -0.11434684693813324
          ],
          [
            0.15993955731391907,
            -0.08979323506355286,
            ......
            -0.048349086195230484,
            -0.10208960622549057
          ],
          [
            0.17710129916667938,
            -0.1038159430027008,
            ......
            0.012168042361736298,
            -0.14460571110248566
          ],
          [
            -0.010962379164993763,
            0.04492606967687607,
            ......
            0.05988454446196556,
            -0.0418911837041378
          ]
        ]
      },
      {
        "dense": [
          0.012257585354308665,
          -0.022775384400263842,
          ......
          -2.8265349292319628e-05,
          -0.013522884487656657
        ],
        "tensor": [
          [
            -0.0021462354343384504,
            0.12765726447105408,
            ......
            0.04326574504375458,
            -0.19384358823299408
          ],
          [
            0.21094878017902374,
            0.03325708955526352,
            ......
            -0.05052570998668671,
            0.019874464720487595
          ],
          [
            0.15182611346244812,
            0.0733555257320404,
            ......
            -0.14868038892745972,
            0.045878272503614426
          ],
          [
            0.1782858967781067,
            0.08701540529727936,
            ......
            -0.08826161175966263,
            0.03097386099398136
          ],
          [
            0.15832433104515076,
            0.08862681686878204,
            ......
            -0.08204027265310287,
            -0.023515628650784492
          ],
          [
            0.006651569157838821,
            0.1267307847738266,
            ......
            0.04425329715013504,
            -0.194008007645607
          ]
        ]
      }
    ],
    "token_usage": {
      "doubao-embedding-vision__250615": {
        "prompt_tokens": 4084,
        "completion_tokens": 0,
        "image_tokens": 0,
        "total_tokens": 4084
      },
      "doubao-embedding-vision__251215": {
        "prompt_tokens": 5032,
        "completion_tokens": 0,
        "image_tokens": 0,
        "total_tokens": 5032
      }
    }
  }
}
```


**full_modal_seq 模式：​**针对整体的多模态序列进行向量化，返回该多模态序列对应的embedding值

* 请求参数

```JSON
{
    "dense_model": {
        "name": "doubao-embedding-vision",
        "version": "251215",
        "dim": 1024,
    },
    "sparse_model": {
        "name": "doubao-embedding-vision",
        "version": "251215",
    },
    "tensor_model": {
        "name": "doubao-embedding-vision",
        "version": "251215",
        "ndim": 2,
        "shape": [64, 2048],
    },
    "data": [
        {
            # 整个 full_modal_seq 作为一个整体进行 embedding
            "full_modal_seq":[
                {
                    "text": "haha haha"
                },
                {
                    "text": "hello world"
                },
                {
                    "image": "tos://liningrui-test/test.jpg"
                }
            ]
        }
    ]
}
```


* 响应参数

```JSON
{
  "code": "Success",
  "message": "The API call was executed successfully.",
  "request_id": "02176397605798400000000000000000000ffff0a006239dfea00",
  "result": {
    "data": [
      {
        "dense": [
            0.03408424784283623,
            -0.005582764732878348,
            ...
        ],
        "tensor": [
          [
            -0.0021462354343384504,
            0.12765726447105408,
            ......
            0.04326574504375458,
            -0.19384358823299408
          ],
          [
            0.21094878017902374,
            0.03325708955526352,
            ......
            -0.05052570998668671,
            0.019874464720487595
          ],
          [
            0.15182611346244812,
            0.0733555257320404,
            ......
            -0.14868038892745972,
            0.045878272503614426
          ],
          [
            0.1782858967781067,
            0.08701540529727936,
            ......
            -0.08826161175966263,
            0.03097386099398136
          ],
          [
            0.15832433104515076,
            0.08862681686878204,
            ......
            -0.08204027265310287,
            -0.023515628650784492
          ],
          [
            0.006651569157838821,
            0.1267307847738266,
            ......
            0.04425329715013504,
            -0.194008007645607
          ]
        ]    
      }
    ],
    "token_usage": {
      "doubao-embedding-vision__251215": {
        "prompt_tokens": 2065,
        "completion_tokens": 0,
        "image_tokens": 0,
        "total_tokens": 2065
      }
    }
  }
}  
```

<span id="0a973e55"></span>
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
    req_path = "/api/vikingdb/embedding"
    req_body = {
        "dense_model": {
            "name": "doubao-embedding-vision",
            "version": "250615",
            "dim": 2048,
         },
        "sparse_model": {
            "name": "doubao-embedding-vision",
            "version": "250615",
         },
        "data": [
            {
                "text": "从前有座山,山里有座庙", 
            } 
        ]
    }
    result = client.do_req(req_method=req_method, req_path=req_path, req_params=req_params, req_body=req_body)
    print("req http status code: ", result.status_code)
    print("req result: \n", result.text)
```
