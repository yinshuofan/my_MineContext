<span id="e303ad2b"></span>
# 概述
接口用于对向量数据库的创建
<span id="a747edfc"></span>
# 接口升级说明

* 对应的V1接口为：https://www.volcengine.com/docs/84313/1254542
* 使用区别：

| |V2接口 |V1接口 |
|---|---|---|
| | | | \
|参数命名风格 |驼峰（`CollectionName`） |下划线（`collection_name`） |
| | | | \
|参数类型 |Vectorize 类型为map |vectorize的数据类型是[]map。 |
| | | | \
|主键参数位置 |Fields的子参数IsPrimaryKey |primary_key |

<span id="064bbbe2"></span>
# 请求接口
:::tip
请求向量数据库 VikingDB 的 OpenAPI 接口时，可以使用 **ak、sk** **构造签名**进行鉴权。请参见[控制面API调用流程](/docs/84313/1791144)，复制调用示例并填入必要信息
:::
<span id="9a97404e"></span>
## 请求参数
请求参数Action取值：CreateVikingdbCollection
下表仅列出该接口特有的请求参数和部分公共参数。更多信息请见[公共参数](https://www.volcengine.com/docs/6369/67268)。

| | | | | | | | \
|参数 |一级参数 |二级参数 |类型 |是否必填 |示例值 |描述 |
|---|---|---|---|---|---|---|
| | | | | | | | \
|ProjectName | | |String |否 |default |项目名称 |
| | | | | | | | \
|CollectionName | | |String |是 | |数据集名称 |\
| | | | | | |说明： |\
| | | | | | | |\
| | | | | | |* 只能使用英文字母、数字、下划线_，并以英文字母开头，不能为空 |\
| | | | | | |* 长度（字节）要求：[1, 128] |\
| | | | | | |* 同账号下，所有 Collection 名称不能重复 |\
| | | | | | |* 同账号下，collection数量不超过200个 |
| | | | | | | | \
|Description | | |String |否 | |数据集描述 |\
| | | | | | |说明：小于65535字节 |
| | | | | | | | \
|Fields | | |List |是 | |字段列表 |
|^^| | | | | | | \
| |FieldName | |String |是 | |字段名称 |\
| | | | | | |说明： |\
| | | | | | | |\
| | | | | | |* 只能使用英文字母、数字、下划线_，并以英文字母开头，不能为空 |\
| | | | | | |* 长度（字节）要求：[1, 128] |\
| | | | | | |* 同一个 Collection 下，Field名称不能重复 |\
| | | | | | |* 单个collection下，field数量不超过200个 |
|^^| | | | | | | \
| |FieldType | |String |是 | |字段类型，详见下文[FieldType](/docs/84313/1791154#30882a43) |
|^^| | | | | | | \
| |Dim | |Integer |否 | |若字段类型是vector，该参数指定稠密向量的维度 |
|^^| | | | | | | \
| |IsPrimaryKey | |Boolean |否 | |是否为主键字段。可以为数据集指定1个主键字段（string或int64类型）。若没有指定，则使用自动生成的主键，字段名为"__AUTO_ID__"。 |
|^^| | | | | | | \
| |DefaultValue | |任意类型 |否 | |字段内容默认值 |
| | | | | | | | \
|Vectorize | | |Map |否 | |向量化模型设置。仅当数据集需要向量化时填写。参考：[【向量库】表征方式配置参考](/docs/84313/2137609) |
|^^| | | | | | | \
| |Dense | |Map |是 | |稠密向量化模型配置 |
|^^|^^| | | | | | \
| | |ModelName |String |是 | |模型名称 |
|^^|^^| | | | | | \
| | |ModelVersion |String |否 | |模型版本 |
|^^|^^| | | | | | \
| | |TextField |String |否 | |文本向量化字段名称 |
|^^|^^| | | | | | \
| | |ImageField |String |否 | |图片向量化字段名称 |
|^^|^^| | | | | | \
| | |VideoField |String |否 | |视频向量化字段名称 |
|^^|^^| | | | | | \
| | |Dim |Integer |否 | |如果需要生成稠密向量，指定向量维度。默认使用模型默认的维度。 |
|^^|^^| | | | | | \
| | |Instruction |Map |否 | |Instruction 配置 |\
| | | | | | |通过设置 instructions 引导模型更准确地聚焦输入内容的关键信息，以提升向量表示精度，详细参考[Instruction](/docs/84313/1791154#93269fe1)： |
|^^| | | | | | | \
| |Sparse | |Map |否 | |稀疏向量化配置 |
|^^|^^| | | | | | \
| | |ModelName |String |是 | |模型名称 |
|^^|^^| | | | | | \
| | |ModelVersion |String |否 | |模型版本 |
|^^|^^| | | | | | \
| | |TextField |String |否 | |文本向量化字段名称 |
|^^|^^| | | | | | \
| | |Instruction |Map |否 | |Instruction 配置 |\
| | | | | | |通过设置 instructions 引导模型更准确地聚焦输入内容的关键信息，以提升向量表示精度，详细参考[Instruction](/docs/84313/1791154#93269fe1)： |
|^^| | | | | | | \
| |Tensor | |Map |否 | |张量化配置 |\
| | | | | | |注：当前若需要配置张量字段，需将张量化字段名、张量化模型名、张量化模型版本的配置与稠密向量化模型名、稠密向量化模型版本的配置保持一致，且模型固定为"doubao-embedding-vision"，模型版本固定为"251215" |
|^^|^^| | | | | | \
| | |ModelName |String |是 | |**模型名称** |\
| | | | | | |注：当前配置张量化字段需将**张量化模型**和**稠密向量化模型**均配置为**"doubao-embedding-vision"，** |
|^^|^^| | | | | | \
| | |ModelVersion |String |否 | |**模型版本** |\
| | | | | | |注：当前配置张量化字段需将**张量化模型版本**和**稠密向量化模型版本**均配置为**"251215"** |
|^^|^^| | | | | | \
| | |TextField |String |否 | |**文本张量化字段名称** |\
| | | | | | |注：张量化字段名称需与稠密向量化字段名称一致 |
|^^|^^| | | | | | \
| | |ImageField |String |否 | |**图片张量化字段名称** |\
| | | | | | |注：张量化字段名称需与稠密向量化字段名称一致 |
|^^|^^| | | | | | \
| | |VideoField |String |否 | |**视频张量化字段名称** |\
| | | | | | |注：张量化字段名称需与稠密向量化字段名称一致 |
|^^|^^| | | | | | \
| | |NDim |int |是 | |**张量阶数** |\
| | | | | | |目前固定张量维度阶数为2，可视作向量矩阵 |
|^^|^^| | | | | | \
| | |Shape |list |是 | |**张量维度** |\
| | | | | | |格式为[M,N]： |\
| | | | | | |M：向量矩阵的行数，取值[2,64]间的整数 |\
| | | | | | |N：向量矩阵的列数，取值[4,2048]间的整数，且当N>4时，N必须是8的倍数 |
|^^|^^| | | | | | \
| | |Instruction |Map |否 | |Instruction 配置 |\
| | | | | | |通过设置 instructions 引导模型更准确地聚焦输入内容的关键信息，以提升向量表示精度，详细参考[Instruction](/docs/84313/1791154#93269fe1)： |
| | | | | | | | \
|FullText | | |List<FullTextConfig> |否 | |FullTextConfig结构见下。 |\
| | | | | | |可配置**最多2个**text字段为全文检索字段。 |

`FullTextConfig` 结构：

| | | | | \
|参数名 |必选 |类型 |备注 |
|---|---|---|---|
| | | | | \
|Field |是 |String |必须是在Fields里存在的一个text类型的字段 |
| | | | | \
|Analyzer |是 |AnalyzerConfig |AnalyzerConfig结构见下 |

`AnalyzerConfig` 结构：

| | | | | \
|参数名 |必选 |类型 |备注 |
|---|---|---|---|
| | | | | \
|Tokenizer |\
| |是 |String |\
| | | |分词器。可选值： |\
| | | | |\
| | | |* `standard`:支持中英文及混合内容，性能较高，满足大部分需求。 |\
| | | |* `zh`：针对中文分词，效果更好，性能稍低。 |\
| | | |* `whitespace`：按空白字符分割（空格、`\t`、`\n`、`\r`、`\v`、`\f`） |

<span id="30882a43"></span>
## FieldType

| | | | | \
|字段类型 |格式 |可为主键 |说明 |
|---|---|---|---|
| | | | | \
|int64 |整型数值 |是 |**整数** |\
| | | |注：当作为主键时，需满足该字段取值不为0 |
| | | | | \
|float32 |浮点数值 |否 |**浮点数** |
| | | | | \
|string |字符串 |是 |**字符串** |\
| | | |推荐使用方式：当用于枚举值过滤时，推荐长度不超过128字节 |\
| | | |硬性限制：要求不超过65535字节；当作为主键时，需满足长度不超过256字节 |
| | | | | \
|bool |true/false |否 |**布尔类型** |
| | | | | \
|list<string> |字符串数组 |否 |**字符串数组** |\
| | | |推荐使用方式：当用于枚举值过滤时，列表长度不超过32个，单元素不超过1024字节 |\
| | | |硬性限制：列表长度不超过20000，单元素不超过1024字节，总大小不超过1MB |
| | | | | \
|list<int64> |整型数组 |否 |**整数数组** |\
| | | |硬性限制：列表长度不超过1024 |
| | | | | \
|vector |* 向量（浮点数数组） |\
| |* float32/float64压缩为bytes后的base64编码 |否 |**稠密向量** |\
| | | |硬性限制：长度不小于4，不大于4096，且为4的整数倍数。  |
| | | | | \
|sparse_vector |\
| | 输入格式<token_id ,token_weight>的字典列表，来表征稀疏稀疏向量的非零位下标及其对应的值, 其中 token_id 是 string 类型, token_weight 是float 类型 |否 |**稀疏向量** |\
| | | | |
| | | | | \
|tensor |* 张量：输入格式为 M*N 的二维数组（向量矩阵），以shape为4*8的二维数组为例 |\
| |* ```Python |\
| |tensor_tuple = [ |\
| |    [0.12, 0.34, -0.21, 0.56, 0.78, -0.45, 0.89, -0.10],  |\
| |    [0.09, 0.41, -0.18, 0.62, 0.71, -0.39, 0.82, -0.15],  |\
| |    [0.15, 0.37, -0.24, 0.59, 0.75, -0.42, 0.86, -0.08],  |\
| |    [0.11, 0.39, -0.20, 0.57, 0.77, -0.44, 0.88, -0.12]  |\
| |] |\
| |``` |\
| | |否 |**张量 tensor** |
| | | | | \
|text |字符串 |否 |**文本** |\
| | | |若为向量化字段，则值不能为空。（若否，可以为空） |\
| | | |硬性限制：需小于65535字节 |
| | | | | \
|image |字符串 |否 |**图片** |\
| | | |若为向量化字段，则值不能为空。（若否，可以为空） |\
| | | | |\
| | | |* 图片tos链接 `tos://{bucket}/{object}` |\
| | | |* http/https格式链接 |\
| | | |* 对应图片需满足宽高长度大于14px，像素小于3600万 |\
| | | |* 可参考：[多模态向量化](https://www.volcengine.com/docs/82379/1409291?lang=zh#a256838b) |
| | | | | \
|video |map |否 |**视频** |\
| | | |若为向量化字段，则值不能为空。（若否，可以为空） |\
| | | | |\
| | | |* 视频tos链接 `tos://{bucket}/{object}` |\
| | | |* http/https格式url链接 |\
| | | |* 对应视频需满足大小不超过50mb |\
| | | |* 可参考：[多模态向量化](https://www.volcengine.com/docs/82379/1409291?lang=zh#a256838b) |\
| | | | |\
| | | |```Plain Text |\
| | | |{ |\
| | | |    "value": tos://{bucket}/{object} # 或http/https格式url链接，该字段必填 |\
| | | |    "fps": 0.2 # 取值0.2-5，选填 |\
| | | |} |\
| | | |``` |\
| | | | |\
| | | | |
| | | | | \
|date_time |string |否 |分钟级别： |\
| | | |`yyyy-MM-ddTHH:mmZ`或`yyyy-MM-ddTHH:mm±HH:mm` |\
| | | |秒级别： |\
| | | |`yyyy-MM-ddTHH:mm:ssZ`或`yyyy-MM-ddTHH:mm:ss±HH:mm` |\
| | | |毫秒级别： |\
| | | |`yyyy-MM-ddTHH:mm:ss.SSSZ`或`yyyy-MM-ddTHH:mm:ss.SSS±HH:mm` |\
| | | |例如："2025-08-12T11:33:56+08:00" |\
| | | |注：取值需满足在 1970-01-01T00:00:00Z 至 3000-01-01T00:00:00Z 之间 |
| | | | | \
|geo_point |string |否 |地理坐标`longitude,latitude`，其中`longitude`取值(-180,180)，`latitude`取值(-90,90) |\
| | | |例如："116.408108,39.915023" |

<span id="93269fe1"></span>
## Instruction

| | | | | | | \
|参数 |一级参数 |类型 |是否必填 |默认值 |描述 |
|---|---|---|---|---|---|
| | | | | | | \
|Instruction | |Map |否 | |**Instruction 配置** |\
| | | | | |通过设置 instructions 引导模型更准确地聚焦输入内容的关键信息，以提升向量表示精度 |\
| | | | | |若不填该字段则等价于 AutoFill=true |
|^^| | | | | | \
| |AutoFill |Boolean |是 |true |**自动填充 Instruction 内容** |\
| | | | | |写入数据时，会自动根据模态信息填充 Instruction 内容。如果模型不支持Instruction，填写后不报错但不生效。各模型的填充规则见下表[模型列表](/docs/84313/1791154#863656ff)。 |


<span id="6f591cab"></span>
## 返回参数
下表仅列出本接口特有的返回参数。更多信息请参见[返回结构](https://www.volcengine.com/docs/6369/80336)。

| | | | | \
|参数 |类型 |示例值 |描述 |
|---|---|---|---|
| | | | | \
|ResourceId |String | |资源ID |
| | | | | \
|Message |String |success |操作结果信息 |

<span id="863656ff"></span>
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


<span id="7e981c1a"></span>
## 请求示例
<span id="468b298c"></span>
### 创建带vector字段的数据集
```Python
action="CreateVikingdbCollection",
body = {
  "ProjectName": "default",
  "CollectionName": "my_test_coll_1",
  "Description": "this is my test collection",
  "Fields": [
    {
      "FieldName": "f_id",
      "FieldType": "string",
      "IsPrimaryKey": true,
    },
    {
      "FieldName": "f_int64",
      "FieldType": "int64",
      "DefaultValue": -1
    },
    {
      "FieldName": "f_vector",
      "FieldType": "vector",
      "Dim": 1024
    },
    {
        "FieldName": "tensor_text_vector",
        "FieldType": "tensor",
        "NDim": 2,# 目前固定为 2
        "Shape": [32, 2048]，
    }
]
}
```

<span id="6e74850d"></span>
### 创建带vectorize(向量化)的数据集
```Python
action = "CreateVikingdbCollection"
body = {
    "ProjectName": "default",
    "CollectionName": "my_test_coll_2",
    "Description": "",
    "Fields": [
        {"FieldName": "f_id", "FieldType": "int64", "IsPrimaryKey": True},
        {"FieldName": "f_text", "FieldType": "text", },
        {"FieldName": "f_image", "FieldType": "image", },
        {"FieldName": "f_video", "FieldType": "video", },
    ],
    "Vectorize": {
        "Dense": {
            "ModelName": "doubao-embedding-vision",
            "ModelVersion": "251215",
            "TextField": "f_text",
            "ImageField": "f_image",
            "Dim": 1024,
            "Instruction": {
                "AutoFill": true
            }
        },
        "Sparse": {
            "ModelName": "doubao-embedding-vision",
            "ModelVersion": "251215",
            "TextField": "f_text"
        },
        "Tensor": {
            "ModelName": "doubao-embedding-vision",
            "ModelVersion": "251215",
            "NDim": 2,
            "Shape": [64, 2048],
            "TextField": "f_text",
            "ImageField": "f_image",
        }
    }
}
```


<span id="5b0aa9ad"></span>
## 返回示例
```Plain Text
{
  "ResponseMetadata": {
    "RequestId": "20250331173619111231152071D44F42",
    "Action": "CreateCollection",
    "Version": "2025-06-09",
    "Service": "vikingdb",
    "Region": "cn-beijing"
  },
  "Result": {
    "Message": "success"
  }
}
```

<span id="8069d269"></span>
## 错误码
公共错误码请参见[公共错误码](https://www.volcengine.com/docs/6369/68677)文档。
