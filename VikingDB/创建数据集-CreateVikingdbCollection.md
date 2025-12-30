<span id="e303ad2b"></span>
# 概述
接口用于对向量数据库的创建
<span id="a747edfc"></span>
# 接口升级说明

* 对应的V1接口为：https://www.volcengine.com/docs/84313/1254542
* 使用区别：


| | | | \
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
|CollectionName | | |String |是 | |数据集名称 |
| | | | | | | | \
|Description | | |String |否 | |数据集描述 |
| | | | | | | | \
|Fields | | |List |是 | |字段列表 |
|^^| | | | | | | \
| |FieldName | |String |是 | |字段名称 |
|^^| | | | | | | \
| |FieldType | |String |是 | |字段类型，详见下文 |
|^^| | | | | | | \
| |Dim | |Integer |否 | |若字段类型是vector，该参数指定稠密向量的维度 |
|^^| | | | | | | \
| |IsPrimaryKey | |Boolean |否 | |是否为主键字段。可以为数据集指定1个主键字段（string或int64类型）。若没有指定，则使用自动生成的主键，字段名为"**AUTO_ID**"。 |
|^^| | | | | | | \
| |DefaultValue | |任意类型 |否 | |字段内容默认值 |
| | | | | | | | \
|Vectorize | | |Map |否 | |向量化模型设置。仅当数据集需要向量化时填写。 |
|^^| | | | | | | \
| |Dense | |Map |是 | |稠密向量化模型配置 |
|^^|^^| | | | | | \
| | |ModelName |String |是 | |模型名称 |
|^^|^^| | | | | | \
| | |ModelVersion |String |否 | |模型版本 |
|^^|^^| | | | | | \
| | |TextField |String |否 | |文本向量化字段名称 |
|^^|^^| | | | | | \
| | |ImageField |String |否 | |图片向量化模型 |
|^^|^^| | | | | | \
| | |Dim |Integer |否 | |如果需要生成稠密向量，指定向量维度。默认使用模型默认的维度。 |
|^^|^^| | | | | | \
| | |VideoField |String |否 | |视频向量化字段 |
|^^| | | | | | | \
| |Sparse | |Map |否 | |稀疏向量化配置 |
|^^|^^| | | | | | \
| | |ModelName |String |是 | |模型名称 |
|^^|^^| | | | | | \
| | |ModelVersion |String |否 | |模型版本 |
|^^|^^| | | | | | \
| | |TextField |String |否 | |文本向量化字段名称 |

<span id="30882a43"></span>
# field_type 

| | | | | \
|字段类型 |格式 |可为主键 |说明 |
|---|---|---|---|
| | | | | \
|int64 |整型数值 |是 |整数 |
| | | | | \
|float32 |浮点数值 |否 |浮点数 |
| | | | | \
|string |字符串 |是 |字符串。内容限制256byte |
| | | | | \
|bool |true/false |否 |布尔类型 |
| | | | | \
|list<string> |字符串数组 |否 |字符串数组 |
| | | | | \
|list<int64> |整型数组 |否 |整数数组 |
| | | | | \
|vector |* 向量（浮点数数组） |\
| |* float32/float64压缩为bytes后的base64编码 |否 |稠密向量 |\
| | | | |
| | | | | \
|sparse_vector |\
| | 输入格式<token_id ,token_weight>的字典列表，来表征稀疏稀疏向量的非零位下标及其对应的值, 其中 token_id 是 string 类型, token_weight 是float 类型 |否 |稀疏向量 |\
| | | | |
| | | | | \
|text |字符串 |否 |若为向量化字段，则值不能为空。（若否，可以为空） |
| | | | | \
|image |字符串 |否 |若为向量化字段，则值不能为空。（若否，可以为空） |\
| | | | |\
| | | |* 图片tos链接 `tos://{bucket}/{object}` |\
| | | |* http/https格式链接 |
| | | | | \
|video |map |否 |{ |\
| | | |"value": `tos://{bucket}/{object}`，http/https格式url链接，该字段必填 |\
| | | |"fps": 0.2 （取值0.2-5，选填） |\
| | | |} |
| | | | | \
|date_time |string |否 |分钟级别： |\
| | | |`yyyy-MM-ddTHH:mmZ`或`yyyy-MM-ddTHH:mm±HH:mm` |\
| | | |秒级别： |\
| | | |`yyyy-MM-ddTHH:mm:ssZ`或`yyyy-MM-ddTHH:mm:ss±HH:mm` |\
| | | |毫秒级别： |\
| | | |`yyyy-MM-ddTHH:mm:ss.SSSZ`或`yyyy-MM-ddTHH:mm:ss.SSS±HH:mm` |
| | | | | \
|geo_point |string |否 |地理坐标`longitude,latitude`，其中`longitude`取值(-180,180)，`latitude`取值(-90,90)，均为float32类型。 |


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

| | | | | | | | | \
|模型名称 |模型版本 |支持向量化类型 |默认稠密向量维度 |可选稠密向量维度 |文本截断长度 |支持稀疏向量 | 可支持instruction |\
| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | | \
|bge-large-zh |(default) |text |1024 |1024 |512 |否 |是 |
| | | | | | | | | \
|bge-m3 |(default) |text |1024 |1024 |8192 |是 |否 |
| | | | | | | | | \
|bge-visualized-m3 |\
| |(default) |text、image及其组合 |1024 |1024 |8192 |否 |否 |
| | | | | | | | | \
|doubao-embedding |*240715* |text |2048 |\
| | | | |512, 1024, 2048 |4096 |否 |是 |\
| | | | | | | | |
| | | | | | | | | \
|doubao-embedding-large |*240915* |text |2048 |512, 1024, 2048, 4096 |4096 |否 |是 |
| | | | | | | | | \
|doubao-embedding-vision |*250328* |text、image及其组合 |2048 |2048, 1024 |\
| | | | | |8192 |否 |是 |
| | | | | | | | | \
|doubao-embedding-vision |*250615* |兼容*241215*和*250328*的用法*。​*另外，支持full_modal_seq（文/图/视频序列） |2048 |\
| | | | |2048, 1024 |\
| | | | | |128k |否 |是 |


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
            "ModelVersion": "250615",
            "TextField": "f_text",
            "ImageField": "f_image",
            "Dimension": 1024,
        },
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

