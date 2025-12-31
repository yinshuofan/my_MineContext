<span id="f866c2a3"></span>
# 概述
接口用于对指定的数据集详情的查询
<span id="8498f0db"></span>
# 接口升级说明

* 对应的V1接口为：https://www.volcengine.com/docs/84313/1254530
* 使用区别：


| | | | \
| |V2接口 |V1接口 |
|---|---|---|
| | | | \
|参数命名风格 |驼峰（`CollectionName`） |下划线（`collection_name`） |

<span id="74d18a99"></span>
# 请求接口
:::tip
请求向量数据库 VikingDB 的 OpenAPI 接口时，可以使用 **ak、sk** **构造签名**进行鉴权。请参见[控制面API调用流程](/docs/84313/1791144)，复制调用示例并填入必要信息
:::
<span id="58894477"></span>
## 请求参数
请求参数Action取值：GetVikingdbCollection
下表仅列出该接口特有的请求参数和部分公共参数。更多信息请见[公共参数](https://www.volcengine.com/docs/6369/67268)。

| | | | | \
|参数 |类型 |是否必填 |描述 |
|---|---|---|---|
| | | | | \
|ProjectName |String |否 |项目名称 |
| | | | | \
|CollectionName |String |2选1 |数据集名称 |
| | |^^| | \
|ResourceId |String | |数据集资源ID。请求必须指定ResourceId和CollectionName其中之一。 |

<span id="50b9fd8e"></span>
## 返回参数
下表仅列出本接口特有的返回参数。更多信息请参见[返回结构](https://www.volcengine.com/docs/6369/80336)。

| | | | | | \
|参数 |一级参数 |二级参数 |类型 |描述 |
|---|---|---|---|---|
| | | | | | \
|ProjectName | | |String |项目名称 |
| | | | | | \
|ResourceId | | |String |资源ID |
| | | | | | \
|CollectionName | | |String |数据集名称 |
| | | | | | \
|Description | | |String |数据集描述 |
| | | | | | \
|Fields | | | Map |字段列表 |
| | | | | | \
| |FieldName | |String |字段名称 |
|^^| | | | | \
| |FieldType | |String |字段类型 |
|^^| | | | | \
| |Dim | |Integer |若字段类型是vector，该参数指定稠密向量的维度 |
|^^| | | | | \
| |IsPrimaryKey | |Boolean |是否为主键字段。可以为数据集指定1个主键字段（string或int64类型）。若没有指定，则使用自动生成的主键，字段名为"**AUTO_ID**"。 |
|^^| | | | | \
| |DefaultValue | |任意类型 |字段内容默认值 |
| | | | | | \
|Vectorize | | |Map |向量化配置 |
|^^| | | | | \
| |Dense | |Map |稠密向量化模型配置 |
|^^|^^| | | | \
| | |ModelName |String |模型名称 |
|^^|^^| | | | \
| | |ModelVersion |String |模型版本 |
|^^|^^| | | | \
| | |TextField |String |文本向量化字段名称 |
|^^|^^| | | | \
| | |ImageField |String |图片向量化模型 |
|^^|^^| | | | \
| | |Dim |Integer |如果需要生成稠密向量，指定向量维度。默认使用模型默认的维度。 |
|^^|^^| | | | \
| | |VideoField |String |视频向量化字段 |
|^^| | | | | \
| |Sparse | |Map |稀疏向量化配置 |
|^^|^^| | | | \
| | |ModelName |String |模型名称 |
|^^|^^| | | | \
| | |ModelVersion |String |模型版本 |
|^^|^^| | | | \
| | |TextField |String |文本向量化字段名称 |
| | | | | | \
|CreateTime | | |String |创建时间 |
| | | | | | \
|UpdateTime | | |String |更新时间 |
| | | | | | \
|UpdatePerson | | |String |更新人 |
| | | | | | \
|CollectionStats | | |Map |统计信息 |
|^^| | | | | \
| |DataCount | |Long |数据条数 |
|^^| | | | | \
| |DataStorage | |Long |数据存储量(byte) |
| | | | | | \
|EnableKeywordsSearch | | |Boolean |是否可支持关键词检索 |
| | | | | | \
|IndexNames | | |Array of String |数据集下的索引名称列表 |
| | | | | | \
|IndexCount | | |Integer |数据集下的索引个数 |

<span id="673f05d3"></span>
## 请求示例
```Plain Text
action = "GetVikingdbCollection",
body = {
  "ProjectName": "default",
  "CollectionName": "my_test_coll_1"
}
```

<span id="62eb184e"></span>
## 返回示例
```Plain Text
{
  "ResponseMetadata": {
    "RequestId": "20250331165532135216135139CD8E48",
    "Action": "GetCollection",
    "Version": "2025-06-09",
    "Service": "vikingdb",
    "Region": "cn-beijing"
  },
  "Result": {
    "ProjectName": "default",
    "CollectionName": "my_test_coll_1",
    "Fields": [
      {
        "FieldName": "f_id",
        "FieldType": "string"
      },
      {
        "FieldName": "f_int64",
        "FieldType": "int64",
        "DefaultValue": -1
      },
      {
        "FieldName": "f_string",
        "FieldType": "string",
        "DefaultValue": "DEFAULT_STRING_VALUE"
      },
      {
        "FieldName": "f_vector",
        "FieldType": "vector",
        "Dim": 1024
      }
    ],
    "PrimaryKey": "f_id",
    "Description": "my test collection",
    "CollectionAliases": [
      "current_collection"
    ],
    "CreateTime": "2023-11-21 12:22:57",
    "UpdateTime": "2023-11-22 13:15:45",
    "UpdatePersion": "xiaoming",
    "IndexNames": [
      "idx_1"
    ],
    "IndexCount": 1,
    "CollectionStats": {
      "DataCount": 1500000,
      "DataStorage": 1500000000
    }
  }
}
```

<span id="87076e54"></span>
## 错误码
公共错误码请参见[公共错误码](https://www.volcengine.com/docs/6369/68677)文档。