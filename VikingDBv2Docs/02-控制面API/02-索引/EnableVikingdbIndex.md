<span id="ea91b283"></span>
# 概述
接口用于对指定"已停用"的索引的启用，启用后索引将重新构建并占用一定资源，构建完成后可正常使用，停用索引参考：[停用索引-DisableVikingdbIndex](/docs/84313/697b1059f4ecc50551659926)。
<span id="f67bc3cc"></span>
# 请求接口
:::tip
请求向量数据库 VikingDB 的 OpenAPI 接口时，可以使用 **ak、sk** **构造签名**进行鉴权。请参见[控制面API调用流程](/docs/84313/1791144)，复制调用示例并填入必要信息
:::
<span id="ad8a44d6"></span>
## 请求参数
请求参数Action取值：EnableVikingdbIndex。
下表仅列出该接口特有的请求参数和部分公共参数。更多信息请见[公共参数](https://www.volcengine.com/docs/6369/67268)。

| | | | | | \
|参数 | |类型 |是否必填 |描述 |
|---|---|---|---|---|
| | | | | | \
|ProjectName | |String |否 |项目名称 |
| | | | | | \
|CollectionName | |String |2选1 |数据集名称 |
| | | |^^| | \
|ResourceId | |String | |数据集资源ID。请求必须指定ResourceId和CollectionName其中之一。 |
| | | | | | \
|IndexName | |String |是 |索引名称 |

<span id="4ee86acf"></span>
## 返回参数
下表仅列出本接口特有的返回参数。更多信息请参见[返回结构](https://www.volcengine.com/docs/6369/80336)。

|参数 |类型 |示例值 |描述 |
|---|---|---|---|
|Message |String |success |操作结果信息 |

<span id="4569b13f"></span>
## 请求示例
```Plain Text
action = "EnableVikingdbIndex"
body = {
  "CollectionName": "aaa"
  "IndexName": "5J"
}
```

<span id="1fd2c4e8"></span>
## 返回示例
```Plain Text
{
  "ResponseMetadata": {
    "RequestId": "202512261424270210401601790C2F23",
    "Action": "EnableVikingdbIndex",
    "Version": "2025-06-09",
    "Service": "vikingdb_stg",
    "Region": "cn-beijing"
  },
  "Result": {
    "Message": "success"
  }
}
```

<span id="d61db379"></span>
## 错误码
错误码请参见[公共错误码](https://www.volcengine.com/docs/6369/68677)文档。