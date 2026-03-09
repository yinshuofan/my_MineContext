<span id="bd7a18fd"></span>
# 概述
接口用于对指定"已就绪"的索引的停用，停用后索引将会释放相关内存资源且无法继续使用，如需继续使用则需启用该索引，启用索引参考[启用索引-EnableVikingdbIndex](/docs/84313/2201634)。
<span id="734acf83"></span>
# 请求接口
:::tip
请求向量数据库 VikingDB 的 OpenAPI 接口时，可以使用 **ak、sk** **构造签名**进行鉴权。请参见[控制面API调用流程](/docs/84313/1791144)，复制调用示例并填入必要信息
:::
<span id="eb5defc3"></span>
## 请求参数
请求参数Action取值：DisableVikingdbIndex。
下表仅列出该接口特有的请求参数和部分公共参数。更多信息请见[公共参数](https://www.volcengine.com/docs/6369/67268)。

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

<span id="f3e793c8"></span>
## 返回参数
下表仅列出本接口特有的返回参数。更多信息请参见[返回结构](https://www.volcengine.com/docs/6369/80336)。

| | | | | \
|参数 |类型 |示例值 |描述 |
|---|---|---|---|
| | | | | \
|Message |String |success |操作结果信息 |

<span id="c3098d56"></span>
## 请求示例
```Plain Text
action = "DisableVikingdbIndex"
body = {
  "CollectionName": "aa",
  "IndexName": "Zfg0D"
}
```

<span id="6c7a1f65"></span>
## 返回示例
```Plain Text
{
  "ResponseMetadata": {
    "RequestId": "20251226143245108247224207F9BA68",
    "Action": "DisableVikingdbIndex",
    "Version": "2025-06-09",
    "Service": "vikingdb_stg",
    "Region": "cn-beijing"
  },
  "Result": {
    "Message": "success"
  }
}
```

<span id="6a2cd0a4"></span>
## 错误码
错误码请参见[公共错误码](https://www.volcengine.com/docs/6369/68677)文档。
