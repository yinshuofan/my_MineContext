<span id="f6304bb3"></span>
# 概述
接口用于对指定索引的删除
<span id="5a2c62df"></span>
# 接口升级说明

* 对应的V1接口为：https://www.volcengine.com/docs/84313/1254540
* 使用区别：


| | | | \
| |V2接口 |V1接口 |
|---|---|---|
| | | | \
|参数命名风格 |驼峰（`IndexName`） |下划线（`index_name`） |


* 注：QPS限流是以数据条数计算，V1与V2接口的限流行为完全相同。

<span id="78e2ee8c"></span>
# 请求接口
:::tip
请求向量数据库 VikingDB 的 OpenAPI 接口时，可以使用 **ak、sk** **构造签名**进行鉴权。请参见[控制面API调用流程](/docs/84313/1791144)，复制调用示例并填入必要信息
:::
<span id="3f577f5e"></span>
## 请求参数
请求参数Action取值：DeleteVikingdbIndex。
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
| | | | | \
|IndexName |String |是 |索引名称 |

<span id="b66b619c"></span>
## 返回参数
下表仅列出本接口特有的返回参数。更多信息请参见[返回结构](https://www.volcengine.com/docs/6369/80336)。

| | | | | \
|参数 |类型 |示例值 |描述 |
|---|---|---|---|
| | | | | \
|Message |String |success |操作结果信息 |

<span id="7e874be9"></span>
## 请求示例
```Plain Text
action = "DeleteVikingdbIndex"
body = {
  "CollectionName": "test_coll",
  "IndexName": "test_coll_index"
}
```

<span id="aa67307d"></span>
## 返回示例
```Plain Text
{
  "ResponseMetadata": {
    "RequestId": "20250527175218115212119089A5D56A",
    "Action": "DeleteVikingdbIndex",
    "Version": "2025-06-09",
    "Service": "vikingdb",
    "Region": "cn-beijing"
  },
  "Result": {
    "Message": "success"
  }
}
```