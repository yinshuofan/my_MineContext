<span id="2a2d6ef5"></span>
# 概述
接口用于对指定的数据集的删除
<span id="b936f63a"></span>
# 接口升级说明

* 对应的V1接口为：https://www.volcengine.com/docs/84313/1254538
* 使用区别：


| | | | \
| |V2接口 |V1接口 |
|---|---|---|
| | | | \
|参数命名风格 |驼峰（`CollectionName`） |下划线（`collection_name`） |

<span id="ab26a756"></span>
# 请求接口
:::tip
请求向量数据库 VikingDB 的 OpenAPI 接口时，可以使用 **ak、sk** **构造签名**进行鉴权。请参见[控制面API调用流程](/docs/84313/1791144)，复制调用示例并填入必要信息
:::
<span id="547159a2"></span>
## 请求参数
请求参数Action取值：DeleteVikingdbCollection
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

<span id="9eacd241"></span>
## 返回参数
下表仅列出本接口特有的返回参数。更多信息请参见[返回结构](https://www.volcengine.com/docs/6369/80336)。

| | | | | \
|参数 |类型 |示例值 |描述 |
|---|---|---|---|
| | | | | \
|Message |String |success |操作结果信息 |

<span id="ca0530a4"></span>
## 请求示例
```Plain Text
action = "DeleteVikingdbCollection",
body = {
    "ProjectName": "default",
    "CollectionName": "my_test_coll_1",
}
```

<span id="308482aa"></span>
## 返回示例
```Plain Text
{
  "ResponseMetadata": {
    "RequestId": "20250331204634101189204179056F1E",
    "Action": "DeleteVikingdbCollection",
    "Version": "2025-06-09",
    "Service": "vikingdb",
    "Region": "cn-beijing"
  },
  "Result": {
    "Message": "success"
  }
}
```

<span id="2211a3c8"></span>
## 错误码
公共错误码请参见[公共错误码](https://www.volcengine.com/docs/6369/68677)文档。