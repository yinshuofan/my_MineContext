<span id="edc17365"></span>
# 概述
接口用于对指定离线任务的删除
<span id="74ba0cb3"></span>
# 地域功能开放说明

|**地域** |**离线任务Task处理** |
|---|---|
|华北 |支持 |
|华东 |支持 |
|华南 |支持 |
|柔佛 |暂不支持，正在开发中 |

<span id="112a2c30"></span>
# 接口升级说明

* 对应的V1接口为：https://www.volcengine.com/docs/84313/1392480
* 使用区别：

| |V2接口 |V1接口 |
|---|---|---|
|参数命名风格 |驼峰（TaskId） |下划线（task_id） |

<span id="b610d803"></span>
# 请求接口
:::tip
请求向量数据库 VikingDB 的 OpenAPI 接口时，可以使用 **ak、sk** **构造签名**进行鉴权。请参见[控制面API调用流程](/docs/84313/1791144)，复制调用示例并填入必要信息
:::
<span id="1e6f1486"></span>
## 请求参数
请求参数Action取值：DeleteVikingdbTask
下表仅列出该接口特有的请求参数和部分公共参数。更多信息请见[公共参数](https://www.volcengine.com/docs/6369/67268)。

|参数 |类型 |是否必填 |描述 |
|---|---|---|---|
|TaskId |String |是 |任务ID |

<span id="ab5ef9c5"></span>
## 返回参数
下表仅列出本接口特有的返回参数。更多信息请参见[返回结构](https://www.volcengine.com/docs/6369/80336)。

|参数 |类型 |示例值 |描述 |
|---|---|---|---|
|Message |String |success |操作结果信息 |

<span id="1a8ce0c1"></span>
## 请求示例
```Plain Text
action = "DeleteVikingdbTask",
body = {
  "TaskId": "7d8e122b-7870-5ff4-9be4-86a099cb7f64"
}
```

<span id="1e1708d6"></span>
## 返回示例
```Plain Text
{
  "ResponseMetadata": {
    "RequestId": "2025060910180919118203304068EA18",
    "Action": "DeleteVikingdbTask",
    "Version": "2025-06-09",
    "Service": "vikingdb",
    "Region": "cn-beijing"
  },
  "Result": {
    "Message": "success"
  }
}
```

<span id="b4ad3e9c"></span>
## 错误码
公共错误码请参见[公共错误码](https://www.volcengine.com/docs/6369/68677)文档。