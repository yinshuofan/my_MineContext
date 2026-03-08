<span id="2ff51be2"></span>
# 概述
接口用于对指定离线任务详情的查看
<span id="725b98b3"></span>
# 地域功能开放说明

|**地域** |**离线任务Task处理** |
|---|---|
|华北 |支持 |
|华东 |支持 |
|华南 |支持 |
|柔佛 |暂不支持，正在开发中 |

<span id="1b42ea9c"></span>
# 接口升级说明

* 对应的V1接口为：https://www.volcengine.com/docs/84313/1544712
* 使用区别：

| |V2接口 |V1接口 |
|---|---|---|
|参数命名风格 |驼峰（TaskId） |下划线（task_id） |

<span id="7c833db5"></span>
# 请求接口
:::tip
请求向量数据库 VikingDB 的 OpenAPI 接口时，可以使用 **ak、sk** **构造签名**进行鉴权。请参见[控制面API调用流程](/docs/84313/1791144)，复制调用示例并填入必要信息
:::
<span id="c2be34ef"></span>

## 请求参数
请求参数Action取值：GetVikingdbTask
下表仅列出该接口特有的请求参数和部分公共参数。更多信息请见[公共参数](https://www.volcengine.com/docs/6369/67268)。

|参数 |类型 |是否必填 |描述 |
|---|---|---|---|
|TaskId |String |是 |任务ID |

<span id="a6340c0b"></span>
## 返回参数
下表仅列出本接口特有的返回参数。更多信息请参见[返回结构](https://www.volcengine.com/docs/6369/80336)。

|参数 |子参数 |类型 |描述 |
|---|---|---|---|
|TaskProcessInfo | |Map |任务处理信息 |
|^^| | | |
| |TaskProgress |String |任务进度。如： |
| | | |50% |
|^^| | | |
| |ErrorMessage |String |任务错误信息 |
|^^| | | |
| |SampleData |Array of Map |采样5条数据用于展示 |
|^^| | | |
| |SampleTimestamp |String |采样时间戳 |
|^^| | | |
| |ScanDataCount |Long |当前已扫描的数据量 |
|^^| | | |
| |TotalDataCount |Long |数据集总条数（预估） |
|^^| | | |
| |TotalFilterCount |Long |已经过滤出的数据量 |
| | | | |
|TaskConfig | |Map |任务配置信息 |
|^^| | | |
| |ProjectName |String |项目名称 |
|^^| | | |
| |ResourceId |String |资源ID |
|^^| | | |
| |CollectionName |String |向量库数据集名称 |
|^^| | | |
| |FileType |String |导入/导出文件的格式，支持parquet/json |
|^^| | | |
| |NeedConfirm |Boolean |是否可跳过人工确认环节，默认为true |
|^^| | | |
| |FilterConds |Array of String |过滤条件。使用参考https://www.volcengine.com/docs/84313/1419289 |
|^^| | | |
| |TosPath |String |将数据文件导入/导出到用户的TOS 路径，格式 ：{桶名}/{路径}，注意不是域名。导入/导出时必填 |
|^^| | | |
| |ExportAll |Boolean |是否导出全部数据，此时filter不生效。默认为false |
|^^| | | |
| |UpdateFields |Map |需要更新的字段值，必须是标量字段，不支持vector、sparse_vector、text 类型字段的更新 |
|^^| | | |
| |IgnoreError |Boolean |用于数据导入。设置为 true 时遇到数据会继续解析文件，默认为 false |
|^^| | | |
| |UsePublic |Boolean |使用公共tos |
| | | | |
|TaskId | |String |任务ID |
| | | | |
|UpdatePerson | |String |更新人 |
| | | | |
|TaskStatus | |String |任务状态 |
| | | | |
|TaskType | |String |任务类型 |
| | | | |
|CreateTime | |String |创建时间 |
| | | | |
|UpdateTime | |String |更新时间 |

<span id="7dbaa1d6"></span>
## 请求示例
```Plain Text
action = "GetVikingdbTask",
body = {
  "TaskId": "0ae385ed-47ec-5661-a82c-965dab9d3b99"
}
```

<span id="0b87d609"></span>
## 返回示例
```Plain Text
{
  "ResponseMetadata": {
    "RequestId": "20250609102352190158161083FBD6BB",
    "Action": "GetVikingdbTask",
    "Version": "2025-06-09",
    "Service": "vikingdb",
    "Region": "cn-beijing"
  },
  "Result": {
    "TaskId": "abcabc",
    "TaskType": "data_export",
    "TaskStatus": "running",
    "UpdatePerson": "Bob",
    "CreateTime": "2025-05-13T03:46:52Z",
    "UpdateTime": "2025-05-13T04:47:52Z",
    "TaskConfig": {
      "ProjectName": "default",
      "ResourceId": "vdb-aaaa",
      "CollectionName": "test_coll",
      "FileType": "json",
      "TosPath": "aaa/b.json",
      "ExportAll": true,
      "IgnoreError": true
    }
  }
}
```

<span id="99123ca2"></span>
## 错误码
错误码请参见[公共错误码](https://www.volcengine.com/docs/6369/68677)文档。