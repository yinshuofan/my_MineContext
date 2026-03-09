<span id="a3e36902"></span>
# 概述
接口用于对指定离线任务列表的查看
<span id="e2329b6d"></span>
# 地域功能开放说明

|**地域** |**离线任务Task处理** |
|---|---|
|华北 |支持 |
|华东 |支持 |
|华南 |支持 |
|柔佛 |暂不支持，正在开发中 |

<span id="ad54114a"></span>
# 接口升级说明

* 对应的V1接口为：https://www.volcengine.com/docs/84313/1544713
* 使用区别：

| |V2接口 |V1接口 |
|---|---|---|
|参数命名风格 |驼峰（TaskStatus） |下划线（task_status） |


<span id="824013cb"></span>
# 请求接口
:::tip

:::
<span id="07893b5b"></span>
## 请求参数
请求参数Action取值：ListVikingdbTask
下表仅列出该接口特有的请求参数和部分公共参数。更多信息请见[公共参数](https://www.volcengine.com/docs/6369/67268)。

|参数 |类型 |是否必填 |描述 |
|---|---|---|---|
| | | | | \
|ProjectName |String |否 |项目名称 |
| | | | | \
|CollectionName |String |2选1 |数据集名称 |
| | |^^| | \
|ResourceId |String | |数据集资源ID。请求必须指定ResourceId和CollectionName其中之一。 |
| | | | | \
|TaskStatus |String |是 |任务状态 |
| | | | | \
|TaskType |String |是 |任务类型 |
| | | | | \
|PageNumber |Integer |否 |翻页页码。起始为1。 |
| | | | | \
|PageSize |Integer |否 |翻页每页的大小。 |

<span id="d434ea93"></span>
## 返回参数
下表仅列出本接口特有的返回参数。更多信息请参见[返回结构](https://www.volcengine.com/docs/6369/80336)。

|参数 |一级子参数 |二级子参数 |类型 |描述 |
|---|---|---|---|---|
| | | | | | \
|Tasks | | |Array of Map |任务信息列表 |
|^^| | | | | \
| |TaskId | |String |任务ID |
|^^| | | | | \
| |TaskType | |String |任务类型 |
|^^| | | | | \
| |TaskStatus | | | |
|^^| | | | | \
| |UpdatePerson | |String |更新人 |
|^^| | | | | \
| |UpdateTime | |String |更新时间 |
|^^| | | | | \
| |CreateTime | |String |创建时间 |
|^^| | | | | \
| |TaskProcessInfo | |Map |任务处理信息 |
|^^|^^| | | | \
| | |TaskProgress |String |任务进度。如50% |
|^^|^^| | | | \
| | |ErrorMessage |String |任务错误信息 |
|^^|^^| | | | \
| | |SampleData |Array of Map |采样5条数据用于展示 |
|^^|^^| | | | \
| | |SampleTimestamp |String |采样时间戳 |
|^^|^^| | | | \
| | |ScanDataCount |Long |当前已扫描的数据量 |
|^^|^^| | | | \
| | |TotalDataCount |Long |数据集总条数（预估） |
|^^|^^| | | | \
| | |TotalFilterCount |Long |已经过滤出的数据量 |
|^^| | | | | \
| |TaskConfig | |Map |任务配置 |
|^^|^^| | | | \
| | |ProjectName |String |项目名称 |
|^^|^^| | | | \
| | |ResourceId |String |资源ID |
|^^|^^| | | | \
| | |CollectionName |String |向量库数据集名称 |
|^^|^^| | | | \
| | |FileType |String |导入/导出文件的格式，支持parquet/json |
|^^|^^| | | | \
| | |NeedConfirm |Boolean |是否可跳过人工确认环节，默认为true |
|^^|^^| | | | \
| | |FilterConds |Array of map |过滤条件。使用参考https://www.volcengine.com/docs/84313/1419289 |
|^^|^^| | | | \
| | |TosPath |String |将数据文件导入/导出到用户的TOS 路径，格式 ：{桶名}/{路径}，注意不是域名。导入/导出时必填 |
|^^|^^| | | | \
| | |ExportAll |Boolean |是否导出全部数据，此时filter不生效。默认为false |
|^^|^^| | | | \
| | |UpdateFields |Map |需要更新的字段值，必须是标量字段，不支持vector、sparse_vector、text 类型字段的更新 |
|^^|^^| | | | \
| | |IgnoreError |Boolean |用于数据导入。设置为 true 时遇到数据会继续解析文件，默认为 false |
|^^|^^| | | | \
| | |UsePublic |Boolean |使用公共tos |
| | | | | | \
|TotalCount | | |Integer |任务总数 |
| | | | | | \
|PageSize | | |Integer |请求时的PageSize值，如果请求时未指定，则为默认值。 |
| | | | | | \
|PageNumber | | |Integer |请求时的PageNumber值，如果请求时未指定，则为默认值。 |

<span id="d1df0993"></span>
## 请求示例
```Plain Text
action = "ListVikingdbTask",
body = {
  "TaskStatus": "success",
  "TaskType": "data_import",
  "PageNumber": 1,
  "PageSize": 10,
  "CollectionName": "test_coll",
}
```

<span id="d539c752"></span>
## 返回示例
```Plain Text
{
  "ResponseMetadata": {
    "RequestId": "20250609102203036054054146B36A88",
    "Action": "ListVikingdbTask",
    "Version": "2025-06-09",
    "Service": "vikingdb",
    "Region": "cn-beijing"
  },
  "Result": {
    "Tasks": [
      {
        "TaskId": "mLVZS",
        "TaskType": "data_import",
        "TaskStatus": "success",
        "UpdatePerson": "Bob",
        "UpdateTime": "2025-05-13T03:48:52Z",
        "CreateTime": "2025-05-13T03:46:52Z",
        "TaskProcessInfo": {
          "TaskProgress": "100%",
          "ErrorMessage": "z",
          "ScanDataCount": 220,
          "TotalDataCount": 220
        },
        "TaskConfig": {
          "ProjectName": "default",
          "ResourceId": "lTrxBgA",
          "CollectionName": "test_coll",
          "FileType": "json",
          "NeedConfirm": true,
          "TosPath": "aaaa/b.json",
          "ExportAll": false,
          "IgnoreError": false
        }
      }
    ],
    "TotalCount": 1,
    "PageSize": 1,
    "PageNumber": 10
  }
}
```

<span id="b1e83c93"></span>
## 错误码
错误码请参见[公共错误码](https://www.volcengine.com/docs/6369/68677)文档。