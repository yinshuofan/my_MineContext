<span id="4ddcee3f"></span>
# 概述
接口用于对向量库离线任务的创建
<span id="1e28edbc"></span>
# 地域功能开放说明

| | | \
|**地域** |**离线任务Task处理** |
|---|---|
|华北 |支持 |
|华东 |支持 |
|华南 |支持 |
|柔佛 |暂不支持，正在开发中 |

<span id="a2c8705d"></span>
# 接口升级说明

* 对应的V1接口为：https://www.volcengine.com/docs/84313/1254531
* 使用区别：

| |V2接口 |V1接口 |
|---|---|---|
|参数命名风格 |驼峰（TaskType） |下划线（task_type） |
|参数命名修改 |TaskConfig |task_params |

<span id="b9c80644"></span>

# 请求接口
:::tip
请求向量数据库 VikingDB 的 OpenAPI 接口时，可以使用 **ak、sk** **构造签名**进行鉴权。请参见[控制面API调用流程](/docs/84313/1791144)，复制调用示例并填入必要信息
:::
<span id="386fed09"></span>
## 请求参数
请求参数Action取值：CreateVikingdbTask
下表仅列出该接口特有的请求参数和部分公共参数。更多信息请见[公共参数](https://www.volcengine.com/docs/6369/67268)。

|参数 |子参数 |类型 |是否必填 |描述 |
|---|---|---|---|---|
| | | | | | \
|ProjectName | |String |否 |项目名称 |
| | | | | | \
|CollectionName | |String |2选1 |数据集名称 |
| | | |^^| | \
|ResourceId | |String | |数据集资源ID。请求必须指定ResourceId和CollectionName其中之一。 |
| | | | | | \
|TaskType | |String |是 |任务类型，不同类型的TaskType，有不同的TaskConfig，详见**不同TaskType的TaskConfig配置** |
| | | | | | \
|TaskConfig | |Map |是 |任务具体配置 |
|^^| | | | | \
| |FileType |String |否 |导入/导出文件的格式，支持parquet/json |
|^^| | | | | \
| |NeedConfirm |Boolean |否 |是否可跳过人工确认环节，默认为true |
|^^| | | | | \
| |FilterConds |Map |否 |过滤条件。使用参考https://www.volcengine.com/docs/84313/1419289 |
|^^| | | | | \
| |TosPath |String |否 |将数据文件导入/导出到用户的TOS 路径，格式 ：{桶名}/{路径}，注意不是域名。导入/导出时必填 |
|^^| | | | | \
| |ExportAll |Boolean |否 |是否导出全部数据，此时filter不生效。默认为false |
|^^| | | | | \
| |UpdateFields |Map |否 |需要更新的字段值，必须是标量字段，不支持vector、sparse_vector、text 类型字段的更新 |
|^^| | | | | \
| |IgnoreError |Boolean |否 |用于数据导入。设置为 true 时遇到数据会继续解析文件，默认为 false |
|^^| | | | | \
| |UsePublic |Boolean |否 |使用公共tos |

<span id="9e839b31"></span>
## **不同TaskType的TaskConfig配置**

* **TaskType**填入的任务类型是**data_import**时，**TaskConfig**参数配置如下**：**

|参数 |类型 |是否必填 |描述 |
|---|---|---|---|
|FileType |string |是 |文件类型, json 或者 parquet，必填 |
|TosPath |string |是 |TOS 路径，格式 ：{桶名}/{路径}，注意不是域名。必填 |
|IgnoreError |bool |否 |设置为 true 时遇到数据会继续解析文件，默认为 false |


* **TaskType**填入的任务类型是**filter_update**时，**TaskConfig**参数配置如下：

|参数 |类型 |是否必填 |描述 |
|---|---|---|---|
| | | | | \
|FilterConds |Map |是 |过滤条件。使用参考https://www.volcengine.com/docs/84313/1419289 |\
| | | | |
| | | | | \
|UpdateFields |Map |是 |需要更新的字段值，必须是标量字段，不支持vector、sparse_vector、text 类型字段的更新 |


* **TaskType**填入的任务类型是**filter_delete**时，**TaskConfig**参数配置如下：

|参数 |类型 |是否必填 |描述 |
|---|---|---|---|
| | | | | \
|FilterConds |Map |是 |过滤条件。使用参考https://www.volcengine.com/docs/84313/1419289 |
| | | | | \
|UpdateFields |Map |是 |需要更新的字段值，必须是标量字段，不支持vector、sparse_vector、text 类型字段的更新 |
| | | | | \
|NeedConfirm |Boolean |否 |是否可跳过人工确认环节，默认为true |
| | | | | \
|TosPath |string |是 |TOS 路径，格式 ：{桶名}/{路径}，注意不是域名。必填 |
| | | | | \
|FileType |string |是 |文件类型, json 或者 parquet，必填 |


* **TaskType**填入的任务类型是**data_export**时，**TaskConfig**参数配置如下：

|参数 |类型 |是否必填 |描述 |
|---|---|---|---|
| | | | | \
|FilterConds |Map |否 |过滤条件。使用参考https://www.volcengine.com/docs/84313/1419289 |\
| | | | |\
| | | |* 如果不填入FilterConds，则无关ExportAll，一定导出全部数据。 |\
| | | |* 如果填入FilterConds： |\
| | | |   * 不写Exportall，或Exportall=false，则默认导出满足条件的数据。 |\
| | | |   * 写exportall=true，则强制导出全部数据，此时FilterConds不生效。 |
| | | | | \
|ExportAll |Boolean |否 |是否导出全部数据，此时filter不生效。默认为false |
| | | | | \
|TosPath |string |是 |TOS 路径，格式 ：{桶名}/{路径}，注意不是域名。必填 |
| | | | | \
|FileType |string |是 |文件类型, json 或者 parquet，必填 |


<span id="ce2f1452"></span>
## 返回参数
下表仅列出本接口特有的返回参数。更多信息请参见[返回结构](https://www.volcengine.com/docs/6369/80336)。

|参数 |类型 |示例值 |描述 |
|---|---|---|---|
|TaskId |String | |任务ID |
|Message |String |success |操作结果信息 |

<span id="895c25e7"></span>
## 请求示例
```Plain Text
action = "CreateVikingdbTask",
body = {
    "TaskType": "data_import",
    "TaskConfig": {
        "FileType": "json",
        "NeedConfirm": True,
        "TosPath": "test-doc1-tos/pic_search_1000_images.json",
        "ExportAll": False
    },
    "ProjectName": "default",
    "CollectionName": "test_doc1_image"
}
```

<span id="1a241b65"></span>
## 返回示例
```Plain Text
{
  "ResponseMetadata": {
    "RequestId": "202506091026120661022170451FDB08",
    "Action": "CreateVikingdbTask",
    "Version": "2025-06-09",
    "Service": "vikingdb",
    "Region": "cn-beijing"
  },
  "Result": {
    "TaskId": "rtB0u",
    "Message": "success"
  }
}
```

<span id="1652b6f3"></span>
## 错误码
公共错误码请参见[公共错误码](https://www.volcengine.com/docs/6369/68677)文档。