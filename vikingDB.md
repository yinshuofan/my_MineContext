接口文档：
# 请求公共参数介绍

| | | | | | \
|参数 |类型 |是否必填 |位置 |描述 |
|---|---|---|---|---|
| | | | | | \
|Action |String |是 |请求参数query中 |要执行的操作，例如CreateVikingdbCollection，具体见接口描述。 |
| | | | | | \
|Version |String |是 |请求参数query中 |API的版本，固定取值：2025-06-09。 |

# 响应体公共参数介绍
正确返回包含ResponseMetadata和Result两部分。

| | | | | \
|参数名 |一级子参数 |类型 |说明 |
|---|---|---|---|
| | | | | \
|ResponseMetadata | |Map |响应元数据 |
|^^| | | | \
| |RequestId | String |\
| | | |请求id |
|^^| | | | \
| |Action |String |请求执行的操作，示例：CreateVikingdbCollection。 |
|^^| | | | \
| |Version |String |API的版本，固定值2025-06-09。 |
|^^| | | | \
| |Service |String |目标服务的标识码，固定值vikingdb |
|^^| | | | \
| |Region |String |服务区域，如cn-beijing |
| | | | | \
|Result | |Map |如果操作没有结果或不需要结果，则返回的 result = null |

响应错误仅返回ResponseMetadata，其中包含错误信息Error。

| | | | | | \
|参数名 |一级子参数 |二级子参数 |类型 |说明 |
|---|---|---|---|---|
| | | | | | \
|ResponseMetadata | | |Map |响应元数据 |
|^^| | | | | \
| |RequestId | | String |\
| | | | |请求id |
|^^| | | | | \
| |Action | |String |请求执行的操作，示例：CreateVikingdbCollection。 |
|^^| | | | | \
| |Version | |String |API的版本，固定值2025-06-09。 |
|^^| | | | | \
| |Service | |String |目标服务的标识码，固定值vikingdb |
|^^| | | | | \
| |Region | |String |服务区域，如cn-beijing |
|^^| | | | | \
| |Error | |Map |错误信息 |
|^^|^^| | | | \
| | |Code |String |Code内容为具体的错误码，可根据错误码查询文档解决问题。 |
|^^|^^| | | | \
| | |Message |String |Message描述了错误发生的具体原因，供排查问题参考。 |


# Collection操作：
## 创建数据集-CreateVikingdbCollection
https://www.volcengine.com/docs/84313/1791154

## 查看数据集列表-ListVikingdbCollection
https://www.volcengine.com/docs/84313/1791145

## 删除数据集-DeleteVikingdbCollection
https://www.volcengine.com/docs/84313/1791148

## 查看数据集详情-GetVikingdbCollection
https://www.volcengine.com/docs/84313/1791142

# 索引操作
## 创建索引-CreateVikingdbIndex
https://www.volcengine.com/docs/84313/1791149

## 删除索引-DeleteVikingdbIndex
https://www.volcengine.com/docs/84313/1791150

## 查看索引列表-ListVikingdbIndex
https://www.volcengine.com/docs/84313/1791156

## 查看索引详情-GetVikingdbIndex
https://www.volcengine.com/docs/84313/1791157

# 数据操作
## 数据写入-UpsertData
https://www.volcengine.com/docs/84313/1791127

## 数据更新-UpdateData
https://www.volcengine.com/docs/84313/1791129

## 数据删除-DeleteData
https://www.volcengine.com/docs/84313/1791130

## 在数据集中点查-FetchDataInCollection
https://www.volcengine.com/docs/84313/1791131

## 在索引中点查-FetchDataInIndex
https://www.volcengine.com/docs/84313/1791132

# 检索操作
## 检索公共参数
https://www.volcengine.com/docs/84313/1791133

## id检索-SearchById
https://www.volcengine.com/docs/84313/1791136

## 向量检索-SearchByVector
https://www.volcengine.com/docs/84313/1791165

## 标量排序检索-SearchByScalar
https://www.volcengine.com/docs/84313/1791137

## 随机检索-SearchByRandom
https://www.volcengine.com/docs/84313/1791138

## 关键词检索-SearchByKeywords
https://www.volcengine.com/docs/84313/1791139