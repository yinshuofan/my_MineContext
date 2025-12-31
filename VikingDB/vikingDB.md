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
## 创建数据集-CreateVikingdbCollection.md

## 查看数据集列表-ListVikingdbCollection.md

## 删除数据集-DeleteVikingdbCollection.md

## 查看数据集详情-GetVikingdbCollection.md

# 索引操作
## 创建索引-CreateVikingdbIndex.md

## 删除索引-DeleteVikingdbIndex.md

## 查看索引列表-ListVikingdbIndex.md

## 查看索引详情-GetVikingdbIndex.md

# 数据操作
## 数据写入-UpsertData.md

## 数据更新-UpdateData.md

## 数据删除-DeleteData.md

## 在数据集中点查-FetchDataInCollection.md

## 在索引中点查-FetchDataInIndex.md

# 检索操作
## 检索公共参数.md

## id检索-SearchById.md

## 向量检索-SearchByVector.md

## 标量排序检索-SearchByScalar.md

## 随机检索-SearchByRandom.md

## 关键词检索-SearchByKeywords.md