<span id="763bf361"></span>
# 概述
接口用于对指定索引详情的查看
<span id="3ae85523"></span>
# 接口升级说明

* 对应的V1接口为：https://www.volcengine.com/docs/84313/1254550
* 使用区别：


| | | | \
| |V2接口 |V1接口 |
|---|---|---|
| | | | \
|参数命名风格 |驼峰（`IndexName`） |下划线（`index_name`） |


* 注：QPS限流是以数据条数计算，V1与V2接口的限流行为完全相同。

<span id="f7e44904"></span>
# 请求接口
:::tip
请求向量数据库 VikingDB 的 OpenAPI 接口时，可以使用 **ak、sk** **构造签名**进行鉴权。请参见[控制面API调用流程](/docs/84313/1791144)，复制调用示例并填入必要信息
:::
<span id="e6291570"></span>
## 请求参数
请求参数Action取值：GetVikingdbIndex。
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

<span id="7117aafa"></span>
## 返回参数
下表仅列出本接口特有的返回参数。更多信息请参见[返回结构](https://www.volcengine.com/docs/6369/80336)。

| | | | | \
|参数 |一级子参数 |类型 |描述 |
|---|---|---|---|
| | | | | \
|CollectionName | |String |数据集名称 |
| | | | | \
|ProjectName | |String |项目名称 |
| | | | | \
|ResourceId | |String |资源ID |
| | | | | \
|IndexName | |String |索引名称 |
| | | | | \
|CpuQuota | |Integer |索引检索消耗的 CPU 配额。 |
| | | | | \
|ShardPolicy | |String |索引分片类型，auto为自动分片、custom为自定义分片。 |
| | | | | \
|ShardCount | |Integer |索引分片数 |
| | | | | \
|Description | |String |索引描述 |
| | | | | \
|VectorIndex | |Map |向量索引配置 |
|^^| | | | \
| |IndexType |String |索引类型 |
|^^| | | | \
| |Distance |String |距离类型，衡量向量之间距离的算法。取值如下： |\
| | | |ip：全称是 Inner Product，内积，该算法基于向量的内积，即两个元素的对应元素相乘并求和的结果计算相似度，内积值越大相似度越高。 |\
| | | |l2：欧几里得距离，它计算两个向量的欧几里得空间距离，欧式距离越小相似度越高。 |\
| | | |cosine：余弦相似度（Cosine Similarity），也称为余弦距离（Cosine Distance），用于计算两个高维向量的夹角余弦值从而衡量向量相似度，夹角余弦值越小表示两向量的夹角越大，则两个向量差异越大。 |\
| | | |当 distance=cosine 时，默认对向量做归一化处理。 |\
| | | |当索引算法选择IVF时，距离类型可选ip、cosine。 |\
| | | |对于hnsw_hybrid索引算法，距离类型选择只对稠密向量生效，稀疏向量仅支持内积。 |
|^^| | | | \
| |Quant |String |量化方式。量化方式是索引中对向量的压缩方式，可以降低向量间相似性计算的复杂度。基于向量的高维度和大规模特点，采用向量量化可以有效减少向量的存储和计算成本。取值如下： |\
| | | | |\
| | | |* int8：将4字节的 float 压缩为单个字节，以获取内存和计算延迟的收益，会造成微小的损失精度，比如 cosine 距离会出现大于1的分值。 |\
| | | |* float：全精度，未做压缩量化。 |\
| | | |* fix16：将4字节的 float 压缩为两个字节，以获取内存和计算延迟的收益，会造成微小的损失精度。通过损失一定的检索精度，提升检索性能，节约资源成本。 |\
| | | |* pq：将高维向量转换为低维码本向量，以减少内存占用并提高搜索效率。 |\
| | | | |\
| | | |int8适用于hnsw、hnsw_hybrid、flat索引算法，距离方式为ip、consine。 |\
| | | | |\
| | | |float适用于hnsw、hnsw_hybrid、flat、diskann索引算法，距离方式为ip、l2、consine。 |\
| | | | |\
| | | |fix16适用于hnsw、hnsw_hybrid、flat索引算法，距离方式为ip、l2、consine。 |\
| | | | |\
| | | |pq适用于diskann、ivf索引算法，距离方式为ip、l2、consine。 |
|^^| | | | \
| |HnswM |Integer |hnsw 索引参数，表示邻居节点个数。 |\
| | | | |\
| | | |* 当 index_type 配置为 hnsw 和 hnsw_hybrid 时可选配置。 |
|^^| | | | \
| |HnswCef |Integer |hnsw 索引参数，表示构建图时搜索邻居节点的广度。 |\
| | | | |\
| | | |* 当 index_type 配置为 hnsw 和 hnsw_hybrid 时可选配置。 |
|^^| | | | \
| |HnswSef |Integer |hnsw 索引参数，表示线上检索的搜索广度。 |\
| | | | |\
| | | |* 当 index_type 配置为 hnsw 和 hnsw_hybrid 时可选配置。 |
|^^| | | | \
| |DiskannM |Integer |diskann参数，标识邻居节点个数。 |\
| | | | |\
| | | |* 当 index_type 配置为 diskann时可选配置。 |
|^^| | | | \
| |DiskannCef |Integer |diskann参数，表示构建图时搜索邻居节点的广度。 |\
| | | | |\
| | | |* 当 index_type 配置为 diskann时可选配置。 |
|^^| | | | \
| |PqCodeRatio |Float |diskann参数，向量维度编码的大小限制。值越大，召回率越高，但会增加内存使用量，范围 (0.0, 0.25]。 |\
| | | | |\
| | | |* 当 index_type 配置为 diskann时可选配置。 |
|^^| | | | \
| |CacheRatio |Float |diskann参数，缓存节点数与原始数据的比率，较大的值会提高索引性能并增加内存使用量。范围 [0.0,0.3)。 |\
| | | | |\
| | | |* 当 index_type 配置为 diskann时可选配置。 |
| | | | | \
|ScalarIndex | |Array of Map |标量字段列表 |\
| | | |```JSON |\
| | | |"ScalarIndex": [ |\
| | | |    { |\
| | | |        "FieldName": "f_int64_1", #string |\
| | | |        "FieldType": "int64", #string |\
| | | |        "DefaultValue": 0 #any |\
| | | |    }, |\
| | | |``` |\
| | | | |
| | | | | \
|ActualCU | |Integer |实际CU用量 |

<span id="5ac7a135"></span>
## 请求示例
```Plain Text
action = "GetVikingdbIndex",
body = {
  "CollectionName": "coll_test1",
  "IndexName": "idx_test1"
}
```

<span id="5f58809e"></span>
## 返回示例
```Plain Text
{
  "ResponseMetadata": {
    "RequestId": "20250527175301151172242186A24F67",
    "Action": "GetVikingdbIndex",
    "Version": "2025-06-09",
    "Service": "vikingdb",
    "Region": "cn-beijing"
  },
  "Result": {
    "CpuQuota": 1,
    "ShardPolicy": "auto",
    "ShardCount": 1,
    "Description": "",
    "ScalarIndex": [
     {
     "FieldName": "f_int64_1",
     "FieldType": "int64",
     "DefaultValue": 0
     },
     {
     "FieldName": "f_string_1",
     "FieldType": "string",
     "DefaultValue": "default"
     }
    ],
    "VectorIndex": {
      "IndexType": "hnsw",
      "Distance": "ip",
      "Quant": "int8",
      "HnswM": 20,
      "HnswCef": 400,
      "HnswSef": 400,
    },
    "IndexCost": {
      "CpuCore": 1,
      "MemGb": 2
    },
    "CollectionName": "coll_test1",
    "IndexName": "idx_test1",
    "ProjectName": "default",
    "ResourceId": "vdb-abcabcabc"
  }
}
```