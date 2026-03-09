<span id="6b7bf094"></span>
# 概述
接口用于对指定索引的更新
<span id="72ab4418"></span>
# 接口升级说明

* 对应的V1接口为：https://www.volcengine.com/docs/84313/1254537
* 使用区别：

| |V2接口 |V1接口 |
|---|---|---|
|参数命名风格 |驼峰（`IndexName`） |下划线（`index_name`） |


* 注：QPS限流是以数据条数计算，V1与V2接口的限流行为完全相同。

<span id="e784a2a5"></span>
# 请求接口
:::tip
请求向量数据库 VikingDB 的 OpenAPI 接口时，可以使用 **ak、sk** **构造签名**进行鉴权。请参见[控制面API调用流程](/docs/84313/1791144)，复制调用示例并填入必要信息
:::
<span id="75df12b5"></span>
## 请求参数
请求参数Action取值：UpdateVikingdbIndex。
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
| | | | | | \
|ShardPolicy | |String |否 |索引分片类型 |\
| | | | |可选值为auto/custom，auto为自动分片、custom为自定义分片。 |\
| | | | |索引分片是指在大规模数据量场景下，将索引数据均分成多个小的索引数据块，并分发到同一个集群不同节点进行管理，每个节点负责存储和处理一部分数据，查询会同时请求不同节点上的索引数据块。由于单节点的容量有限，无法将索引全部数据存放到单节点中，因此需要设置合适的索引分片数，否则会影响索引到数据的时效性。另，分片数与成本相关， 分片数越多成本越高。 |
| | | | | | \
|Description | |String |否 |索引的自定义描述。 |
| | | | | | \
|ScalarIndex | |Array of String |否 |标量字段列表，用于设置需要构建到标量索引的字段。 |\
| | | | | |\
| | | | |* scalar_index 默认为 None，表示所有字段构建到标量索引。 |\
| | | | |* scalar_index 为 [] 时，表示无标量索引。 |\
| | | | |* scalar_index 为非空列表时，表示将列表内字段构建到标量索引。 |\
| | | | | |\
| | | | |如果标量字段进入标量索引，主要用于范围过滤和枚举过滤，会占用额外资源： |\
| | | | | |\
| | | | |* 范围过滤：float32、int64 |\
| | | | |* 枚举过滤：int64、string、list<int64>、list<string>、bool |\
| | | | | |\
| | | | |如果标量字段不进入标量索引，仍支持作为正排字段选取使用和部分正排计算。 |\
| | | | |注：更新操作只用于新增标量索引字段，不会修改或删除已有的标量索引字段。 |
| | | | | | \
|ShardCount | |Integer |否 |自定义分片数。 |\
| | | | | |\
| | | | |* 当shard_policy为auto时，shard_count不生效。 |\
| | | | |* 当shard_policy为custom时，shard_count。 |\
| | | | |   * 取值范围：[1, 256]。 |\
| | | | |   * 默认为1，分片数预估参考：数据预估数据量/3000万。 |
| | | | | | \
|CpuQuota | |Integer |否 |索引检索消耗的 CPU 配额，格式为正整数。 |\
| | | | |与吞吐量有关，和延迟无关，1CPU 核约为 100QPS。 |\
| | | | |N个分片数量N倍的 CPU 消耗；如果检索消耗的 CPU 超过配额，该索引会被限流。 |\
| | | | |取值范围：[1, 10240]。 |
| | | | | | \
|VectorIndex | | Map  |是 |向量索引配置 |\
| | | | |注：如需更新索引类型等相关配置，索引将重新构建，请谨慎更新 |
| | | | | | \
| |IndexType |String |是 |向量索引类型。详见[创建索引-CreateVikingdbIndex](/docs/84313/1791149)，取值如下：hnsw；hnsw_hybrid；flat；diskann，其中： |\
| | | | | |\
| | | | |* 若原索引类型为hnsw、diskann 或 flat：原索引类型可变更为三者中任一类型，不可变更为 hnsw_hybrid 类型 |\
| | | | |* 若原索引类型为 hnsw_hybrid ：不可变更索引类型 |\
| | | | |* 当前包含 tensor 表征类型的索引仅支持使用 hnsw，不允许变更索引类型 |
| | | | | | \
| |Distance |String |否 |距离类型，衡量向量之间距离的算法。取值如下： |\
| | | | |ip：全称是 Inner Product，内积，该算法基于向量的内积，即两个元素的对应元素相乘并求和的结果计算相似度，内积值越大相似度越高。 |\
| | | | |l2：欧几里得距离，它计算两个向量的欧几里得空间距离，欧式距离越小相似度越高。 |\
| | | | |cosine：余弦相似度（Cosine Similarity），也称为余弦距离（Cosine Distance），用于计算两个高维向量的夹角余弦值从而衡量向量相似度，夹角余弦值越小表示两向量的夹角越大，则两个向量差异越大。 |\
| | | | |当 distance=cosine 时，默认对向量做归一化处理。 |\
| | | | |对于hnsw_hybrid索引算法，距离类型选择只对稠密向量生效，稀疏向量仅支持内积。 |
|^^| | | | | \
| |Quant |String |否 |量化方式。量化方式是索引中对向量的压缩方式，可以降低向量间相似性计算的复杂度。基于向量的高维度和大规模特点，采用向量量化可以有效减少向量的存储和计算成本。取值如下： |\
| | | | | |\
| | | | |* int8：将4字节的 float 压缩为单个字节，以获取内存和计算延迟的收益，会造成微小的损失精度，比如 cosine 距离会出现大于1的分值。 |\
| | | | |* float：全精度，未做压缩量化。 |\
| | | | |* fix16：将4字节的 float 压缩为两个字节，以获取内存和计算延迟的收益，会造成微小的损失精度。通过损失一定的检索精度，提升检索性能，节约资源成本。 |\
| | | | |* pq：将高维向量转换为低维码本向量，以减少内存占用并提高搜索效率。 |\
| | | | |* 适用情况 |\
| | | | |   * int8适用于hnsw、hnsw_hybrid、flat索引算法，距离方式为ip、consine。 |\
| | | | |   * float适用于hnsw、hnsw_hybrid、flat、diskann索引算法，距离方式为ip、l2、consine。 |\
| | | | |   * fix16适用于hnsw、hnsw_hybrid、flat索引算法，距离方式为ip、l2、consine。 |\
| | | | |   * pq适用于diskann索引算法，距离方式为ip、l2、consine。 |
|^^| | | | | \
| |HnswM |Integer |否 |hnsw 索引参数，表示邻居节点个数。 |\
| | | | | |\
| | | | |* 当 index_type 配置为 hnsw 和 hnsw_hybrid 时可选配置。 |
|^^| | | | | \
| |HnswCef |Integer |否 |hnsw 索引参数，表示构建图时搜索邻居节点的广度。 |\
| | | | | |\
| | | | |* 当 index_type 配置为 hnsw 和 hnsw_hybrid 时可选配置。 |
|^^| | | | | \
| |HnswSef |Integer |否 |hnsw 索引参数，表示线上检索的搜索广度。 |\
| | | | | |\
| | | | |* 当 index_type 配置为 hnsw 和 hnsw_hybrid 时可选配置。 |
|^^| | | | | \
| |DiskannM |Integer |否 |diskann参数，标识邻居节点个数。 |\
| | | | | |\
| | | | |* 当 index_type 配置为 diskann时可选配置。 |
|^^| | | | | \
| |DiskannCef |Integer |否 |diskann参数，表示构建图时搜索邻居节点的广度。 |\
| | | | | |\
| | | | |* 当 index_type 配置为 diskann时可选配置。 |
|^^| | | | | \
| |PqCodeRatio |Float |否 |diskann参数，向量维度编码的大小限制。值越大，召回率越高，但会增加内存使用量，范围 (0.0, 0.25]。 |\
| | | | | |\
| | | | |* 当 index_type 配置为 diskann时可选配置。 |
|^^| | | | | \
| |CacheRatio |Float |否 |diskann参数，缓存节点数与原始数据的比率，较大的值会提高索引性能并增加内存使用量。范围 [0.0,0.3)。 |\
| | | | | |\
| | | | |* 当 index_type 配置为 diskann时可选配置。 |

<span id="6306949a"></span>
## 返回参数
下表仅列出本接口特有的返回参数。更多信息请参见[返回结构](https://www.volcengine.com/docs/6369/80336)。

|参数 |类型 |示例值 |描述 |
|---|---|---|---|
|Message |String |success |操作结果信息 |

<span id="532d186a"></span>
## 请求示例
```Plain Text
action = "UpdateVikingdbIndex"
body = {
  "Description": "it is a test index",
  "CpuQuota": 10,
  "CollectionName": "test_coll",
  "IndexName": "test_index_1"
}
```

<span id="104ff208"></span>
## 返回示例
```Plain Text
{
  "ResponseMetadata": {
    "RequestId": "20250527175413216098005053ADD0C8",
    "Action": "UpdateVikingdbIndex",
    "Version": "2025-06-09",
    "Service": "vikingdb",
    "Region": "cn-beijing"
  },
  "Result": {
    "Message": "success"
  }
}
```

<span id="5495ab87"></span>
## 错误码
错误码请参见[公共错误码](https://www.volcengine.com/docs/6369/68677)文档。