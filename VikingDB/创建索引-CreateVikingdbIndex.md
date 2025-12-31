<span id="d07f3366"></span>
# 概述
接口用于对指定索引的创建
<span id="37a88875"></span>
# 地域功能开放说明

| | | \
|**地域** |**Diskann功能** |
|---|---|
| | | \
|华北 |支持 |
| | | \
|华东 |支持 |
| | | \
|华南 |暂不支持，正在开发中 |
| | | \
|柔佛 |暂不支持，正在开发中 |

<span id="0e7d9842"></span>
# 接口升级说明

* 对应的V1接口为：https://www.volcengine.com/docs/84313/1254531
* 使用区别：


| | | | \
| |V2接口 |V1接口 |
|---|---|---|
| | | | \
|参数命名风格 |驼峰（`IndexName`） |下划线（`index_name`） |


* 注：QPS限流是以数据条数计算，V1与V2接口的限流行为完全相同。

<span id="a0a01701"></span>
# 请求接口
:::tip
请求向量数据库 VikingDB 的 OpenAPI 接口时，可以使用 **ak、sk** **构造签名**进行鉴权。请参见[控制面API调用流程](/docs/84313/1791144)，复制调用示例并填入必要信息
:::
<span id="d3af65bc"></span>
## 请求参数
请求参数Action取值：CreateVikingdbIndex。
下表仅列出该接口特有的请求参数和部分公共参数。更多信息请见[公共参数](https://www.volcengine.com/docs/6369/67268)。

| | | | | | \
|参数 |一级子参数 |类型 |是否必填 |描述 |
|---|---|---|---|---|
| | | | | | \
|ProjectName | |String |否 |项目名称 |
| | | | | | \
|CollectionName | |String |2选1 |数据集名称 |
| | | |^^| | \
|ResourceId | |String | |数据集资源ID。请求必须指定ResourceId和CollectionName其中之一。 |
| | | | | | \
|Description | |String |否 |索引的自定义描述。 |
| | | | | | \
|VectorIndex | | Map  |是 |向量索引配置 |
| | | | | | \
| |IndexType |String |是 |向量索引类型。取值如下： |\
| | | | | |\
| | | | |* hnsw：全称是 Hierarchical Navigable Small World，一种用于在高维空间中采用 ANN 搜索的数据结构和算法，是基于图的索引。HNSW通过构建多层网络减少搜索过程中需要访问的节点数量，实现快速高效地搜索最近邻，适合对搜索效率要求较高的场景。 |\
| | | | |   hnsw的相关参数包含 **Quant、Distance、Hnsw_m、Hnsw_cef、Hnsw_sef**。 |\
| | | | |* hnsw_hybrid：支持混合索引的 hnsw 算法。混合索引算法可以同时对数据集中的稠密向量和稀疏向量进行索引，并在检索时返回兼顾两种类型相似性的结果。适用于对搜索效率要求较高，且需要同时检索稀疏和稠密向量的场景。 |\
| | | | |   hnsw_hybrid的相关参数包含 **Quant、Distance、HnswM、HnswCef、HnswSef**。 |\
| | | | |   hnsw_hybrid所索引的数据集必须包含 sparse_vector类型数据，即定义了sparse_vector类型字段，或绑定了能产生sparse_vector 类型向量的 pipeline。 |\
| | | | |* flat：暴力索引，搜索时遍历整个向量数据库的所有向量与目标向量进行距离计算和比较，查询速度较慢，但是 flat 能提供100％的检索召回率，适用于向量候选集较少，且需要100％检索召回率的场景。 |\
| | | | |   flat 的相关参数包含 **Quant、Distance**。 |\
| | | | |* diskann：基于 Vamana 图的磁盘索引算法，将 Vamana 图与 PQ 量化压缩方案结合，构建DiskANN索引。图索引和原始数据存在SSD中，压缩索引放在内存中。检索请求时会将query向量与聚簇中心比较，然后从磁盘读取对应的原始数据进行算分。适用于大规模数据量，性能不是特别敏感，内存成本更低，且召回率较高的场景。 |\
| | | | |   diskann的相关参数包含 **Quant、Distance、DiskannM、DiskannCef、CacheRatio、PqCodeRatio。** |
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
| | | | | |\
| | | | |int8适用于hnsw、hnsw_hybrid、flat索引算法，距离方式为ip、consine。 |\
| | | | | |\
| | | | |float适用于hnsw、hnsw_hybrid、flat、diskann索引算法，距离方式为ip、l2、consine。 |\
| | | | | |\
| | | | |fix16适用于hnsw、hnsw_hybrid、flat索引算法，距离方式为ip、l2、consine。 |\
| | | | | |\
| | | | |pq适用于diskann索引算法，距离方式为ip、l2、consine。 |
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
| | | | |如果标量字段不进入标量索引，仍支持作为正排字段选取使用和部分正排计算。 |
| | | | | | \
|CpuQuota | |Integer |否 |索引检索消耗的 CPU 配额，格式为正整数。 |\
| | | | |与吞吐量有关，和延迟无关，1CPU 核约为 100QPS。 |\
| | | | |N个分片数量N倍的 CPU 消耗；如果检索消耗的 CPU 超过配额，该索引会被限流。 |\
| | | | |取值范围：[1, 10240]。 |
| | | | | | \
|ShardPolicy | |String |否 |索引分片类型 |\
| | | | |可选值为auto/custom，auto为自动分片、custom为自定义分片。 |\
| | | | |索引分片是指在大规模数据量场景下，将索引数据均分成多个小的索引数据块，并分发到同一个集群不同节点进行管理，每个节点负责存储和处理一部分数据，查询会同时请求不同节点上的索引数据块。由于单节点的容量有限，无法将索引全部数据存放到单节点中，因此需要设置合适的索引分片数，否则会影响索引到数据的时效性。另，分片数与成本相关， 分片数越多成本越高。 |
| | | | | | \
|ShardCount | |Integer |否 |自定义分片数。 |\
| | | | | |\
| | | | |* 当shard_policy为auto时，shard_count不生效。 |\
| | | | |* 当shard_policy为custom时，shard_count。 |\
| | | | |   * 取值范围：[1, 256]。 |\
| | | | |   * 默认为1，分片数预估参考：数据预估数据量/3000万。 |
| | | | | | \
|IndexName | |String |是 |索引名称 |

<span id="3bc86647"></span>
## 返回参数
下表仅列出本接口特有的返回参数。更多信息请参见[返回结构](https://www.volcengine.com/docs/6369/80336)。

| | | | | \
|参数 |类型 |示例值 |描述 |
|---|---|---|---|
| | | | | \
|Message |String |success |操作结果信息 |

<span id="fb665dd7"></span>
## 请求示例-分场景

---


<span id="e5045fd6"></span>
### 1. 简单的纯向量检索场景
例如：图片检索、视频检索等纯向量检索场景。
<span id="8161364c"></span>
#### 1.1 低延时、高吞吐
hnsw 索引适用于**快速、高吞吐量**向量搜索场景。一般推荐 Distance 设为 “ip”，Quant 量化方式设为 “int8” 以提升内存效率，ShardPolicy 用默认 “auto”，此时后台自动开启分片策略可避免容量瓶颈、提高检索效率，且不影响总体内存占用。 
```Plain Text
action = "CreateVikingdbIndex",
body = {
    "CollectionName": "your_collection",
    "IndexName": "idx_1",
    "VectorIndex": {
        "IndexType": "hnsw",
        "Distance": "ip",
        "Quant": "int8"
    }
}
```

<span id="6e64b6c3"></span>
#### 1.2 低吞吐、大数据量
当**数据规模超大，且对成本敏感，而检索QPS不高**的情况下，可以考虑使用内存占用更低的diskann索引来替代hnsw索引。
```Plain Text
action = "CreateVikingdbIndex",
body = {
    "CollectionName": "your_collection",
    "IndexName": "idx_1",
    "VectorIndex": {
        "IndexType": "diskann",
        "Distance": "ip",
        "Quant": "pq"
    }
}
```

<span id="65a1da42"></span>
### 2. 增强关键词检索能力的场景
若向量检索含文本信息，如**新闻、论文文章**检索场景，可创建带稀疏向量字段的数据集，创建索引时选带稀疏向量能力的索引。检索时采用稠密 + 稀疏向量混合检索，有助于提升捕获关键词信息的能力。
**低延时、高吞吐**场景对应上述1.1，将hnsw索引替换为hnsw_hybrid索引。
```Plain Text
action = "CreateVikingdbIndex",
body = {
    "CollectionName": "your_collection",
    "IndexName": "idx_1",
    "VectorIndex": {
        "IndexType": "hnsw_hybrid",
        "Distance": "ip",
        "Quant": "int8"
    }
}
```

<span id="0eda8849"></span>
### 3.向量检索+标量过滤场景
若在向量检索时带有某些标量字段过滤条件，可在向量索引基础上设置标量索引字段（ScalarIndex）。默认开启所有标量字段索引，用于范围和枚举过滤；若仅部分标量字段需开启索引，可在ScalarIndex中配置字段列表。 
例如，您有一个商品数据集，包括的字段有：

| | | | | \
|字段名 |类型 |含义 |用途 |
|---|---|---|---|
| | | | | \
|product_info |text |商品名和描述 |用于向量化、语义向量检索 |
| | | | | \
|product_type |string |商品类别 |可标量枚举过滤 |
| | | | | \
|manufacturer_name |string |制造商名 |可标量枚举过滤 |
| | | | | \
|price |float |价格 |可标量范围过滤 |
| | | | | \
|sales |int64 |销量 |可标量范围、枚举过滤 |
| | | | | \
|shipping_info |string |发货地点和方式说明 |不进索引、仅作为属性列 |
| | | | | \
|comment |string |备注信息 |不进索引、仅作为属性列 |

可按如下方式创建索引：
```Plain Text
action = "CreateVikingdbIndex",
body = {
    "CollectionName": "your_collection",
    "IndexName": "idx_1",
    "VectorIndex": {
        "IndexType": "hnsw_hybrid",
        "Distance": "ip",
        "Quant": "int8"
    },
    "ScalarIndex": [
        "product_type",
        "brand_name",
        "price",
        "sales"
    ]
}
```

<span id="2247f45c"></span>
## 返回示例
```Plain Text
{
  "ResponseMetadata": {
    "RequestId": "202505271754501200391670290C48BB",
    "Action": "CreateVikingdbIndex",
    "Version": "2025-06-09",
    "Service": "vikingdb",
    "Region": "cn-beijing"
  },
  "Result": {
    "Message": "success"
  }
}
```