# VikingDB API V2 参考文档

## 目录结构

```
├── 01-数据面API/
│   ├── 01-数据/
│   └── 02-检索/
├── 02-控制面API/
│   ├── 01-数据集/
│   ├── 02-索引/
│   └── 03-任务/
└── 03-错误码/
```

---

## API 索引

### 一、数据面 API

#### 1.1 数据操作

| 接口 | 说明 |
|------|------|
| [UpsertData](01-数据面API/01-数据/UpsertData.md) | 向指定 Collection 中写入数据，支持单条或批量写入（单次上限 100 条） |
| [UpdateData](01-数据面API/01-数据/UpdateData.md) | 更新已有数据的部分字段，支持文本、标量、向量字段的局部更新 |
| [DeleteData](01-数据面API/01-数据/DeleteData.md) | 删除指定 Collection 中的数据，支持单条或批量删除（单次上限 100 条） |
| [FetchDataInCollection](01-数据面API/01-数据/FetchDataInCollection.md) | 根据主键查询指定 Collection 中的单条或多条数据（单次上限 100 条） |
| [FetchDataInIndex](01-数据面API/01-数据/FetchDataInIndex.md) | 根据主键查询指定 Index 中的单条或多条数据（单次上限 100 条） |

#### 1.2 检索

| 接口 | 说明 |
|------|------|
| [检索通用参数](01-数据面API/02-检索/检索通用参数.md) | 检索接口的通用参数说明，涵盖各检索模式的公共配置 |
| [SearchByVector](01-数据面API/02-检索/SearchByVector.md) | 向量检索，基于向量空间模型计算相似度，高效检索 K 个相似向量 |
| [SearchById](01-数据面API/02-检索/SearchById.md) | 主键检索，以指定 ID 对应的向量值作为检索条件进行向量检索 |
| [SearchByScalar](01-数据面API/02-检索/SearchByScalar.md) | 标量检索，基于 int64/float32 等标量字段类型进行数据检索 |
| [SearchByKeywords](01-数据面API/02-检索/SearchByKeywords.md) | 关键词检索，结合传统关键词匹配与向量语义检索，支持全文搜索 |
| [SearchByMultiModal](01-数据面API/02-检索/SearchByMultiModal.md) | 多模态检索，支持文本+图片、文本+视频等多模态数据直接搜索及结果重排 |
| [SearchByRandom](01-数据面API/02-检索/SearchByRandom.md) | 随机检索，返回随机记录，支持过滤与后处理，适用于数据对比和筛选场景 |
| [Aggregate](01-数据面API/02-检索/Aggregate.md) | 聚合统计，对指定字段进行分组聚合并支持过滤操作，用于数据分布分析 |
| [Embedding](01-数据面API/02-检索/Embedding.md) | Embedding 计算接口升级说明，包含新旧接口模型组合差异 |
| [PostProcess](01-数据面API/02-检索/PostProcess.md) | 后处理能力说明，描述检索→过滤→后处理的流水线执行流程 |

### 二、控制面 API

#### 2.1 数据集（Collection）

| 接口 | 说明 |
|------|------|
| [CreateVikingdbCollection](02-控制面API/01-数据集/CreateVikingdbCollection.md) | 创建向量数据库 Collection |
| [GetVikingdbCollection](02-控制面API/01-数据集/GetVikingdbCollection.md) | 查询指定 Collection 的详细信息 |
| [ListVikingdbCollection](02-控制面API/01-数据集/ListVikingdbCollection.md) | 查询 Collection 列表 |
| [UpdateVikingdbCollection](02-控制面API/01-数据集/UpdateVikingdbCollection.md) | 更新指定 Collection |
| [DeleteVikingdbCollection](02-控制面API/01-数据集/DeleteVikingdbCollection.md) | 删除指定 Collection |

#### 2.2 索引（Index）

| 接口 | 说明 |
|------|------|
| [CreateVikingdbIndex](02-控制面API/02-索引/CreateVikingdbIndex.md) | 创建索引（部分地域支持 DiskANN） |
| [GetVikingdbIndex](02-控制面API/02-索引/GetVikingdbIndex.md) | 查看指定索引的详细信息 |
| [ListVikingdbIndex](02-控制面API/02-索引/ListVikingdbIndex.md) | 查看索引列表 |
| [UpdateVikingdbIndex](02-控制面API/02-索引/UpdateVikingdbIndex.md) | 更新指定索引 |
| [EnableVikingdbIndex](02-控制面API/02-索引/EnableVikingdbIndex.md) | 启用已禁用的索引，重新构建并分配资源 |
| [DisableVikingdbIndex](02-控制面API/02-索引/DisableVikingdbIndex.md) | 禁用就绪状态的索引，释放内存资源 |
| [DeleteVikingdbIndex](02-控制面API/02-索引/DeleteVikingdbIndex.md) | 删除指定索引 |

#### 2.3 离线任务（Task）

| 接口 | 说明 |
|------|------|
| [CreateVikingdbTask](02-控制面API/03-任务/CreateVikingdbTask.md) | 创建离线任务 |
| [GetVikingdbTask](02-控制面API/03-任务/GetVikingdbTask.md) | 查看指定离线任务详情 |
| [ListVikingdbTask](02-控制面API/03-任务/ListVikingdbTask.md) | 查看离线任务列表 |
| [UpdateVikingdbTask](02-控制面API/03-任务/UpdateVikingdbTask.md) | 更新指定离线任务 |
| [DeleteVikingdbTask](02-控制面API/03-任务/DeleteVikingdbTask.md) | 删除指定离线任务 |

### 三、错误码

| 文档 | 说明 |
|------|------|
| [错误码与问题排查](03-错误码/错误码与问题排查.md) | API V2 错误码一览及常见问题排查指南 |

---

## 源地址

https://www.volcengine.com/docs/84313/1791127?lang=zh
