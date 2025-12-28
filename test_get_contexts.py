#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 get_all_processed_contexts 接口"""

from opencontext.storage.global_storage import get_storage


def test_get_processed_contexts():
    """测试获取处理后的上下文"""
    storage = get_storage()
    
    filter_conditions = {
        "create_time_ts": {"$gte": 1766836980.0, "$lte": 1766923380.0},
        # "has_compression": False,
        # "enable_merge": True,
    }
    
    limit = 100
    offset = 0
    
    print("开始测试 get_all_processed_contexts 接口...")
    print(f"过滤条件: {filter_conditions}")
    print(f"用户ID: user_321")
    print(f"设备ID: device_321")
    print(f"代理ID: agent_321")
    print("-" * 50)
    
    results = storage.get_all_processed_contexts(
        limit=limit,
        offset=offset,
        filter=filter_conditions,
        # user_id='user_321',
        # device_id='device_321',
        # agent_id='agent_321',
    )
    
    # from opencontext.models.context import ProcessedContext, RawContextProperties, Vectorize
    # query = "冰淇淋"
    # query_vectorize = Vectorize(text=query)
    # results = storage.search(
    #     query=query_vectorize,
    #     filters=filter_conditions,
    #     # user_id='user_321',
    #     # device_id='device_321',
    #     # agent_id='agent_321',
    # )
    # print(results)
    
    print(f"\n查询结果 (按后端分组):")
    total_count = 0
    for backend_name, contexts in results.items():
        if contexts:
            print(f"\n后端 '{backend_name}': {len(contexts)} 条记录")
            for ctx in contexts[:3]:
                print(f"  - ID: {ctx.id}")
                print(f"    类型: {ctx.extracted_data.context_type.value}")
                print(f"    创建时间: {ctx.properties.create_time}")
                print(f"    更新时间: {ctx.properties.update_time}")
            if len(contexts) > 3:
                print(f"  ... 还有 {len(contexts) - 3} 条记录")
        total_count += len(contexts)
    
    print("-" * 50)
    print(f"总计: {total_count} 条记录")
    return results


if __name__ == "__main__":
    test_get_processed_contexts()