#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成假数据并添加到VikingDB数据库的脚本
生成2025年12月1日到2025年12月30日的数据，每天3条
"""

import datetime
import json
import random
import uuid

from opencontext.models.context import (
    ContextProperties,
    ExtractedData,
    ProcessedContext,
    RawContextProperties,
    Vectorize,
)
from opencontext.models.enums import ContentFormat, ContextSource, ContextType
from opencontext.storage.backends.vikingdb_backend import VikingDBBackend


SCENARIOS = [
    {
        "title": "用户询问当前时间",
        "summary": "用户向助手发起当前时间查询，助手回复当前时间为2025年12月30日17:19:02。",
        "keywords": ["时间查询", "当前时间"],
        "entities": ["当前时间"],
    },
    {
        "title": "用户询问天气情况",
        "summary": "用户询问今天北京的天气状况，助手回复今天北京晴朗，气温15-22摄氏度。",
        "keywords": ["天气查询", "北京天气"],
        "entities": ["北京"],
    },
    {
        "title": "用户设置提醒事项",
        "summary": "用户要求助手在下午3点提醒开会，助手成功设置了提醒事项。",
        "keywords": ["提醒事项", "日程管理"],
        "entities": ["会议"],
    },
    {
        "title": "用户查询日程安排",
        "summary": "用户询问今天的日程安排，助手查询后告知用户今天有两个会议和一个待办事项。",
        "keywords": ["日程安排", "今日计划"],
        "entities": [],
    },
    {
        "title": "用户查询快递信息",
        "summary": "用户询问快递单号SF123456789的物流信息，助手查询后告知包裹正在派送中。",
        "keywords": ["快递查询", "物流信息"],
        "entities": ["快递", "包裹"],
    },
    {
        "title": "用户翻译句子",
        "summary": "用户请求将Hello World翻译成中文，助手回复你好世界。",
        "keywords": ["翻译", "中英文翻译"],
        "entities": [],
    },
    {
        "title": "用户询问计算问题",
        "summary": "用户询问365乘以48等于多少，助手计算后回复17520。",
        "keywords": ["数学计算", "乘法运算"],
        "entities": [],
    },
    {
        "title": "用户查询新闻资讯",
        "summary": "用户请求推荐今天的科技新闻，助手整理了三条热门科技资讯。",
        "keywords": ["新闻资讯", "科技新闻"],
        "entities": [],
    },
    {
        "title": "用户设置闹钟",
        "summary": "用户要求助手在明天早上7点设置闹钟，助手成功设置了闹钟提醒。",
        "keywords": ["闹钟设置", "起床提醒"],
        "entities": [],
    },
    {
        "title": "用户查询汇率",
        "summary": "用户询问美元对人民币的汇率，助手查询后告知当前汇率约为7.24。",
        "keywords": ["汇率查询", "货币兑换"],
        "entities": ["美元", "人民币"],
    },
]

CONTENT_TEMPLATES = [
    '[{{"role": "user", "content": "{user_msg}", "timestamp": "{timestamp1}"}}, {{"role": "assistant", "content": "{assistant_msg}", "timestamp": "{timestamp2}"}}]',
    '[{{"role": "user", "content": "{user_msg}", "timestamp": "{timestamp1}"}}, {{"role": "assistant", "content": "{assistant_msg}", "timestamp": "{timestamp2}"}}, {{"role": "user", "content": "谢谢", "timestamp": "{timestamp3}"}}]',
]


def generate_chat_content(scenario: dict, event_time: datetime.datetime) -> str:
    """生成聊天记录内容"""
    user_messages = [
        "现在几点了",
        "今天北京天气怎么样",
        "下午3点提醒我开会",
        "我今天有什么日程",
        "帮我查一下快递SF123456789",
        "把Hello World翻译成中文",
        "365乘以48等于多少",
        "有什么科技新闻吗",
        "明天早上7点叫我起床",
        "美元对人民币汇率是多少",
    ]
    assistant_messages = [
        f"当前时间是{event_time.strftime('%Y年%m月%d日 %H:%M:%S')}。",
        f"今天北京晴朗，气温15-22摄氏度。",
        "好的，已经在下午3点设置开会提醒。",
        f"您今天有两个会议（上午10点和下午3点）和一个待办事项（完成报告）。",
        "快递SF123456789正在派送中，预计今天送达。",
        "Hello World的中文翻译是：你好世界。",
        "365乘以48等于17520。",
        "今日科技新闻：1.某公司发布新产品；2.科技股普遍上涨；3.人工智能又有新突破。",
        "好的，已经在明天早上7点设置闹钟。",
        "当前美元对人民币汇率约为7.24。",
    ]

    template = random.choice(CONTENT_TEMPLATES)

    user_msg = random.choice(user_messages)
    assistant_msg = random.choice(assistant_messages)

    ts1 = event_time.strftime("%Y-%m-%dT%H:%M:%S.%f")
    ts2 = (event_time + datetime.timedelta(seconds=random.randint(1, 5))).strftime("%Y-%m-%dT%H:%M:%S.%f")
    ts3 = (event_time + datetime.timedelta(seconds=random.randint(6, 10))).strftime("%Y-%m-%dT%H:%M:%S.%f")

    return template.format(
        user_msg=user_msg,
        assistant_msg=assistant_msg,
        timestamp1=ts1,
        timestamp2=ts2,
        timestamp3=ts3
    )


def generate_one_context(
    date: datetime.datetime,
    index: int,
    scenario: dict
) -> ProcessedContext:
    """生成单条ProcessedContext数据"""

    base_time = datetime.datetime(2025, 12, 1, 0, 0, 0)
    event_time = base_time + datetime.timedelta(days=date.day - 1, hours=random.randint(8, 22), minutes=random.randint(0, 59))

    chat_content = generate_chat_content(scenario, event_time)

    raw_properties = RawContextProperties(
        content_format=ContentFormat.TEXT,
        source=ContextSource.CHAT_LOG,
        create_time=event_time,
        object_id=str(uuid.uuid4()),
        content_text=chat_content,
        additional_info={
            "message_count": 2,
            "roles": ["user", "assistant"]
        },
        enable_merge=True,
        user_id="user_321",
        device_id="device_321",
        agent_id="agent_321"
    )

    context_properties = ContextProperties(
        raw_properties=[raw_properties],
        create_time=event_time,
        event_time=event_time,
        is_processed=True,
        has_compression=False,
        update_time=event_time + datetime.timedelta(seconds=random.randint(10, 30)),
        call_count=0,
        merge_count=0,
        duration_count=1,
        enable_merge=True,
        is_happend=False,
        last_call_time=None,
        file_path=None,
        raw_type=None,
        raw_id=None,
        user_id="user_321",
        device_id="device_321",
        agent_id="agent_321"
    )

    extracted_data = ExtractedData(
        title=scenario["title"],
        summary=scenario["summary"],
        keywords=scenario["keywords"],
        entities=scenario["entities"],
        context_type=ContextType.ACTIVITY_CONTEXT,
        confidence=random.randint(7, 10),
        importance=random.randint(1, 5)
    )

    vectorize = Vectorize(
        content_format=ContentFormat.TEXT,
        image_path=None,
        text=f"{scenario['title']}\n{scenario['summary']}\n{' '.join(scenario['keywords'])}",
        vector=None
    )

    return ProcessedContext(
        id=str(uuid.uuid4()),
        properties=context_properties,
        extracted_data=extracted_data,
        vectorize=vectorize,
        metadata={}
    )


def generate_all_contexts(start_date: datetime.datetime, end_date: datetime.datetime, per_day: int = 3) -> list[ProcessedContext]:
    """生成指定日期范围内的所有上下文数据"""

    contexts = []
    current_date = start_date

    while current_date <= end_date:
        for i in range(per_day):
            scenario = random.choice(SCENARIOS)
            context = generate_one_context(current_date, i, scenario)
            contexts.append(context)
        current_date += datetime.timedelta(days=1)

    return contexts


def init_vikingdb_backend() -> VikingDBBackend:
    """初始化VikingDB后端"""
    config = {
        "config": {
            "access_key_id": "AKLTY2YyZGQxNGQxMWI0NGUzOTkzZTU1Mjg0NjlkYWFhOTE",
            "secret_access_key": "TUdSbVl6YzVOMkpoWkdGaE5Ea3pNamszT0RnNFltTTNOV0kzWkRBNU9Uaw==",
            "region": "cn-beijing",
            "dimension": 1024,
            "collection_name": "opencontext",
        }
    }
    backend = VikingDBBackend()
    backend.initialize(config)
    return backend


def main():
    """主函数"""
    print("开始生成假数据...")

    start_date = datetime.datetime(2025, 12, 1)
    end_date = datetime.datetime(2025, 12, 30)

    print(f"生成时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")

    contexts = generate_all_contexts(start_date, end_date, per_day=3)
    print(f"共生成 {len(contexts)} 条数据")

    print("正在初始化VikingDB后端...")
    backend = init_vikingdb_backend()

    print("正在上传数据到数据库...")
    batch_size = 50
    for i in range(0, len(contexts), batch_size):
        batch = contexts[i:i + batch_size]
        stored_ids = backend.batch_upsert_processed_context(batch)
        print(f"已上传 {min(i + batch_size, len(contexts))}/{len(contexts)} 条数据，存储ID数量: {len(stored_ids)}")

    print("数据上传完成！")


if __name__ == "__main__":
    main()
