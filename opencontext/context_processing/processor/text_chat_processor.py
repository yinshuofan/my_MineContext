import json
import datetime
import asyncio
from typing import List, Any, Dict

from opencontext.context_processing.processor.base_processor import BaseContextProcessor
from opencontext.models.context import RawContextProperties, ProcessedContext, ExtractedData, ContextProperties, Vectorize
from opencontext.models.enums import ContentFormat, ContextSource, get_context_type_for_analysis
from opencontext.llm.global_vlm_client import generate_with_messages_async
from opencontext.utils.json_parser import parse_json_from_response
from opencontext.config.global_config import get_prompt_group
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

class TextChatProcessor(BaseContextProcessor):
    def __init__(self):
        super().__init__({}) 

    def get_name(self) -> str:
        return "text_chat_processor"

    def get_description(self) -> str:
        return "Processes text chat history to extract structured context information."

    def can_process(self, context: RawContextProperties) -> bool:
        return (isinstance(context, RawContextProperties) and 
                context.source == ContextSource.INPUT and 
                context.content_format == ContentFormat.TEXT)

    def process(self, context: RawContextProperties) -> bool:
        """
        同步入口方法。
        智能检测当前运行环境，兼容同步和异步调用。
        """
        try:
            try:
                # 尝试获取当前正在运行的事件循环
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # 如果已经在异步循环中（比如在 simple_bot.py 运行时），
                # 我们不能使用 asyncio.run()，否则会报错。
                # 应该创建一个后台任务来执行处理。
                loop.create_task(self._process_and_callback(context))
            else:
                # 如果没有运行的循环（比如在单独的后台进程中），
                # 使用 asyncio.run() 启动一个新的循环。
                asyncio.run(self._process_and_callback(context))
            
            return True
        except Exception as e:
            logger.error(f"Failed to process chat context: {e}")
            return False

    async def _process_and_callback(self, context: RawContextProperties):
        """执行异步处理并调用回调"""
        try:
            processed_list = await self._process_async(context)
            if processed_list and self._callback:
                self._callback(processed_list)
        except Exception as e:
            logger.error(f"Error in async processing task: {e}")

    async def _process_async(self, raw_context: RawContextProperties) -> List[ProcessedContext]:
        # 1. 获取 Prompt
        prompt_group = get_prompt_group("processing.extraction.chat_analyze")
        if not prompt_group:
            logger.warning("Prompt 'chat_analyze' not found, using default fallback.")
            return []

        # 2. 准备 LLM 输入
        chat_history_str = raw_context.content_text
        
        messages = [
            {"role": "system", "content": prompt_group.get("system", "")},
            {"role": "user", "content": prompt_group.get("user", "").format(
                chat_history=chat_history_str,
                current_time=datetime.datetime.now().isoformat()
            )}
        ]

        # 3. 调用 LLM
        response = await generate_with_messages_async(messages)
        
        # 4. 解析结果
        analysis = parse_json_from_response(response)
        if not analysis:
            return []

        # 5. 构建 ProcessedContext
        context_type = get_context_type_for_analysis(analysis.get("context_type", "activity_context"))
        
        extracted_data = ExtractedData(
            title=analysis.get("title", "Chat Summary"),
            summary=analysis.get("summary", ""),
            keywords=analysis.get("keywords", []),
            entities=analysis.get("entities", []),
            context_type=context_type,
            importance=analysis.get("importance", 0),
            confidence=analysis.get("confidence", 0)
        )

        processed_context = ProcessedContext(
            properties=ContextProperties(
                raw_properties=[raw_context],
                create_time=raw_context.create_time,
                update_time=datetime.datetime.now(),
                event_time=raw_context.create_time,
                is_processed=True,
                enable_merge=True,
            ),
            extracted_data=extracted_data,
            vectorize=Vectorize(
                content_format=ContentFormat.TEXT,
                text=f"{extracted_data.title}\n{extracted_data.summary}\n{' '.join(extracted_data.keywords)}"
            )
        )

        return [processed_context]