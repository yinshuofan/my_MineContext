import os
import sys
import json
import asyncio
from typing import List, Dict, Any

from opencontext.storage.global_storage import get_storage
from opencontext.config.global_config import GlobalConfig

# 假设代码在项目根目录，添加路径以便导入 opencontext
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import openai
from openai.types.chat import ChatCompletionMessageToolCall, ChatCompletionChunk

# 导入 MineContext 的核心组件
# 注意：你需要确保上一轮生成的 text_chat_capture.py 和 text_chat_processor.py 已经存在
from opencontext.context_capture.text_chat_capture import TextChatCapture
from opencontext.server.component_initializer import ComponentInitializer
from opencontext.managers.capture_manager import ContextCaptureManager
from opencontext.tools.tool_definitions import ALL_TOOL_DEFINITIONS

# 导入具体的工具类以便执行（这里仅列举部分核心工具作为示例）
# 实际运行时建议使用 ToolsExecutor 或构建一个映射表
from opencontext.tools.retrieval_tools import (
    ActivityContextTool, 
    SemanticContextTool, 
    IntentContextTool,
    GetTodosTool
)

# --- 1. 初始化 MineContext 记忆模块 ---
def init_memory_module():
    # 1. 务必先获取全局配置和存储实例
    GlobalConfig.get_instance()
    storage = get_storage()  # <---【关键修复】：这里需要获取存储实例
    
    if not storage:
        print("[System] 警告：存储模块初始化失败！")

    capture_manager = ContextCaptureManager()
    
    # 初始化聊天捕获组件
    chat_capture = TextChatCapture()
    chat_capture.initialize({"buffer_size": 4}) 
    chat_capture.start()
    
    capture_manager.register_component("text_chat", chat_capture)
    
    from opencontext.context_processing.processor.text_chat_processor import TextChatProcessor
    processor = TextChatProcessor()
    
    # 定义处理完成后的回调
    def on_processed(contexts):
        # 现在这里的 storage 引用的是上面获取到的实例
        if contexts and storage:
            try:
                # 真正写入向量数据库
                doc_ids = storage.batch_upsert_processed_context(contexts)
                print(f"\n[System] 🧠 记忆总结完成，已持久化 {len(contexts)} 条记录。IDs: {doc_ids}")
            except Exception as e:
                print(f"\n[System] ❌ 记忆存储失败: {e}")
        else:
            print("\n[System] 处理完成，但没有内容需要存储或存储模块未就绪。")
        
    processor.set_callback(on_processed)
    
    # 将 Capture 的输出连接到 Processor
    chat_capture.set_callback(lambda ctxs: [processor.process(c) for c in ctxs])
    
    return chat_capture

# --- 2. 工具执行器 (简单的工具分发逻辑) ---
async def execute_tool(tool_call: ChatCompletionMessageToolCall) -> Dict[str, Any]:
    name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    print(f"\n[Tool] 正在调用工具: {name} 参数: {arguments}")

    # 简单的工具映射表
    tool_map = {
        "retrieve_activity_context": ActivityContextTool,
        "retrieve_semantic_context": SemanticContextTool,
        "retrieve_intent_context": IntentContextTool,
        "get_todos": GetTodosTool,
        # 添加更多工具...
    }

    if name in tool_map:
        tool_instance = tool_map[name]()
        # 假设工具都有 run 方法，根据 MineContext 的 BaseTool 定义
        # 大多数工具的参数是 query 或 filters
        try:
            result = tool_instance.run(**arguments)
            return {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": name,
                "content": str(result)
            }
        except Exception as e:
            return {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": name,
                "content": f"Error: {str(e)}"
            }
    else:
        return {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": name,
            "content": "Error: Tool not found"
        }

# --- 3. 主聊天逻辑 ---
async def chat_loop():
    # 配置 LLM
    client = openai.AsyncOpenAI(
        api_key="cd8b23c5-45f1-48a8-9009-e1ba7f592cfe",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
    
    model_name = "doubao-seed-1-6-251015"
    
    # 初始化记忆捕获
    chat_capture = init_memory_module()
    
    messages = [
        {"role": "system", "content": "你是一个拥有长期记忆的智能助手。你可以使用工具检索过去的对话和活动。"}
    ]

    print("=== MineContext 聊天机器人 (输入 'quit' 退出) ===")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        # 1. 记录用户消息到短期上下文
        messages.append({"role": "user", "content": user_input})
        
        # 2. 推送消息到 MineContext 长期记忆捕获模块
        chat_capture.push_message("user", user_input)

        # 3. 请求 LLM
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            reasoning_effort="minimal", # 如果模型支持
            tools=ALL_TOOL_DEFINITIONS, # 注入 MineContext 的所有工具定义
            tool_choice="auto",
        )

        # 4. 处理流式响应
        print("Assistant: ", end="", flush=True)
        
        collected_content = ""
        tool_calls_buffer = []
        current_tool_call = None

        async for chunk in response:
            delta = chunk.choices[0].delta
            
            # A. 处理文本内容
            if delta.content:
                print(delta.content, end="", flush=True)
                collected_content += delta.content
            
            # B. 处理工具调用 (流式工具调用需要拼接)
            if delta.tool_calls:
                for tc_chunk in delta.tool_calls:
                    if len(tool_calls_buffer) <= tc_chunk.index:
                        tool_calls_buffer.append({
                            "id": "", "type": "function", "function": {"name": "", "arguments": ""}
                        })
                    
                    tc = tool_calls_buffer[tc_chunk.index]
                    if tc_chunk.id: tc["id"] += tc_chunk.id
                    if tc_chunk.function.name: tc["function"]["name"] += tc_chunk.function.name
                    if tc_chunk.function.arguments: tc["function"]["arguments"] += tc_chunk.function.arguments

        # 5. 如果有工具调用，执行并进行第二轮对话
        if tool_calls_buffer:
            # 添加 Assistant 的 tool_calls 消息到历史
            assistant_msg = {
                "role": "assistant",
                "content": collected_content if collected_content else None,
                "tool_calls": tool_calls_buffer
            }
            messages.append(assistant_msg)
            
            # 执行所有工具
            for tc_data in tool_calls_buffer:
                # 构造临时的 ToolCall 对象以便复用 execute_tool 函数
                class MockToolCall:
                    id = tc_data["id"]
                    class Function:
                        name = tc_data["function"]["name"]
                        arguments = tc_data["function"]["arguments"]
                    function = Function()
                
                tool_result_msg = await execute_tool(MockToolCall())
                messages.append(tool_result_msg) # 添加 Tool 结果消息

            # 带上工具结果再次请求 LLM
            # 注意：这里不需要再记录一次 push_message，因为这属于思考过程
            response_2 = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True,
                tools=ALL_TOOL_DEFINITIONS
            )
            
            # 输出第二轮结果
            collected_content = "" # 重置内容
            async for chunk in response_2:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
                    collected_content += content

        print() # 换行

        # 6. 记录助手消息到上下文
        messages.append({"role": "assistant", "content": collected_content})
        
        # 7. 推送助手回复到 MineContext 长期记忆捕获模块
        chat_capture.push_message("assistant", collected_content)

if __name__ == "__main__":
    asyncio.run(chat_loop())