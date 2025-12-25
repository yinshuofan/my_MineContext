import os
import sys
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import openai
from openai.types.chat import ChatCompletionMessageToolCall, ChatCompletionChunk

from opencontext.server.opencontext import OpenContext
from opencontext.tools.tool_definitions import ALL_TOOL_DEFINITIONS
from opencontext.tools.tools_executor import ToolsExecutor

from opencontext.utils.logging_utils import setup_logging, get_logger
import opencontext
print(f"\n🔍 [Debug] 当前加载的 OpenContext 路径: {opencontext.__file__}\n")


setup_logging({
    "level": "DEBUG",
    "log_path": "logs/minecontext.log"
})

logger = get_logger(__name__)

# 异步输入函数
async def async_input(prompt: str = "") -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)

# --- 1. 初始化 MineContext 完整系统 ---
def init_minecontext():
    oc = OpenContext()
    oc.initialize()
    oc.start_capture()
    return oc

# --- 2. 工具执行器 ---
async def execute_tool(tool_call: ChatCompletionMessageToolCall) -> Dict[str, Any]:
    name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    print(f"\n[Tool] 正在调用工具: {name} 参数: {arguments}")
    executor = ToolsExecutor()

    try:
        print(f"\n[Tool] 即将执行工具: {name} 参数: {arguments}")
        results = await executor.batch_run_tools_async([tool_call])
        result = results[0]
        print(f"\n[Tool] 工具 {name} 执行结果: {result}")
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

# --- 3. 主聊天逻辑 ---
async def chat_loop():
    client = openai.AsyncOpenAI(
        api_key="cd8b23c5-45f1-48a8-9009-e1ba7f592cfe",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
    
    model_name = "doubao-seed-1-6-251015"
    
    oc = init_minecontext()
    
    chat_capture = oc.capture_manager.get_component("text_chat")
    
    messages = [
        {"role": "system", "content": "你是一个拥有长期记忆的智能助手。你可以使用工具检索过去的对话和活动。"},
        {"role": "system", "content": f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}
    ]

    print("=== MineContext 聊天机器人 (输入 'quit' 退出) ===")

    while True:
        try:
            user_input = await async_input("\nUser: ")
        except EOFError:
            break

        if user_input.lower() in ["quit", "exit"]:
            print("\n🛑 正在停止并保存剩余记忆...")
            oc.shutdown(graceful=True)
            break

        messages.append({"role": "user", "content": user_input})
        
        chat_capture.push_message("user", user_input, user_id="user_123", device_id="device_123", agent_id="agent_123")

        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            reasoning_effort="minimal",
            tools=ALL_TOOL_DEFINITIONS,
            tool_choice="auto",
        )

        print("Assistant: ", end="", flush=True)
        
        collected_content = ""
        tool_calls_buffer = []
        current_tool_call = None

        async for chunk in response:
            delta = chunk.choices[0].delta
            
            if delta.content:
                print(delta.content, end="", flush=True)
                collected_content += delta.content
            
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

        if tool_calls_buffer:
            assistant_msg = {
                "role": "assistant",
                "content": collected_content if collected_content else None,
                "tool_calls": tool_calls_buffer
            }
            messages.append(assistant_msg)
            
            for tc_data in tool_calls_buffer:
                class MockToolCall:
                    id = tc_data["id"]
                    class Function:
                        name = tc_data["function"]["name"]
                        arguments = tc_data["function"]["arguments"]
                    function = Function()
                
                tool_result_msg = await execute_tool(MockToolCall())
                messages.append(tool_result_msg)

            response_2 = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True,
                reasoning_effort="minimal",
                tools=ALL_TOOL_DEFINITIONS
            )
            
            collected_content = ""
            async for chunk in response_2:
                delta = chunk.choices[0].delta
            
                if delta.content:
                    print(delta.content, end="", flush=True)
                    collected_content += delta.content

        print()

        messages.append({"role": "assistant", "content": collected_content})
        
        chat_capture.push_message("assistant", collected_content, user_id="user_123", device_id="device_123", agent_id="agent_123")

if __name__ == "__main__":
    asyncio.run(chat_loop())
