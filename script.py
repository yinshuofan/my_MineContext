import os
import sys
import json
import asyncio
import httpx
from datetime import datetime
from typing import List, Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import openai
from openai.types.chat import ChatCompletionMessageToolCall, ChatCompletionChunk

from opencontext.tools.tool_definitions import ALL_TOOL_DEFINITIONS
from opencontext.tools.tools_executor import ToolsExecutor

from opencontext.utils.logging_utils import setup_logging, get_logger

setup_logging({
    "level": "DEBUG",
    "log_path": "logs/minecontext.log"
})

logger = get_logger(__name__)


# ============================================================================
# Push API Client - 用于与 MineContext 后端服务通信
# ============================================================================

class MineContextClient:
    """
    MineContext Push API 客户端
    通过 HTTP API 与后端服务通信，推送聊天消息和其他上下文数据
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:1733",
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ):
        """
        初始化客户端
        
        Args:
            base_url: MineContext 后端服务地址
            api_key: API 认证密钥（如果启用了认证）
            user_id: 用户标识符
            device_id: 设备标识符
            agent_id: Agent标识符
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.user_id = user_id
        self.device_id = device_id
        self.agent_id = agent_id
        
        # 构建请求头
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        
        # 创建异步 HTTP 客户端
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """获取或创建 HTTP 客户端"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=30.0
            )
        return self._client
    
    async def close(self):
        """关闭 HTTP 客户端"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    async def push_chat_message(
        self,
        role: str,
        content: str,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        推送单条聊天消息
        
        Args:
            role: 消息角色 (user/assistant/system)
            content: 消息内容
            user_id: 用户标识符（覆盖默认值）
            device_id: 设备标识符（覆盖默认值）
            agent_id: Agent标识符（覆盖默认值）
            metadata: 额外元数据
        
        Returns:
            API 响应
        """
        client = await self._get_client()
        
        payload = {
            "role": role,
            "content": content,
            "user_id": user_id or self.user_id,
            "device_id": device_id or self.device_id,
            "agent_id": agent_id or self.agent_id,
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            payload["metadata"] = metadata
        
        try:
            response = await client.post("/api/push/chat/message", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Push chat message failed: {e.response.status_code} - {e.response.text}")
            return {"code": e.response.status_code, "message": str(e)}
        except Exception as e:
            logger.error(f"Push chat message error: {e}")
            return {"code": 500, "message": str(e)}
    
    async def push_chat_messages(
        self,
        messages: List[Dict[str, str]],
        flush_immediately: bool = False,
    ) -> Dict[str, Any]:
        """
        批量推送聊天消息
        
        Args:
            messages: 消息列表，每条消息包含 role 和 content
            flush_immediately: 是否立即刷新缓冲区
        
        Returns:
            API 响应
        """
        client = await self._get_client()
        
        payload = {
            "messages": [
                {
                    "role": msg["role"],
                    "content": msg["content"],
                    "user_id": self.user_id,
                    "device_id": self.device_id,
                    "agent_id": self.agent_id,
                }
                for msg in messages
            ],
            "user_id": self.user_id,
            "device_id": self.device_id,
            "agent_id": self.agent_id,
            "flush_immediately": flush_immediately,
        }
        
        try:
            response = await client.post("/api/push/chat/messages", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Push chat messages failed: {e.response.status_code} - {e.response.text}")
            return {"code": e.response.status_code, "message": str(e)}
        except Exception as e:
            logger.error(f"Push chat messages error: {e}")
            return {"code": 500, "message": str(e)}
    
    async def flush_chat_buffer(self) -> Dict[str, Any]:
        """
        手动刷新聊天缓冲区
        
        Returns:
            API 响应
        """
        client = await self._get_client()
        
        payload = {
            "user_id": self.user_id,
            "device_id": self.device_id,
            "agent_id": self.agent_id,
        }
        
        try:
            response = await client.post("/api/push/chat/flush", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Flush chat buffer failed: {e.response.status_code} - {e.response.text}")
            return {"code": e.response.status_code, "message": str(e)}
        except Exception as e:
            logger.error(f"Flush chat buffer error: {e}")
            return {"code": 500, "message": str(e)}
    
    async def push_activity(
        self,
        title: str,
        content: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        resources: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        推送活动记录
        
        Args:
            title: 活动标题
            content: 活动内容/描述
            start_time: 开始时间 (ISO format)
            end_time: 结束时间 (ISO format)
            resources: 相关资源路径/URL列表
            metadata: 额外元数据
        
        Returns:
            API 响应
        """
        client = await self._get_client()
        
        payload = {
            "title": title,
            "content": content,
            "user_id": self.user_id,
            "device_id": self.device_id,
        }
        if start_time:
            payload["start_time"] = start_time
        if end_time:
            payload["end_time"] = end_time
        if resources:
            payload["resources"] = resources
        if metadata:
            payload["metadata"] = metadata
        
        try:
            response = await client.post("/api/push/activity", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Push activity failed: {e.response.status_code} - {e.response.text}")
            return {"code": e.response.status_code, "message": str(e)}
        except Exception as e:
            logger.error(f"Push activity error: {e}")
            return {"code": 500, "message": str(e)}


# ============================================================================
# 配置
# ============================================================================

# MineContext 后端服务配置
MINECONTEXT_BASE_URL = os.getenv("MINECONTEXT_BASE_URL", "http://localhost:1733")
MINECONTEXT_API_KEY = os.getenv("MINECONTEXT_API_KEY", None)

# 用户标识配置
USER_ID = os.getenv("USER_ID", "user_321")
DEVICE_ID = os.getenv("DEVICE_ID", "device_321")
AGENT_ID = os.getenv("AGENT_ID", "agent_321")

# LLM 配置
LLM_API_KEY = os.getenv("LLM_API_KEY", "cd8b23c5-45f1-48a8-9009-e1ba7f592cfe")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
LLM_MODEL = os.getenv("LLM_MODEL", "doubao-seed-1-6-251015")


# ============================================================================
# 工具函数
# ============================================================================

async def async_input(prompt: str = "") -> str:
    """异步输入函数"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)


async def execute_tool(tool_call: ChatCompletionMessageToolCall) -> Dict[str, Any]:
    """执行工具调用"""
    name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    print(f"\n[Tool] 正在调用工具: {name} 参数: {arguments}")
    executor = ToolsExecutor()
    import time
    try:
        print(f"\n[Tool] 即将执行工具: {name} 参数: {arguments}")
        start_time = time.time()
        results = await executor.batch_run_tools_async([tool_call])
        result = results[0]
        end_time = time.time()
        print(f"\n[Tool] 工具 {name} 执行结果: {result} 耗时: {end_time - start_time:.4f}秒")
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


# ============================================================================
# 主聊天逻辑
# ============================================================================

async def chat_loop():
    """主聊天循环"""
    
    # 初始化 LLM 客户端
    client = openai.AsyncOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
    )
    
    # 初始化 MineContext 客户端（通过 HTTP API）
    mc_client = MineContextClient(
        base_url=MINECONTEXT_BASE_URL,
        api_key=MINECONTEXT_API_KEY,
        user_id=USER_ID,
        device_id=DEVICE_ID,
        agent_id=AGENT_ID,
    )
    
    print(f"\n🔗 MineContext 后端服务: {MINECONTEXT_BASE_URL}")
    print(f"👤 用户标识: user_id={USER_ID}, device_id={DEVICE_ID}, agent_id={AGENT_ID}")
    
    messages = [
        {"role": "system", "content": "你是一个拥有长期记忆的智能助手。你可以使用工具检索过去的对话和活动。"},
        {"role": "system", "content": f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}
    ]

    print("\n=== MineContext 聊天机器人 (输入 'quit' 退出) ===")

    try:
        while True:
            try:
                user_input = await async_input("\nUser: ")
            except EOFError:
                break

            if user_input.lower() in ["quit", "exit"]:
                print("\n🛑 正在停止并保存剩余记忆...")
                # 刷新聊天缓冲区，确保所有消息都被保存
                await mc_client.flush_chat_buffer()
                break

            messages.append({"role": "user", "content": user_input})
            
            # 通过 HTTP API 推送用户消息
            await mc_client.push_chat_message("user", user_input, user_id=USER_ID, device_id=DEVICE_ID, agent_id=AGENT_ID)

            response = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                stream=True,
                reasoning_effort="minimal",
                # tools=ALL_TOOL_DEFINITIONS,
                # tool_choice="auto",
            )

            print("Assistant: ", end="", flush=True)
            
            collected_content = ""
            tool_calls_buffer = []

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
                    model=LLM_MODEL,
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
            
            # 通过 HTTP API 推送助手回复
            await mc_client.push_chat_message("assistant", collected_content, user_id=USER_ID, device_id=DEVICE_ID, agent_id=AGENT_ID)
    
    finally:
        # 确保关闭 HTTP 客户端
        await mc_client.close()


if __name__ == "__main__":
    asyncio.run(chat_loop())
