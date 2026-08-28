from __future__ import annotations

import json

import httpx
import pytest
import respx

from llm_client import LLMError, OllamaClient

BASE_URL = "http://localhost:11434"


@pytest.mark.asyncio
@respx.mock
async def test_chat_sends_tools_in_payload():
    route = respx.post(f"{BASE_URL}/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "hi", "tool_calls": []}},
        )
    )
    client = OllamaClient(base_url=BASE_URL, model="qwen2.5:7b")

    tools = [{"type": "function", "function": {"name": "exec", "parameters": {}}}]
    result = await client.chat([{"role": "user", "content": "hi"}], tools=tools)

    assert result["content"] == "hi"
    assert result["tool_calls"] == []
    sent_body = json.loads(route.calls[0].request.content)
    assert sent_body["tools"] == tools


@pytest.mark.asyncio
@respx.mock
async def test_chat_parses_string_encoded_arguments():
    respx.post(f"{BASE_URL}/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "exec",
                                "arguments": '{"command": "smctemp -c"}',
                            }
                        }
                    ],
                }
            },
        )
    )
    client = OllamaClient(base_url=BASE_URL, model="qwen2.5:7b")

    result = await client.chat([{"role": "user", "content": "temp?"}])

    assert result["tool_calls"][0]["function"]["arguments"] == {"command": "smctemp -c"}


@pytest.mark.asyncio
@respx.mock
async def test_chat_passes_through_dict_arguments():
    respx.post(f"{BASE_URL}/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "exec", "arguments": {"command": "vm_stat"}}}
                    ],
                }
            },
        )
    )
    client = OllamaClient(base_url=BASE_URL, model="qwen2.5:7b")

    result = await client.chat([{"role": "user", "content": "mem?"}])

    assert result["tool_calls"][0]["function"]["arguments"] == {"command": "vm_stat"}


@pytest.mark.asyncio
@respx.mock
async def test_chat_raises_llm_error_on_connection_failure():
    respx.post(f"{BASE_URL}/api/chat").mock(side_effect=httpx.ConnectError("refused"))
    client = OllamaClient(base_url=BASE_URL, model="qwen2.5:7b")

    with pytest.raises(LLMError):
        await client.chat([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
@respx.mock
async def test_chat_raises_llm_error_on_bad_status():
    respx.post(f"{BASE_URL}/api/chat").mock(return_value=httpx.Response(500, text="boom"))
    client = OllamaClient(base_url=BASE_URL, model="qwen2.5:7b")

    with pytest.raises(LLMError):
        await client.chat([{"role": "user", "content": "hi"}])
