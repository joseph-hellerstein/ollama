"""
Unit tests for ollama_mcp.py.

Tests are isolated from a live Ollama instance — all HTTP calls are mocked.
Run with:  python -m unittest mcp/tests/test_ollama_mcp -v
"""

import json
import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# Minimal stubs so the module can be imported without mcp/httpx installed
# ---------------------------------------------------------------------------

def _make_mcp_stub():
    """Return a minimal stub for mcp.server.fastmcp."""
    fastmcp_mod = types.ModuleType("mcp.server.fastmcp")

    class _FakeFastMCP:
        def __init__(self, *a, **kw):
            pass

        def tool(self, *a, **kw):
            def decorator(fn):
                return fn
            return decorator

        def run(self, *a, **kw):
            pass

    fastmcp_mod.FastMCP = _FakeFastMCP  # type: ignore
    fastmcp_mod.Context = object  # type: ignore
    return fastmcp_mod


def _install_stubs():
    if "mcp" not in sys.modules:
        mcp_mod = types.ModuleType("mcp")
        mcp_server = types.ModuleType("mcp.server")
        fastmcp = _make_mcp_stub()
        sys.modules["mcp"] = mcp_mod
        sys.modules["mcp.server"] = mcp_server
        sys.modules["mcp.server.fastmcp"] = fastmcp

    if "httpx" not in sys.modules:
        import httpx  # type: ignore  # real httpx — needed for exception types
        sys.modules["httpx"] = httpx

    if "pydantic" not in sys.modules:
        import pydantic  # type: ignore  # real pydantic — needed for validation in input models
        sys.modules["pydantic"] = pydantic


_install_stubs()

# Now import the module under test
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import ollama_mcp  # type: ignore  # noqa: E402  (import after path manipulation)
from ollama_mcp import (  # noqa: E402
    ChatInput,
    ConversationInput,
    EmbedInput,
    PullModelInput,
    _handle_error,
    ollama_chat,
    ollama_conversation,
    ollama_embed,
    ollama_list_models,
    ollama_model_info,
    ollama_pull_model,
)
import httpx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(http_client=None):
    """Build a minimal fake Context with an optional HTTP client."""
    ctx = MagicMock()
    ctx.report_progress = AsyncMock()
    ctx.request_context.lifespan_state = {"http": http_client or MagicMock()}
    return ctx


def _mock_response(json_data: dict, status_code: int = 200):
    """Return a MagicMock that behaves like an httpx.Response."""
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# _handle_error
# ---------------------------------------------------------------------------

class TestHandleError(unittest.TestCase):
    def test_connect_error(self):
        err = httpx.ConnectError("refused")
        self.assertIn("Cannot connect to Ollama", _handle_error(err))

    def test_timeout_error(self):
        err = httpx.TimeoutException("timeout")
        self.assertIn("timed out", _handle_error(err))

    def test_http_status_error(self):
        raw = MagicMock()
        raw.status_code = 404
        raw.text = "not found"
        err = httpx.HTTPStatusError("404", request=MagicMock(), response=raw)
        result = _handle_error(err)
        self.assertIn("404", result)
        self.assertIn("not found", result)

    def test_generic_error(self):
        result = _handle_error(ValueError("oops"))
        self.assertIn("ValueError", result)
        self.assertIn("oops", result)


# ---------------------------------------------------------------------------
# Input model validation
# ---------------------------------------------------------------------------

class TestChatInput(unittest.TestCase):
    def test_defaults(self):
        inp = ChatInput(prompt="hello")
        self.assertEqual(inp.model, ollama_mcp.DEFAULT_MODEL)
        self.assertIsNone(inp.system)
        self.assertIsNone(inp.temperature)
        self.assertIsNone(inp.max_tokens)

    def test_strips_whitespace(self):
        inp = ChatInput(prompt="  hi  ")
        self.assertEqual(inp.prompt, "hi")

    def test_empty_prompt_rejected(self):
        with self.assertRaises(Exception):
            ChatInput(prompt="")

    def test_temperature_bounds(self):
        with self.assertRaises(Exception):
            ChatInput(prompt="hi", temperature=3.0)
        with self.assertRaises(Exception):
            ChatInput(prompt="hi", temperature=-0.1)

    def test_max_tokens_bounds(self):
        with self.assertRaises(Exception):
            ChatInput(prompt="hi", max_tokens=0)
        with self.assertRaises(Exception):
            ChatInput(prompt="hi", max_tokens=99999)

    def test_extra_fields_rejected(self):
        with self.assertRaises(Exception):
            ChatInput(prompt="hi", unknown_field="x")  # type: ignore


class TestConversationInput(unittest.TestCase):
    def test_requires_messages(self):
        with self.assertRaises(Exception):
            ConversationInput(messages=[])

    def test_valid(self):
        inp = ConversationInput(messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(inp.model, ollama_mcp.DEFAULT_MODEL)


class TestEmbedInput(unittest.TestCase):
    def test_defaults(self):
        inp = EmbedInput(text="hello")
        self.assertEqual(inp.model, ollama_mcp.DEFAULT_EMBED)

    def test_empty_text_rejected(self):
        with self.assertRaises(Exception):
            EmbedInput(text="")


class TestPullModelInput(unittest.TestCase):
    def test_empty_model_rejected(self):
        with self.assertRaises(Exception):
            PullModelInput(model="")


# ---------------------------------------------------------------------------
# ollama_chat
# ---------------------------------------------------------------------------

class TestOllamaChat(unittest.IsolatedAsyncioTestCase):
    async def test_success_basic(self):
        api_resp = {
            "model": "codellama:34b",
            "message": {"content": "Hello!"},
            "prompt_eval_count": 10,
            "eval_count": 5,
            "total_duration": 2_000_000_000,
        }
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_response(api_resp))
        ctx = _make_ctx(client)

        result = json.loads(await ollama_chat(ChatInput(prompt="hi"), ctx))
        import pdb; pdb.set_trace()

        self.assertEqual(result["response"], "Hello!")
        self.assertEqual(result["prompt_tokens"], 10)
        self.assertEqual(result["completion_tokens"], 5)
        self.assertEqual(result["total_duration_s"], 2.0)

    async def test_includes_system_message(self):
        api_resp = {
            "model": "m",
            "message": {"content": "ok"},
            "total_duration": 0,
        }
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_response(api_resp))
        ctx = _make_ctx(client)

        await ollama_chat(ChatInput(prompt="hi", system="You are helpful."), ctx)

        payload = client.post.call_args.kwargs["json"]
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "You are helpful."})

    async def test_options_forwarded(self):
        api_resp = {"model": "m", "message": {"content": "ok"}, "total_duration": 0}
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_response(api_resp))
        ctx = _make_ctx(client)

        await ollama_chat(ChatInput(prompt="hi", temperature=0.5, max_tokens=128), ctx)

        payload = client.post.call_args.kwargs["json"]
        self.assertEqual(payload["options"]["temperature"], 0.5)
        self.assertEqual(payload["options"]["num_predict"], 128)

    async def test_connect_error_returned_as_string(self):
        client = AsyncMock()
        client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        ctx = _make_ctx(client)

        result = await ollama_chat(ChatInput(prompt="hi"), ctx)
        self.assertIn("Cannot connect", result)

    async def test_no_options_when_none(self):
        api_resp = {"model": "m", "message": {"content": "ok"}, "total_duration": 0}
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_response(api_resp))
        ctx = _make_ctx(client)

        await ollama_chat(ChatInput(prompt="hi"), ctx)

        payload = client.post.call_args.kwargs["json"]
        self.assertNotIn("options", payload)


# ---------------------------------------------------------------------------
# ollama_conversation
# ---------------------------------------------------------------------------

class TestOllamaConversation(unittest.IsolatedAsyncioTestCase):
    async def test_appends_assistant_turn(self):
        assistant_msg = {"role": "assistant", "content": "Sure!"}
        api_resp = {"message": assistant_msg}
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_response(api_resp))
        ctx = _make_ctx(client)

        messages = [{"role": "user", "content": "Help?"}]
        result = json.loads(
            await ollama_conversation(ConversationInput(messages=messages), ctx)
        )

        self.assertEqual(result["response"], "Sure!")
        self.assertEqual(result["messages"][-1], assistant_msg)
        self.assertEqual(len(result["messages"]), 2)

    async def test_temperature_forwarded(self):
        api_resp = {"message": {"role": "assistant", "content": "ok"}}
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_response(api_resp))
        ctx = _make_ctx(client)

        msgs = [{"role": "user", "content": "hi"}]
        await ollama_conversation(ConversationInput(messages=msgs, temperature=0.7), ctx)

        payload = client.post.call_args.kwargs["json"]
        self.assertEqual(payload["options"]["temperature"], 0.7)

    async def test_http_error_returned_as_string(self):
        client = AsyncMock()
        client.post = AsyncMock(side_effect=httpx.TimeoutException("slow"))
        ctx = _make_ctx(client)

        result = await ollama_conversation(
            ConversationInput(messages=[{"role": "user", "content": "hi"}]), ctx
        )
        self.assertIn("timed out", result)


# ---------------------------------------------------------------------------
# ollama_embed
# ---------------------------------------------------------------------------

class TestOllamaEmbed(unittest.IsolatedAsyncioTestCase):
    async def test_success(self):
        embedding = [0.1, 0.2, 0.3]
        api_resp = {"embeddings": [embedding]}
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_response(api_resp))
        ctx = _make_ctx(client)

        result = json.loads(await ollama_embed(EmbedInput(text="hello"), ctx))

        self.assertEqual(result["embedding"], embedding)
        self.assertEqual(result["dimensions"], 3)
        self.assertEqual(result["model"], ollama_mcp.DEFAULT_EMBED)

    async def test_error_returned_as_string(self):
        client = AsyncMock()
        client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        ctx = _make_ctx(client)

        result = await ollama_embed(EmbedInput(text="hello"), ctx)
        self.assertIn("Cannot connect", result)


# ---------------------------------------------------------------------------
# ollama_list_models
# ---------------------------------------------------------------------------

class TestOllamaListModels(unittest.IsolatedAsyncioTestCase):
    async def test_success(self):
        api_resp = {
            "models": [
                {"name": "llama3.2", "size": 4_200_000_000, "modified_at": "2024-01-01"},
                {"name": "mistral", "size": 7_000_000_000, "modified_at": "2024-02-01"},
            ]
        }
        client = AsyncMock()
        client.get = AsyncMock(return_value=_mock_response(api_resp))
        ctx = _make_ctx(client)

        result = json.loads(await ollama_list_models(ctx))

        self.assertEqual(result["count"], 2)
        self.assertEqual(result["models"][0]["name"], "llama3.2")
        self.assertEqual(result["models"][0]["size_gb"], round(4_200_000_000 / 1e9, 2))

    async def test_empty_models(self):
        client = AsyncMock()
        client.get = AsyncMock(return_value=_mock_response({"models": []}))
        ctx = _make_ctx(client)

        result = json.loads(await ollama_list_models(ctx))
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["models"], [])

    async def test_error_returned_as_string(self):
        client = AsyncMock()
        client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        ctx = _make_ctx(client)

        result = await ollama_list_models(ctx)
        self.assertIn("Cannot connect", result)


# ---------------------------------------------------------------------------
# ollama_pull_model
# ---------------------------------------------------------------------------

class TestOllamaPullModel(unittest.IsolatedAsyncioTestCase):
    async def test_success(self):
        lines = [
            json.dumps({"status": "pulling manifest"}),
            json.dumps({"status": "downloading", "completed": 50, "total": 100}),
            json.dumps({"status": "success"}),
        ]

        async def _aiter_lines():
            for line in lines:
                yield line

        resp_cm = MagicMock()
        resp_cm.__aenter__ = AsyncMock(return_value=resp_cm)
        resp_cm.__aexit__ = AsyncMock(return_value=False)
        resp_cm.raise_for_status = MagicMock()
        resp_cm.aiter_lines = _aiter_lines

        client = AsyncMock()
        client.stream = MagicMock(return_value=resp_cm)
        ctx = _make_ctx(client)

        result = json.loads(
            await ollama_pull_model(PullModelInput(model="llama3.2"), ctx)
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["model"], "llama3.2")
        self.assertEqual(result["final_message"], "success")

    async def test_progress_reported(self):
        lines = [
            json.dumps({"status": "downloading", "completed": 25, "total": 100}),
        ]

        async def _aiter_lines():
            for line in lines:
                yield line

        resp_cm = MagicMock()
        resp_cm.__aenter__ = AsyncMock(return_value=resp_cm)
        resp_cm.__aexit__ = AsyncMock(return_value=False)
        resp_cm.raise_for_status = MagicMock()
        resp_cm.aiter_lines = _aiter_lines

        client = AsyncMock()
        client.stream = MagicMock(return_value=resp_cm)
        ctx = _make_ctx(client)

        await ollama_pull_model(PullModelInput(model="llama3.2"), ctx)

        progress_calls = [
            call.args[0] for call in ctx.report_progress.call_args_list
        ]
        self.assertIn(0.25, progress_calls)

    async def test_error_returned_as_string(self):
        resp_cm = MagicMock()
        resp_cm.__aenter__ = AsyncMock(side_effect=httpx.ConnectError("refused"))
        resp_cm.__aexit__ = AsyncMock(return_value=False)

        client = AsyncMock()
        client.stream = MagicMock(return_value=resp_cm)
        ctx = _make_ctx(client)

        result = await ollama_pull_model(PullModelInput(model="llama3.2"), ctx)
        self.assertIn("Cannot connect", result)


# ---------------------------------------------------------------------------
# ollama_model_info
# ---------------------------------------------------------------------------

class TestOllamaModelInfo(unittest.IsolatedAsyncioTestCase):
    async def test_success(self):
        api_resp = {
            "parameters": "7B",
            "template": "{{ .System }}",
            "system": "You are helpful.",
            "details": {"family": "llama"},
        }
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_response(api_resp))
        ctx = _make_ctx(client)

        result = json.loads(await ollama_model_info("llama3.2", ctx))

        self.assertEqual(result["model"], "llama3.2")
        self.assertEqual(result["parameters"], "7B")
        self.assertEqual(result["details"], {"family": "llama"})

    async def test_missing_fields_default_none(self):
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_response({}))
        ctx = _make_ctx(client)

        result = json.loads(await ollama_model_info("llama3.2", ctx))

        self.assertIsNone(result["parameters"])
        self.assertIsNone(result["template"])
        self.assertIsNone(result["system"])
        self.assertEqual(result["details"], {})

    async def test_error_returned_as_string(self):
        client = AsyncMock()
        client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        ctx = _make_ctx(client)

        result = await ollama_model_info("llama3.2", ctx)
        self.assertIn("Cannot connect", result)


if __name__ == "__main__":
    unittest.main()
