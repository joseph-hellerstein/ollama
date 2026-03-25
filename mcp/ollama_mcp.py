"""
Ollama MCP Server
Exposes a local Ollama instance as MCP tools for use with Claude or any MCP client.

Install:
    pip install mcp httpx

Run (stdio transport for Claude Desktop / Claude Code):
    python ollama_mcp.py

Add to Claude Desktop config (~/.claude/claude_desktop_config.json):
    {
    "mcpServers": {
        "ollama": {
        "command": "python",
        "args": ["/path/to/ollama_mcp.py"]
        }
    } }
"""

import json
from typing import Optional, List
from contextlib import asynccontextmanager

import httpx # type: ignore
from pydantic import BaseModel, Field, ConfigDict  # type: ignore
from mcp.server.fastmcp import FastMCP, Context  # type: ignore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL   = "codellama:34b"
DEFAULT_EMBED   = "nomic-embed-text"
HTTP_TIMEOUT    = 120.0  # seconds — generation can be slow

# ---------------------------------------------------------------------------
# Shared HTTP client (reused across all tool calls)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def app_lifespan():
    """Create a persistent async HTTP client for the server lifetime."""
    async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=HTTP_TIMEOUT) as client:
        yield {"http": client}

mcp = FastMCP("ollama_mcp", lifespan=app_lifespan)  # type: ignore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _http(ctx: Context) -> httpx.AsyncClient:
    return ctx.request_context.lifespan_state["http"]  # type: ignore

def _handle_error(e: Exception) -> str:
    if isinstance(e, httpx.ConnectError):
        return "Error: Cannot connect to Ollama. Is it running? Try: ollama serve"
    if isinstance(e, httpx.TimeoutException):
        return "Error: Request timed out. The model may be loading — try again."
    if isinstance(e, httpx.HTTPStatusError):
        return f"Error: Ollama returned HTTP {e.response.status_code}: {e.response.text}"
    return f"Error: {type(e).__name__}: {e}"

# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class ChatInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    prompt: str = Field(
        ...,
        description="The user message to send to the model.",
        min_length=1,
    )
    model: str = Field(
        default=DEFAULT_MODEL,
        description=f"Ollama model name (e.g. 'llama3.2', 'mistral', 'phi3'). Default: {DEFAULT_MODEL}",
    )
    system: Optional[str] = Field(
        default=None,
        description="Optional system prompt to set the model's behaviour.",
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Sampling temperature 0–2. Lower = more focused, higher = more creative.",
        ge=0.0,
        le=2.0,
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens to generate.",
        ge=1,
        le=32768,
    )


class ConversationInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    messages: List[dict] = Field(
        ...,
        description=(
            "Full conversation history as a list of {role, content} dicts. "
            "role must be 'system', 'user', or 'assistant'."
        ),
        min_length=1,
    )
    model: str = Field(
        default=DEFAULT_MODEL,
        description=f"Ollama model name. Default: {DEFAULT_MODEL}",
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Sampling temperature 0–2.",
        ge=0.0,
        le=2.0,
    )


class EmbedInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    text: str = Field(
        ...,
        description="Text to embed.",
        min_length=1,
    )
    model: str = Field(
        default=DEFAULT_EMBED,
        description=f"Embedding model name. Default: {DEFAULT_EMBED}",
    )


class PullModelInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    model: str = Field(
        ...,
        description="Model name to pull from the Ollama registry, e.g. 'llama3.2', 'mistral:7b'.",
        min_length=1,
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

annotations={
    "title": "Chat with a local Ollama model",
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": False,
    "openWorldHint": False,
}
@mcp.tool(
    name="ollama_chat",
    annotations=annotations,  # type: ignore
)
async def ollama_chat(params: ChatInput, ctx: Context) -> str:
    """Send a single prompt to a local Ollama model and return the response.

    Useful for one-shot questions, code generation, summarisation, or any
    task where you don't need to maintain a multi-turn history.

    Args:
        params (ChatInput): Validated input containing:
            - prompt (str): The user message.
            - model (str): Ollama model to use.
            - system (Optional[str]): System prompt.
            - temperature (Optional[float]): Sampling temperature.
            - max_tokens (Optional[int]): Max tokens to generate.

    Returns:
        str: JSON with keys 'model', 'response', 'prompt_tokens',
            'completion_tokens', 'total_duration_s'.
    """
    messages = []
    if params.system:
        messages.append({"role": "system", "content": params.system})
    messages.append({"role": "user", "content": params.prompt})

    options: dict = {}
    if params.temperature is not None:
        options["temperature"] = params.temperature
    if params.max_tokens is not None:
        options["num_predict"] = params.max_tokens

    payload: dict = {"model": params.model, "messages": messages, "stream": False}
    if options:
        payload["options"] = options

    await ctx.report_progress(0.1, message=f"Sending prompt to {params.model}…")
    try:
        resp = await _http(ctx).post("/api/chat", json=payload)
        resp.raise_for_status()
    except Exception as e:
        return _handle_error(e)

    data = resp.json()
    await ctx.report_progress(1.0, message="Done")

    result = {
        "model": data.get("model"),
        "response": data["message"]["content"],
        "prompt_tokens": data.get("prompt_eval_count"),
        "completion_tokens": data.get("eval_count"),
        "total_duration_s": round(data.get("total_duration", 0) / 1e9, 2),
    }
    return json.dumps(result, indent=2)


annotations={
    "title": "Multi-turn conversation with a local Ollama model",
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": False,
    "openWorldHint": False,
}
@mcp.tool(
    name="ollama_conversation",
    annotations=annotations,  # type: ignore
)
async def ollama_conversation(params: ConversationInput, ctx: Context) -> str:
    """Send a full conversation history to Ollama and get the next assistant turn.

    Pass the complete message list (system + user + assistant turns so far).
    The tool appends the model's reply and returns the updated history.

    Args:
        params (ConversationInput): Validated input containing:
            - messages (List[dict]): Full conversation [{role, content}, …].
            - model (str): Ollama model to use.
            - temperature (Optional[float]): Sampling temperature.

    Returns:
        str: JSON with 'messages' (updated history) and 'response' (latest reply).
    """
    options: dict = {}
    if params.temperature is not None:
        options["temperature"] = params.temperature

    payload: dict = {"model": params.model, "messages": params.messages, "stream": False}
    if options:
        payload["options"] = options

    await ctx.report_progress(0.1, message=f"Calling {params.model}…")
    try:
        resp = await _http(ctx).post("/api/chat", json=payload)
        resp.raise_for_status()
    except Exception as e:
        return _handle_error(e)

    data = resp.json()
    assistant_msg = data["message"]
    updated_history = params.messages + [assistant_msg]

    await ctx.report_progress(1.0, message="Done")
    return json.dumps({"messages": updated_history, "response": assistant_msg["content"]}, indent=2)


annotations={
    "title": "Generate text embeddings with a local Ollama model",
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}
@mcp.tool(
    name="ollama_embed",
    annotations=annotations,  # type: ignore
)
async def ollama_embed(params: EmbedInput, ctx: Context) -> str:
    """Generate a vector embedding for the provided text using a local Ollama model.

    Useful for semantic search, similarity comparisons, or RAG pipelines.
    The default model is 'nomic-embed-text'; ensure it is pulled first.

    Args:
        params (EmbedInput): Validated input containing:
            - text (str): Text to embed.
            - model (str): Embedding model name.

    Returns:
        str: JSON with 'model', 'dimensions', and 'embedding' (list of floats).
    """
    try:
        resp = await _http(ctx).post("/api/embed", json={"model": params.model, "input": params.text})
        resp.raise_for_status()
    except Exception as e:
        return _handle_error(e)

    data = resp.json()
    embedding: List[float] = data["embeddings"][0]
    return json.dumps({"model": params.model, "dimensions": len(embedding), "embedding": embedding}, indent=2)


annotations={
    "title": "List locally available Ollama models",
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}
@mcp.tool(
    name="ollama_list_models",
    annotations=annotations,  # type: ignore
)
async def ollama_list_models(ctx: Context) -> str:
    """List all Ollama models currently available on this machine.

    Returns names, sizes, and modification dates. Use this to discover
    which models can be passed to ollama_chat or ollama_embed.

    Returns:
        str: JSON list of model objects with 'name', 'size_gb', 'modified_at'.
    """
    try:
        resp = await _http(ctx).get("/api/tags")
        resp.raise_for_status()
    except Exception as e:
        return _handle_error(e)

    models = [
        {
            "name": m["name"],
            "size_gb": round(m["size"] / 1e9, 2),
            "modified_at": m.get("modified_at", ""),
        }
        for m in resp.json().get("models", [])
    ]
    return json.dumps({"count": len(models), "models": models}, indent=2)


annotations={
    "title": "Pull (download) an Ollama model",
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": True,
}
@mcp.tool(
    name="ollama_pull_model",
    annotations=annotations,  # type: ignore
)
async def ollama_pull_model(params: PullModelInput, ctx: Context) -> str:
    """Download a model from the Ollama registry to the local machine.

    This may take several minutes for large models. Progress is streamed
    and summarised in the response. The model is ready for use once complete.

    Args:
        params (PullModelInput): Validated input containing:
            - model (str): Model name, e.g. 'llama3.2', 'mistral:7b'.

    Returns:
        str: JSON with 'model', 'status', and 'final_message'.
    """
    await ctx.report_progress(0.0, message=f"Pulling {params.model}…")
    try:
        async with _http(ctx).stream("POST", "/api/pull", json={"model": params.model}) as resp:
            resp.raise_for_status()
            last_status = ""
            async for line in resp.aiter_lines():
                if line:
                    chunk = json.loads(line)
                    last_status = chunk.get("status", last_status)
                    if "completed" in chunk and "total" in chunk:
                        pct = chunk["completed"] / chunk["total"]
                        await ctx.report_progress(pct, message=last_status)
    except Exception as e:
        return _handle_error(e)

    await ctx.report_progress(1.0, message="Pull complete")
    return json.dumps({"model": params.model, "status": "success", "final_message": last_status}, indent=2)


annotations={
    "title": "Show metadata for an Ollama model",
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}
@mcp.tool(
    name="ollama_model_info",
    annotations=annotations,  # type: ignore
)
async def ollama_model_info(model: str, ctx: Context) -> str:
    """Return metadata and the modelfile for a locally available Ollama model.

    Args:
        model (str): Model name exactly as listed by ollama_list_models.

    Returns:
        str: JSON with 'model', 'parameters', 'template', 'system', and 'details'.
    """
    try:
        resp = await _http(ctx).post("/api/show", json={"name": model})
        resp.raise_for_status()
    except Exception as e:
        return _handle_error(e)

    data = resp.json()
    result = {
        "model": model,
        "parameters": data.get("parameters"),
        "template": data.get("template"),
        "system": data.get("system"),
        "details": data.get("details", {}),
    }
    return json.dumps(result, indent=2)

@mcp.tool()
async def ollama_generate(model: str, prompt: str) -> str:
    """Generate a completion from a local Ollama model."""
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False})
        r.raise_for_status()
        return r.json()["response"]



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Default: stdio transport — works with Claude Desktop, Claude Code,
    # and any MCP client that spawns the server as a subprocess.
    mcp.run()

    # Uncomment for HTTP transport (e.g. to test with curl or a web client):
    # mcp.run(transport="streamable_http", port=8000)
