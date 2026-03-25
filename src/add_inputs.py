import json
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

mcp = FastMCP("example_mcp")

class AddInput(BaseModel):
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")

@mcp.tool(
    name="add_numbers",
    annotations={"readOnlyHint": True, "idempotentHint": True}
)
async def add_numbers(params: AddInput) -> str:
    """Adds two numbers together."""
    return json.dumps({"result": params.a + params.b})

if __name__ == "__main__":
    mcp.run()  # stdio transport (default)
    # or: mcp.run(transport="streamable_http", port=8000)
