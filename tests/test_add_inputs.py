# test_server.py
import pytest
from src.add_inputs import add_numbers, AddInput

@pytest.mark.asyncio
async def test_add_numbers():
    result = await add_numbers(AddInput(a=2.0, b=3.0))
    assert '"result": 5.0' in result

@pytest.mark.asyncio
async def test_validation_error():
    with pytest.raises(Exception):
        AddInput(a="not_a_number", b=1.0)  # should fail Pydantic validation
