from fde_kit.builtin_tools import registry as builtin
from fde_kit.tools import ToolRegistry


def test_schema_from_type_hints():
    reg = ToolRegistry()

    @reg.tool
    def greet(name: str, excited: bool = False) -> str:
        """Say hello."""
        return f"hi {name}{'!' if excited else ''}"

    schema = reg.api_schemas()[0]
    assert schema["name"] == "greet"
    assert schema["description"] == "Say hello."
    props = schema["input_schema"]["properties"]
    assert props["name"]["type"] == "string"
    assert props["excited"]["type"] == "boolean"
    assert schema["input_schema"]["required"] == ["name"]


def test_execute_validates_input():
    reg = ToolRegistry()

    @reg.tool
    def square(x: float) -> float:
        """Square a number."""
        return x * x

    assert reg.execute("square", {"x": 3}).content == "9.0"
    bad = reg.execute("square", {"x": "not a number"})
    assert bad.is_error and "Invalid input" in bad.content


def test_execute_unknown_tool_and_exception():
    reg = ToolRegistry()

    @reg.tool
    def boom() -> str:
        """Always fails."""
        raise RuntimeError("kaboom")

    unknown = reg.execute("nope", {})
    assert unknown.is_error and "Unknown tool" in unknown.content
    err = reg.execute("boom", {})
    assert err.is_error and "kaboom" in err.content


def test_duplicate_registration_rejected():
    reg = ToolRegistry()

    @reg.tool
    def a() -> str:
        """a"""
        return "a"

    try:
        reg.tool(name="a")(lambda: "b")
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_builtin_calculator():
    assert builtin.execute("calculator", {"expression": "17 * 23"}).content == "391"
    assert builtin.execute("calculator", {"expression": "1284 / 6"}).content == "214"
    assert builtin.execute("calculator", {"expression": "2 ** 16"}).content == "65536"
    unsafe = builtin.execute("calculator", {"expression": "__import__('os')"})
    assert unsafe.is_error


def test_builtin_lookup_policy():
    assert "24 months" in builtin.execute("lookup_policy", {"topic": "warranty"}).content
    assert "24 months" in builtin.execute("lookup_policy", {"topic": "Warranty period"}).content
    miss = builtin.execute("lookup_policy", {"topic": "pets"})
    assert not miss.is_error and "No policy found" in miss.content
