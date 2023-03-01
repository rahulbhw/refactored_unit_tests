import pytest

from src.new_utils import get_foo

def test_get_foo():
    assert get_foo() == "This is Foo"