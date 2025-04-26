import pytest

def test_example_functionality():
    assert True

def test_example_fixture(fixture_name):
    assert fixture_name is not None

@pytest.fixture
def fixture_name():
    return "example"