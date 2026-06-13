import pytest

from ior.json_utils import extract_json


def test_bare_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_fenced_no_lang():
    assert extract_json("```\n{\"a\": 1}\n```") == {"a": 1}


def test_preamble_and_trailing_text():
    assert extract_json('Sure! {"a": 1} hope that helps') == {"a": 1}


def test_nested_object_span():
    assert extract_json('noise {"a": {"b": 2}} more') == {"a": {"b": 2}}


def test_raises_when_no_json():
    with pytest.raises(ValueError):
        extract_json("there is no json here")
