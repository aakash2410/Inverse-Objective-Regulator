from __future__ import annotations

import json
import re


def extract_json(text: str) -> dict:
    """Parse a JSON object from a model response, tolerating common wrappers.

    Models differ in how strictly they honour a "JSON only" instruction: some
    return bare JSON, others wrap it in markdown fences or add a sentence of
    preamble. This helper tries bare parsing first, then a fenced block, then the
    first balanced ``{...}`` span, before giving up.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError(f"No JSON object found in model response: {text[:200]!r}")
