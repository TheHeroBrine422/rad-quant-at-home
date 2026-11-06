# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""A strict subset of JSON Schema, enough for job parameters (stdlib only, so the worker image needs nothing extra).

Supported: type (one or a list of object / array / string / integer / number / boolean / null), properties,
required, additionalProperties (default **false**: unknown keys are errors), enum, const, minimum, maximum,
multipleOf, pattern (full match), minLength / maxLength, items, minItems / maxItems. Booleans are never numbers.
Any other keyword in a schema is a programming error (raises), so a schema cannot silently mean less than it says."""
import math, re

_KNOWN = {"type", "properties", "required", "additionalProperties", "enum", "const", "minimum", "maximum",
          "multipleOf", "pattern", "minLength", "maxLength", "items", "minItems", "maxItems", "description"}


class SchemaError(ValueError):
    pass


def _type_ok(v, t):
    if t == "object": return isinstance(v, dict)
    if t == "array": return isinstance(v, list)
    if t == "string": return isinstance(v, str)
    if t == "boolean": return isinstance(v, bool)
    if t == "null": return v is None
    if t == "integer": return isinstance(v, int) and not isinstance(v, bool)
    if t == "number": return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    raise SchemaError(f"schema: unknown type {t!r}")


def validate(v, s, path="params"):
    """raises SchemaError (with the path of the first problem) unless v matches schema s"""
    bad = set(s) - _KNOWN
    if bad: raise SchemaError(f"schema at {path}: unsupported keywords {sorted(bad)}")
    if "type" in s:
        ts = s["type"] if isinstance(s["type"], list) else [s["type"]]
        if not any(_type_ok(v, t) for t in ts): raise SchemaError(f"{path}: expected {'/'.join(ts)}, got {type(v).__name__}")
    if "const" in s and v != s["const"]: raise SchemaError(f"{path}: must be {s['const']!r}")
    if "enum" in s and not any(v == e and type(v) is type(e) for e in s["enum"]):
        raise SchemaError(f"{path}: {v!r} not in {s['enum']}")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if "minimum" in s and v < s["minimum"]: raise SchemaError(f"{path}: {v} < {s['minimum']}")
        if "maximum" in s and v > s["maximum"]: raise SchemaError(f"{path}: {v} > {s['maximum']}")
        if "multipleOf" in s:
            q = v / s["multipleOf"]
            if abs(q - round(q)) > 1e-9: raise SchemaError(f"{path}: {v} is not a multiple of {s['multipleOf']}")
    if isinstance(v, str):
        if "minLength" in s and len(v) < s["minLength"]: raise SchemaError(f"{path}: too short")
        if "maxLength" in s and len(v) > s["maxLength"]: raise SchemaError(f"{path}: too long")
        if "pattern" in s and not re.fullmatch(s["pattern"], v): raise SchemaError(f"{path}: {v!r} does not match {s['pattern']}")
    if isinstance(v, list):
        if "minItems" in s and len(v) < s["minItems"]: raise SchemaError(f"{path}: fewer than {s['minItems']} items")
        if "maxItems" in s and len(v) > s["maxItems"]: raise SchemaError(f"{path}: more than {s['maxItems']} items")
        if "items" in s:
            for i, x in enumerate(v): validate(x, s["items"], f"{path}[{i}]")
    if isinstance(v, dict):
        props = s.get("properties", {})
        for k in s.get("required", []):
            if k not in v: raise SchemaError(f"{path}: missing {k!r}")
        extra = s.get("additionalProperties", False)
        for k, x in v.items():
            if not isinstance(k, str): raise SchemaError(f"{path}: non-string key")
            if k in props: validate(x, props[k], f"{path}.{k}")
            elif extra is False: raise SchemaError(f"{path}: unknown key {k!r}")
            elif isinstance(extra, dict): validate(x, extra, f"{path}.{k}")
    return v
