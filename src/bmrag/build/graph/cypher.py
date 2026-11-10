"""Writing and reading the Cypher text the graph stages pass along.

Records carry their graph as Cypher MERGE statements; these helpers quote values, split the
statements and find the labels and relationship types they use.
"""

from __future__ import annotations

import json
import re
from typing import Any

from bmrag.build.graph.ontology import canonical_name

COMMENT_RE = re.compile(r"^\s*//.*$", re.MULTILINE)
LABEL_RE = re.compile(r"\([A-Za-z_][A-Za-z0-9_]*:([A-Za-z_][A-Za-z0-9_]*)")
REL_RE = re.compile(r"\[:([A-Za-z_][A-Za-z0-9_]*)")
NAME_PROP_RE = re.compile(r"(name\s*:\s*)'((?:\\'|[^'])*)'")


def clean_markdown_cypher(text: str) -> str:
    """Remove Markdown fences and literal backticks from generated Cypher."""
    text = text or ""
    text = re.sub(r"```(?:cypher|json)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    return text.replace("`", "").strip()


def cypher_string(value: Any) -> str:
    """A single-quoted Cypher string. Semicolons become commas, because statements are split on them."""
    if value is None:
        return "null"
    text = str(value).replace(";", ",")
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def cypher_value(value: Any) -> str:
    """A Cypher literal; dicts and lists of dicts are stored as JSON strings."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        return cypher_string(json.dumps(value, ensure_ascii=False, sort_keys=True))
    if isinstance(value, (list, tuple)):
        if any(isinstance(v, dict) for v in value):
            return cypher_string(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return "[" + ", ".join(cypher_value(v) for v in value) + "]"
    return cypher_string(value)


def cypher_props(props: dict[str, Any]) -> str:
    """A Cypher property map; None values are left out and keys made identifier-safe."""
    parts = []
    for key, value in props.items():
        if value is None:
            continue
        safe_key = re.sub(r"[^A-Za-z0-9_]", "_", key)
        if not safe_key or safe_key[0].isdigit():
            safe_key = f"p_{safe_key}"
        parts.append(f"{safe_key}: {cypher_value(value)}")
    return "{ " + ", ".join(parts) + " }" if parts else "{}"


def split_cypher_statements(text: str) -> list[str]:
    cleaned = COMMENT_RE.sub("", clean_markdown_cypher(text))
    return [stmt.strip() + ";" for stmt in cleaned.split(";") if stmt.strip()]


def extract_labels(text: str) -> set[str]:
    return set(LABEL_RE.findall(text))


def extract_relations(text: str) -> set[str]:
    return set(REL_RE.findall(text))


def substitute_canonical_names(cypher: str, canonical_map: dict[str, str]) -> str:
    """Replace every name: '...' value with its canonical form."""

    def repl(match: re.Match[str]) -> str:
        prefix, value = match.groups()
        original = value.replace("\\'", "'")
        return f"{prefix}{cypher_string(canonical_map.get(original, canonical_name(original)))}"

    return NAME_PROP_RE.sub(repl, cypher)
