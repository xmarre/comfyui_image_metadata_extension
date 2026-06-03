from __future__ import annotations

from collections.abc import Mapping

from .samplers import SAMPLERS


_MAX_PROMPT_NODES = 8192


def is_positive_prompt(node_id, obj, prompt, extra_data, outputs, input_data_all):
    return _node_is_directly_connected_to_side(prompt, node_id, "positive")


def is_negative_prompt(node_id, obj, prompt, extra_data, outputs, input_data_all):
    return _node_is_directly_connected_to_side(prompt, node_id, "negative")


def _prompt_items(prompt):
    """Return a bounded prompt snapshot for metadata-only validation."""
    if not isinstance(prompt, Mapping):
        return ()

    try:
        return tuple(prompt.items())[:_MAX_PROMPT_NODES]
    except Exception:
        return ()


def _inputs(node):
    if not isinstance(node, Mapping):
        return {}

    inputs = node.get("inputs") or {}
    return inputs if isinstance(inputs, Mapping) else {}


def _class_type(node):
    if not isinstance(node, Mapping):
        return ""

    value = node.get("class_type", "")
    return value if isinstance(value, str) else str(value)


def _is_link(value):
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and isinstance(value[0], (str, int))
    )


def _linked_node_id(value):
    if not _is_link(value):
        return None
    return str(value[0])


def _sampler_field_map(class_type):
    if not class_type or not isinstance(SAMPLERS, Mapping):
        return None

    try:
        field_map = SAMPLERS.get(class_type)
    except Exception:
        return None

    return field_map if isinstance(field_map, Mapping) else None


def _node_is_directly_connected_to_side(prompt, node_id, field_name):
    wanted = str(node_id)

    for _current_id, current_node in _prompt_items(prompt):
        inputs = _inputs(current_node)
        if not inputs:
            continue

        field_map = _sampler_field_map(_class_type(current_node))
        if field_map is None:
            continue

        input_name = field_map.get(field_name)
        if isinstance(input_name, str) and _linked_node_id(inputs.get(input_name)) == wanted:
            return True

    return False
