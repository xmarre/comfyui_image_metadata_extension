import logging
from collections import deque

from .samplers import SAMPLERS

try:
    from comfy_execution.graph_utils import is_link as _comfy_is_link
except ImportError:
    _comfy_is_link = None

logger = logging.getLogger(__name__)


def is_positive_prompt(node_id, obj, prompt, extra_data, outputs, input_data_all):
    return _same_node_id(node_id, _get_node_id_list(prompt, "positive"))


def is_negative_prompt(node_id, obj, prompt, extra_data, outputs, input_data_all):
    return _same_node_id(node_id, _get_node_id_list(prompt, "negative"))


def _same_node_id(node_id, node_ids):
    node_id = str(node_id)
    return any(str(candidate) == node_id for candidate in node_ids)


def _prompt_dict(prompt):
    return prompt if isinstance(prompt, dict) else {}


def _get_node(prompt, node_id):
    prompt = _prompt_dict(prompt)
    if node_id in prompt:
        return prompt[node_id]
    str_node_id = str(node_id)
    if str_node_id in prompt:
        return prompt[str_node_id]
    try:
        int_node_id = int(str_node_id)
    except (TypeError, ValueError):
        return None
    return prompt.get(int_node_id)


def _get_inputs(node):
    if not isinstance(node, dict):
        return {}
    inputs = node.get("inputs", {})
    return inputs if isinstance(inputs, dict) else {}


def _get_class_type(node):
    if not isinstance(node, dict):
        return ""
    class_type = node.get("class_type", "")
    return class_type if isinstance(class_type, str) else str(class_type)


def _is_link(value):
    if _comfy_is_link is not None:
        try:
            if _comfy_is_link(value):
                return True
        except Exception as e:
            logger.debug("ComfyUI is_link failed; falling back to local link check: %s", e)

    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
    )


def _linked_node_id(value):
    if not _is_link(value):
        return None
    return str(value[0])


def _iter_linked_node_ids(inputs):
    for value in inputs.values():
        linked_node_id = _linked_node_id(value)
        if linked_node_id is not None:
            yield linked_node_id


def _looks_like_prompt_encode_node(node):
    class_type = _get_class_type(node)
    class_type_lower = class_type.lower()
    return (
        "cliptextencode" in class_type_lower
        or "textencode" in class_type_lower
        or class_type in {"CLIPTextEncodeWithBreak", "AdvancedCLIPTextEncodeWithBreak"}
    )


def _walk_upstream_prompt_nodes(prompt, start_node_id):
    prompt = _prompt_dict(prompt)
    found = []
    queue = deque([str(start_node_id)])
    visited = set()

    while queue:
        current_id = queue.popleft()
        if current_id in visited:
            continue
        visited.add(current_id)

        current_node = _get_node(prompt, current_id)
        if current_node is None:
            continue

        if _looks_like_prompt_encode_node(current_node):
            found.append(current_id)
            # Keep walking: wrapper prompt nodes can feed into CLIPTextEncode nodes,
            # and the nearest/trace filter will decide which metadata survives.

        for upstream_id in _iter_linked_node_ids(_get_inputs(current_node)):
            if upstream_id not in visited:
                queue.append(upstream_id)

    return found


def _iter_sampler_prompt_links(prompt, field_name):
    prompt = _prompt_dict(prompt)

    # Use a snapshot because extensions mutate SAMPLERS during startup.
    sampler_items = tuple(SAMPLERS.items()) if isinstance(SAMPLERS, dict) else ()

    for _node_id, node in prompt.items():
        class_type = _get_class_type(node)
        inputs = _get_inputs(node)
        yielded_for_node = set()

        for sampler_type, field_map in sampler_items:
            if class_type != sampler_type or not isinstance(field_map, dict):
                continue

            input_name = field_map.get(field_name)
            if not input_name:
                continue

            linked_node_id = _linked_node_id(inputs.get(input_name))
            if linked_node_id is not None:
                yielded_for_node.add(linked_node_id)
                yield linked_node_id

        # Fallback for custom sampler/guider nodes that expose direct
        # positive/negative conditioning inputs but are not in SAMPLERS yet.
        linked_node_id = _linked_node_id(inputs.get(field_name))
        if linked_node_id is not None and linked_node_id not in yielded_for_node:
            yield linked_node_id


def _get_node_id_list(prompt, field_name):
    prompt = _prompt_dict(prompt)
    node_id_list = []
    seen = set()

    for linked_node_id in _iter_sampler_prompt_links(prompt, field_name):
        candidates = _walk_upstream_prompt_nodes(prompt, linked_node_id)
        if not candidates:
            candidates = [linked_node_id]

        for candidate in candidates:
            candidate = str(candidate)
            if candidate not in seen:
                seen.add(candidate)
                node_id_list.append(candidate)

    return node_id_list
