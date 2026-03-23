from collections import deque

from .samplers import SAMPLERS
from comfy_execution.graph_utils import is_link


def is_positive_prompt(node_id, obj, prompt, extra_data, outputs, input_data_all):
    return node_id in _get_node_id_list(prompt, "positive")


def is_negative_prompt(node_id, obj, prompt, extra_data, outputs, input_data_all):
    return node_id in _get_node_id_list(prompt, "negative")


def _get_node_id_list(prompt, field_name):
    node_id_list = {}
    for nid, node in prompt.items():
        for sampler_type, field_map in SAMPLERS.items():
            if node["class_type"] == sampler_type:
                # There are nodes between "KSampler" and "CLIP Text Encode" in the SD3 workflow
                visited = set()
                d = deque()
                if field_name in field_map:
                    link_value = node["inputs"].get(field_map[field_name])
                    if is_link(link_value):
                        upstream_id = link_value[0]
                        if upstream_id in prompt:
                            d.append(upstream_id)

                while d:
                    nid2 = d.popleft()
                    if nid2 in visited:
                        continue
                    visited.add(nid2)

                    if nid2 not in prompt:
                        continue

                    class_type = prompt[nid2]["class_type"]
                    if "CLIPTextEncode" in class_type:
                        node_id_list[nid] = nid2
                        break
                    for v in prompt[nid2].get("inputs", {}).values():
                        if is_link(v):
                            upstream_id = v[0]
                            if upstream_id in prompt and upstream_id not in visited:
                                d.append(upstream_id)

    return list(node_id_list.values())
