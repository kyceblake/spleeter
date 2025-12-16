#!/usr/bin/env python
# coding: utf8

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import tensorflow as tf  # type: ignore


@dataclass(frozen=True)
class V1RestoreReport:
    matched: int
    total: int
    unmatched: Tuple[str, ...]


def _candidate_checkpoint_keys(
    variable_name: str, checkpoint_keys: Iterable[str]
) -> Iterable[str]:
    yield variable_name
    if variable_name.startswith("model/"):
        yield variable_name[len("model/") :]
    suffix = f"/{variable_name}"
    for key in checkpoint_keys:
        if key.endswith(suffix):
            yield key


def assign_from_v1_checkpoint(
    model: tf.keras.Model, checkpoint_prefix: str
) -> V1RestoreReport:
    reader = tf.train.load_checkpoint(checkpoint_prefix)
    checkpoint_map: Dict[str, tf.TensorShape] = reader.get_variable_to_shape_map()
    checkpoint_keys = tuple(checkpoint_map.keys())

    matched = 0
    unmatched = []
    for var in model.variables:
        var_name = var.name.split(":")[0]
        restored = False
        for key in _candidate_checkpoint_keys(var_name, checkpoint_keys):
            if key not in checkpoint_map:
                continue
            value = reader.get_tensor(key)
            if tuple(value.shape) != tuple(var.shape):
                continue
            var.assign(value)
            matched += 1
            restored = True
            break
        if not restored:
            unmatched.append(var_name)
    return V1RestoreReport(matched=matched, total=len(model.variables), unmatched=tuple(unmatched))

