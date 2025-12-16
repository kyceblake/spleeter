#!/usr/bin/env python
# coding: utf8

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict

import tensorflow as tf  # type: ignore


def _get_model_function(model_type: str) -> Callable[..., Dict[str, tf.Tensor]]:
    relative_path_to_module = ".".join(model_type.split(".")[:-1])
    model_name = model_type.split(".")[-1]
    module_path = "spleeter.model.functions"
    if relative_path_to_module:
        module_path = f"{module_path}.{relative_path_to_module}"
    module = importlib.import_module(module_path)
    model_function = getattr(module, model_name)
    return model_function


def build_spectrogram_model(params: Dict[str, Any]) -> tf.keras.Model:
    mix_name = params["mix_name"]
    instruments = params["instrument_list"]
    T = int(params["T"])
    F = int(params["F"])
    n_channels = int(params["n_channels"])

    model_cfg = params.get("model") or {}
    model_type = str(model_cfg.get("type", "unet.unet"))
    model_params = dict(model_cfg.get("params") or {})

    mix_input = tf.keras.Input(
        shape=(T, F, n_channels), name=f"{mix_name}_spectrogram"
    )
    model_fn = _get_model_function(model_type)
    outputs = model_fn(mix_input, instruments, model_params)
    return tf.keras.Model(
        inputs={f"{mix_name}_spectrogram": mix_input},
        outputs=outputs,
        name="spleeter",
    )

