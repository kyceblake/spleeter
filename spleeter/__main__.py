#!/usr/bin/env python
# coding: utf8

"""
Python oneliner script usage.

USAGE: python -m spleeter {train,evaluate,separate} ...

Notes:
    All critical import involving TF, numpy or Pandas are deported to
    command function scope to avoid heavy import on CLI evaluation,
    leading to large bootstraping time.
"""
import json
from glob import glob
from itertools import product
from os.path import join
from typing import Dict, List, Optional, Tuple

# pyright: reportMissingImports=false
# pylint: disable=import-error
from typer import Exit, Typer

from . import SpleeterError
from .audio import Codec
from .options import (
    AudioAdapterOption,
    AudioBitrateOption,
    AudioCodecOption,
    AudioDurationOption,
    AudioInputArgument,
    AudioInputOption,
    AudioOffsetOption,
    AudioOutputOption,
    FilenameFormatOption,
    ModelParametersOption,
    MUSDBDirectoryOption,
    MWFOption,
    TrainingDataDirectoryOption,
    VerboseOption,
    VersionOption,
)
from .utils.logging import configure_logger, logger

# pylint: enable=import-error

spleeter: Typer = Typer(add_completion=False, no_args_is_help=True, short_help="-h")
""" CLI application. """


@spleeter.callback()
def default(
    version: bool = VersionOption,
) -> None:
    pass


@spleeter.command(no_args_is_help=True)
def train(
    adapter: str = AudioAdapterOption,
    data: str = TrainingDataDirectoryOption,
    params_filename: str = ModelParametersOption,
    verbose: bool = VerboseOption,
) -> None:
    """
    Train a source separation model
    """
    import tensorflow as tf  # type: ignore

    from .audio.adapter import AudioAdapter
    from .dataset import get_training_dataset, get_validation_dataset
    from .keras.model import build_spectrogram_model
    from .model.provider import ModelProvider
    from .utils.configuration import load_configuration

    configure_logger(verbose)
    audio_adapter = AudioAdapter.get(adapter)
    params = load_configuration(params_filename)

    import os

    os.makedirs(params["model_dir"], exist_ok=True)
    model = build_spectrogram_model(params)

    optimizer_name = params.get("optimizer")
    learning_rate = float(params.get("learning_rate", 1e-4))
    if optimizer_name == "Adadelta":
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    loss_type = str(params.get("loss_type", "L1_mask"))

    def l1_loss(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_pred - y_true))

    def weighted_l1_loss(y_true, y_pred):
        weights = tf.reduce_mean(y_true, axis=[1, 2, 3], keepdims=True)
        return tf.reduce_mean(weights * tf.abs(y_pred - y_true))

    if loss_type == "weighted_L1_mask":
        loss_fn = weighted_l1_loss
    else:
        loss_fn = l1_loss

    instruments = list(params["instrument_list"])
    model.compile(
        optimizer=optimizer,
        loss={f"{instrument}_spectrogram": loss_fn for instrument in instruments},
    )

    train_dataset = get_training_dataset(params, audio_adapter, data)
    validation_dataset = get_validation_dataset(params, audio_adapter, data)

    step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
    checkpoint = tf.train.Checkpoint(step=step, optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, params["model_dir"], max_to_keep=2)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).expect_partial()

    save_steps = int(params.get("save_checkpoints_steps", 100))

    class _StepCheckpointCallback(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            step.assign_add(1)
            current = int(step.numpy())
            if current % save_steps == 0:
                manager.save(checkpoint_number=current)

    max_steps = int(params.get("train_max_steps", 10))
    logger.info("Start model training")
    model.fit(
        train_dataset,
        steps_per_epoch=max_steps,
        epochs=1,
        validation_data=validation_dataset,
        validation_steps=1,
        callbacks=[_StepCheckpointCallback()],
        verbose=1 if verbose else 0,
    )

    manager.save(checkpoint_number=int(step.numpy()))
    ModelProvider.writeProbe(params["model_dir"])
    logger.info("Model training done")


@spleeter.command(no_args_is_help=True)
def separate(
    deprecated_files: Optional[str] = AudioInputOption,
    files: List[str] = AudioInputArgument,
    adapter: str = AudioAdapterOption,
    bitrate: str = AudioBitrateOption,
    codec: Codec = AudioCodecOption,
    duration: float = AudioDurationOption,
    offset: float = AudioOffsetOption,
    output_path: str = AudioOutputOption,
    filename_format: str = FilenameFormatOption,
    params_filename: str = ModelParametersOption,
    mwf: bool = MWFOption,
    verbose: bool = VerboseOption,
) -> None:
    """
    Separate audio file(s)
    """
    from .audio.adapter import AudioAdapter
    from .separator import Separator

    configure_logger(verbose)
    if deprecated_files is not None:
        logger.error(
            "⚠️ -i option is not supported anymore, audio files must be supplied "
            "using input argument instead (see spleeter separate --help)"
        )
        raise Exit(20)
    audio_adapter: AudioAdapter = AudioAdapter.get(adapter)
    separator: Separator = Separator(params_filename, MWF=mwf)

    for filename in files:
        separator.separate_to_file(
            filename,
            output_path,
            audio_adapter=audio_adapter,
            offset=offset,
            duration=duration,
            codec=codec,
            bitrate=bitrate,
            filename_format=filename_format,
            synchronous=False,
        )
    separator.join()


EVALUATION_SPLIT: str = "test"
EVALUATION_METRICS_DIRECTORY: str = "metrics"
EVALUATION_INSTRUMENTS: Tuple[str, ...] = ("vocals", "drums", "bass", "other")
EVALUATION_METRICS: Tuple[str, ...] = ("SDR", "SAR", "SIR", "ISR")
EVALUATION_MIXTURE: str = "mixture.wav"
EVALUATION_AUDIO_DIRECTORY: str = "audio"


def _compile_metrics(metrics_output_directory: str) -> Dict:
    """
    Compiles metrics from given directory and returns results as dict.

    Parameters:
        metrics_output_directory (str):
            Directory to get metrics from.

    Returns:
        Dict:
            Compiled metrics as dict.
    """
    import numpy as np
    import pandas as pd  # type: ignore

    songs = glob(join(metrics_output_directory, "test/*.json"))
    index = pd.MultiIndex.from_tuples(
        product(EVALUATION_INSTRUMENTS, EVALUATION_METRICS),
        names=["instrument", "metric"],
    )
    pd.DataFrame([], index=["config1", "config2"], columns=index)
    metrics: Dict = {
        instrument: {k: [] for k in EVALUATION_METRICS}
        for instrument in EVALUATION_INSTRUMENTS
    }
    for song in songs:
        with open(song, "r") as stream:
            data = json.load(stream)
        for target in data["targets"]:
            instrument = target["name"]
            for metric in EVALUATION_METRICS:
                sdr_med = np.median(
                    [
                        frame["metrics"][metric]
                        for frame in target["frames"]
                        if not np.isnan(frame["metrics"][metric])
                    ]
                )
                metrics[instrument][metric].append(sdr_med)
    return metrics


@spleeter.command(no_args_is_help=True)
def evaluate(
    adapter: str = AudioAdapterOption,
    output_path: str = AudioOutputOption,
    params_filename: str = ModelParametersOption,
    mus_dir: str = MUSDBDirectoryOption,
    mwf: bool = MWFOption,
    verbose: bool = VerboseOption,
) -> Dict:
    """
    Evaluate a model on the musDB test dataset
    """
    import numpy as np

    # Compatibility for third-party packages (e.g., stempeg/musdb) on newer NumPy.
    if not hasattr(np, "float_"):
        np.float_ = np.float64  # type: ignore[attr-defined]
    if not hasattr(np, "complex_"):
        np.complex_ = np.complex128  # type: ignore[attr-defined]

    configure_logger(verbose)
    try:
        import musdb  # type: ignore
        import museval  # type: ignore
    except ImportError:
        logger.error("Extra dependencies musdb and museval not found")
        logger.error("Please install musdb and museval first, abort")
        raise Exit(10)
    # Separate musdb sources.
    songs = glob(join(mus_dir, EVALUATION_SPLIT, "*/"))
    mixtures = [join(song, EVALUATION_MIXTURE) for song in songs]
    audio_output_directory = join(output_path, EVALUATION_AUDIO_DIRECTORY)
    separate(
        deprecated_files=None,
        files=mixtures,
        adapter=adapter,
        bitrate="128k",
        codec=Codec.WAV,
        duration=600.0,
        offset=0,
        output_path=join(audio_output_directory, EVALUATION_SPLIT),
        filename_format="{foldername}/{instrument}.{codec}",
        params_filename=params_filename,
        mwf=mwf,
        verbose=verbose,
    )
    # Compute metrics with musdb.
    metrics_output_directory = join(output_path, EVALUATION_METRICS_DIRECTORY)
    logger.info("Starting musdb evaluation (this could be long) ...")
    dataset = musdb.DB(root=mus_dir, is_wav=True, subsets=[EVALUATION_SPLIT])
    museval.eval_mus_dir(
        dataset=dataset,
        estimates_dir=audio_output_directory,
        output_dir=metrics_output_directory,
    )
    logger.info("musdb evaluation done")
    # Compute and pretty print median metrics.
    metrics = _compile_metrics(metrics_output_directory)
    for instrument, metric in metrics.items():
        logger.info(f"{instrument}:")
        for metric, value in metric.items():
            logger.info(f"{metric}: {np.median(value):.3f}")
    return metrics


def entrypoint():
    """Application entrypoint."""
    try:
        spleeter()
    except SpleeterError as e:
        logger.error(e)


if __name__ == "__main__":
    entrypoint()
