#!/usr/bin/env python
# coding: utf8

"""
Module that provides a class wrapper for source separation.

Examples:

```python
>>> from spleeter.separator import Separator
>>> separator = Separator('spleeter:2stems')
>>> separator.separate(waveform, lambda instrument, data: ...)
>>> separator.separate_to_file(...)
```
"""

import atexit
import os
from multiprocessing import Pool
from os.path import basename, dirname, isabs, join, splitext
from typing import Any, Dict, List, Optional

# pyright: reportMissingImports=false
# pylint: disable=import-error
import numpy as np
import tensorflow as tf  # type: ignore

from . import SpleeterError
from .audio import Codec
from .audio.adapter import AudioAdapter
from .audio.convertor import to_stereo
from .keras.checkpoint import assign_from_v1_checkpoint
from .keras.model import build_spectrogram_model
from .model.provider import ModelProvider
from .types import AudioDescriptor
from .utils.configuration import load_configuration
from .utils.tensor import pad_and_partition, pad_and_reshape

# pylint: enable=import-error

__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"

WINDOW_COMPENSATION_FACTOR = 2.0 / 3.0
EPSILON = 1e-10


def _hann_window(frame_length: int, dtype: tf.DType) -> tf.Tensor:
    return tf.signal.hann_window(frame_length, periodic=True, dtype=dtype)


class Separator(object):
    """A wrapper class for performing separation."""

    def __init__(
        self,
        params_descriptor: str,
        MWF: bool = False,
        multiprocess: bool = True,
    ) -> None:
        """
        Default constructor.

        Parameters:
            params_descriptor (str):
                Descriptor for TF params to be used.
            MWF (bool):
                (Optional) `True` if MWF should be used, `False` otherwise.
            multiprocess (bool):
                (Optional) Enable multi-processing.
        """
        self._params = load_configuration(params_descriptor)
        seed = int(self._params.get("random_seed", 0))
        tf.random.set_seed(seed)
        np.random.seed(seed)
        provider: ModelProvider = ModelProvider.default()
        if not isabs(self._params["model_dir"]):
            self._params["model_dir"] = provider.get(self._params["model_dir"])
        elif not os.path.exists(self._params["model_dir"]):
            raise SpleeterError(
                f"Model directory {self._params['model_dir']} does not exist"
            )
        self._sample_rate = self._params["sample_rate"]
        self._MWF = MWF

        self._model = build_spectrogram_model(self._params)
        self._checkpoint = tf.train.Checkpoint(model=self._model)
        self._checkpoint_manager = tf.train.CheckpointManager(
            self._checkpoint,
            directory=self._params["model_dir"],
            max_to_keep=2,
        )
        self._restore_model_weights()

        if multiprocess:
            try:
                self._pool = Pool()
                atexit.register(self._pool.close)
            except (OSError, PermissionError) as exc:
                tf.get_logger().warning(
                    "Failed to initialize multiprocessing pool (%s); running single-process.",
                    exc,
                )
                self._pool = None
        else:
            self._pool = None
        self._tasks: List = []

    def join(self, timeout: int = 200) -> None:
        """
        Wait for all pending tasks to be finished.

        Parameters:
            timeout (int):
                (Optional) Task waiting timeout.
        """
        while len(self._tasks) > 0:
            task = self._tasks.pop()
            task.get()
            task.wait(timeout=timeout)

    def _restore_model_weights(self) -> None:
        latest = tf.train.latest_checkpoint(self._params["model_dir"])
        if latest is None:
            tf.get_logger().warning(
                "No checkpoint found in %s; using randomly initialized weights.",
                self._params["model_dir"],
            )
            self._checkpoint_manager.save(checkpoint_number=0)
            return
        if basename(latest).startswith("ckpt-"):
            self._checkpoint.restore(latest).expect_partial()
            return

        report = assign_from_v1_checkpoint(self._model, latest)
        if report.total == 0 or report.matched < max(1, int(0.5 * report.total)):
            raise SpleeterError(
                "Failed to restore enough weights from legacy TF1 checkpoint "
                f"({report.matched}/{report.total} matched) at {latest}."
            )
        self._checkpoint_manager.save(checkpoint_number=0)

    def _inverse_stft(
        self, stft_t: tf.Tensor, time_crop: tf.Tensor, frame_length: int, frame_step: int
    ) -> tf.Tensor:
        inversed = (
            tf.signal.inverse_stft(
                tf.transpose(stft_t, perm=[2, 0, 1]),
                frame_length,
                frame_step,
                window_fn=lambda fl, dtype: _hann_window(fl, dtype),
            )
            * WINDOW_COMPENSATION_FACTOR
        )
        reshaped = tf.transpose(inversed)
        return reshaped[frame_length : frame_length + time_crop, :]

    def _separate_keras(self, waveform: np.ndarray, audio_descriptor: AudioDescriptor) -> Dict:
        """
        Performs source separation over the given waveform with Keras backend.

        Parameters:
            waveform (np.ndarray):
                Waveform to be separated (as a numpy array)
            audio_descriptor (AudioDescriptor):
                Audio descriptor to be used.

        Returns:
            Dict:
                Separated waveforms.
        """
        if not waveform.shape[-1] == 2:
            waveform = to_stereo(waveform)

        frame_length = int(self._params["frame_length"])
        frame_step = int(self._params["frame_step"])
        T = int(self._params["T"])
        F = int(self._params["F"])
        n_channels = int(self._params["n_channels"])
        mix_name = str(self._params["mix_name"])
        instruments = list(self._params["instrument_list"])

        waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
        time_crop = tf.shape(waveform_tf)[0]

        waveform_padded = tf.concat(
            [tf.zeros((frame_length, n_channels), dtype=waveform_tf.dtype), waveform_tf],
            axis=0,
        )
        stft_feature = tf.transpose(
            tf.signal.stft(
                tf.transpose(waveform_padded),
                frame_length,
                frame_step,
                window_fn=lambda fl, dtype: _hann_window(fl, dtype),
                pad_end=True,
            ),
            perm=[1, 2, 0],
        )

        stft_partitioned = pad_and_partition(stft_feature, T)
        mix_spectrogram = tf.abs(stft_partitioned)[:, :, :F, :]

        outputs = self._model({f"{mix_name}_spectrogram": mix_spectrogram}, training=False)

        if self._MWF:
            import norbert  # type: ignore # pylint: disable=import-error

            x = stft_feature.numpy()
            v = np.stack(
                [
                    pad_and_reshape(outputs[f"{instrument}_spectrogram"], frame_length, F)
                    .numpy()[: x.shape[0], ...]
                    for instrument in instruments
                ],
                axis=3,
            )
            y = norbert.wiener(v, x)
            return {
                instrument: self._inverse_stft(
                    tf.convert_to_tensor(y[:, :, :, k], dtype=tf.complex64),
                    time_crop,
                    frame_length,
                    frame_step,
                ).numpy()
                for k, instrument in enumerate(instruments)
            }

        separation_exponent = float(self._params.get("separation_exponent", 2.0))
        output_sum = (
            tf.add_n(
                [tf.pow(outputs[f"{instrument}_spectrogram"], separation_exponent) for instrument in instruments]
            )
            + EPSILON
        )

        def extend_mask(mask: tf.Tensor) -> tf.Tensor:
            extension = str(self._params["mask_extension"])
            if extension == "average":
                extension_row = tf.reduce_mean(mask, axis=2, keepdims=True)
            elif extension == "zeros":
                mask_shape = tf.shape(mask)
                extension_row = tf.zeros(
                    (mask_shape[0], mask_shape[1], 1, mask_shape[-1]), dtype=mask.dtype
                )
            else:
                raise SpleeterError(f"Invalid mask_extension parameter {extension}")
            n_extra_row = frame_length // 2 + 1 - F
            extension_t = tf.tile(extension_row, [1, 1, n_extra_row, 1])
            return tf.concat([mask, extension_t], axis=2)

        prediction: Dict[str, np.ndarray] = {}
        for instrument in instruments:
            output = outputs[f"{instrument}_spectrogram"]
            instrument_mask = (
                tf.pow(output, separation_exponent) + (EPSILON / len(instruments))
            ) / output_sum
            instrument_mask = extend_mask(instrument_mask)
            old_shape = tf.shape(instrument_mask)
            instrument_mask = tf.reshape(
                instrument_mask,
                tf.concat([[old_shape[0] * old_shape[1]], old_shape[2:]], axis=0),
            )
            instrument_mask = instrument_mask[: tf.shape(stft_feature)[0], ...]
            masked_stft = tf.cast(instrument_mask, dtype=tf.complex64) * stft_feature
            prediction[instrument] = self._inverse_stft(
                masked_stft, time_crop, frame_length, frame_step
            ).numpy()
        return prediction

    def separate(
        self, waveform: np.ndarray, audio_descriptor: Optional[str] = ""
    ) -> Dict:
        """
        Performs separation on a waveform.

        Parameters:
            waveform (np.ndarray):
                Waveform to be separated (as a numpy array)
            audio_descriptor (Optional[str]):
                (Optional) string describing the waveform (e.g. filename).

        Returns:
            Dict:
                Separated waveforms.
        """
        return self._separate_keras(waveform, audio_descriptor)

    def separate_to_file(
        self,
        audio_descriptor: AudioDescriptor,
        destination: str,
        audio_adapter: Optional[AudioAdapter] = None,
        offset: float = 0,
        duration: float = 600.0,
        codec: Codec = Codec.WAV,
        bitrate: str = "128k",
        filename_format: str = "{filename}/{instrument}.{codec}",
        synchronous: bool = True,
    ) -> None:
        """
        Performs source separation and export result to file using
        given audio adapter.

        Filename format should be a Python formattable string that could
        use following parameters :

        - {instrument}
        - {filename}
        - {foldername}
        - {codec}.

        Parameters:
            audio_descriptor (AudioDescriptor):
                Describe song to separate, used by audio adapter to
                retrieve and load audio data, in case of file based
                audio adapter, such descriptor would be a file path.
            destination (str):
                Target directory to write output to.
            audio_adapter (AudioAdapter):
                (Optional) Audio adapter to use for I/O.
            offset (int):
                (Optional) Offset of loaded song.
            duration (float):
                (Optional) Duration of loaded song (default: 600s).
            codec (Codec):
                (Optional) Export codec.
            bitrate (str):
                (Optional) Export bitrate.
            filename_format (str):
                (Optional) Filename format.
            synchronous (bool):
                (Optional) True is should by synchronous.
        """
        if audio_adapter is None:
            audio_adapter = AudioAdapter.default()
        waveform, _ = audio_adapter.load(
            audio_descriptor,
            offset=offset,
            duration=duration,
            sample_rate=self._sample_rate,
        )
        sources = self.separate(waveform, audio_descriptor)
        self.save_to_file(
            sources,
            audio_descriptor,
            destination,
            filename_format,
            codec,
            audio_adapter,
            bitrate,
            synchronous,
        )

    def save_to_file(
        self,
        sources: Dict,
        audio_descriptor: AudioDescriptor,
        destination: str,
        filename_format: str = "{filename}/{instrument}.{codec}",
        codec: Codec = Codec.WAV,
        audio_adapter: Optional[AudioAdapter] = None,
        bitrate: str = "128k",
        synchronous: bool = True,
    ) -> None:
        """
        Export dictionary of sources to files.

        Parameters:
            sources (Dict):
                Dictionary of sources to be exported. The keys are the name
                of the instruments, and the values are `N x 2` numpy arrays
                containing the corresponding intrument waveform, as
                returned by the separate method
            audio_descriptor (AudioDescriptor):
                Describe song to separate, used by audio adapter to
                retrieve and load audio data, in case of file based audio
                adapter, such descriptor would be a file path.
            destination (str):
                Target directory to write output to.
            filename_format (str):
                (Optional) Filename format.
            codec (Codec):
                (Optional) Export codec.
            audio_adapter (Optional[AudioAdapter]):
                (Optional) Audio adapter to use for I/O.
            bitrate (str):
                (Optional) Export bitrate.
            synchronous (bool):
                (Optional) True is should by synchronous.
        """
        if audio_adapter is None:
            audio_adapter = AudioAdapter.default()
        foldername = basename(dirname(audio_descriptor))
        filename = splitext(basename(audio_descriptor))[0]
        generated = []
        for instrument, data in sources.items():
            path = join(
                destination,
                filename_format.format(
                    filename=filename,
                    instrument=instrument,
                    foldername=foldername,
                    codec=codec,
                ),
            )
            directory = os.path.dirname(path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            if path in generated:
                raise SpleeterError(
                    (
                        f"Separated source path conflict : {path},"
                        "please check your filename format"
                    )
                )
            generated.append(path)
            if self._pool:
                task = self._pool.apply_async(
                    audio_adapter.save, (path, data, self._sample_rate, codec, bitrate)
                )
                self._tasks.append(task)
            else:
                audio_adapter.save(path, data, self._sample_rate, codec, bitrate)
        if synchronous and self._pool:
            self.join()
