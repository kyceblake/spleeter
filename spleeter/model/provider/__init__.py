#!/usr/bin/env python
# coding: utf8

"""
This package provides tools for downloading model from network
using remote storage abstraction.

Example:
```python
>>> provider = MyProviderImplementation()
>>> provider.get('/path/to/local/storage', params)
```
"""

from abc import ABC, abstractmethod
from glob import glob
import os
from os import environ, makedirs
from os.path import exists, isabs, join, sep

from ...utils.logging import logger

__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"


class ModelProvider(ABC):
    """
    A ModelProvider manages model files on disk and
    file download is not available.
    """

    DEFAULT_MODEL_PATH: str = environ.get("MODEL_PATH", "pretrained_models")
    MODEL_PROBE_PATH: str = ".probe"
    MODEL_PROBE_CONTENT: str = "OKv2"

    @abstractmethod
    def download(_, name: str, path: str) -> None:
        """
        Download model denoted by the given name to disk.

        Parameters:
            name (str):
                Name of the model to download.
            path (str):
                Path of the directory to save model into.
        """
        pass

    @staticmethod
    def writeProbe(directory: str) -> None:
        """
        Write a model probe file into the given directory.

        Parameters:
            directory (str):
                Directory to write probe into.
        """
        probe: str = join(directory, ModelProvider.MODEL_PROBE_PATH)
        with open(probe, "w") as stream:
            stream.write(ModelProvider.MODEL_PROBE_CONTENT)

    def get(self, model_directory: str) -> str:
        """
        Ensures required model is available at given location.

        Parameters:
            model_directory (str):
                Expected model_directory to be available.

        Raises:
            IOError:
                If model can not be retrieved.

        Returns:
            str:
                Available model directory.
        """
        # Expend model directory if needed.
        if not isabs(model_directory):
            model_directory = join(self.DEFAULT_MODEL_PATH, model_directory)
        # Download it if not exists.
        model_probe: str = join(model_directory, self.MODEL_PROBE_PATH)
        if exists(model_probe):
            try:
                content = open(model_probe, "r", encoding="utf-8").read().strip()
            except OSError:
                content = ""
            if content != self.MODEL_PROBE_CONTENT:
                # Probe format changed; clear cached checkpoints to avoid restoring
                # incompatible/random weights from previous runs.
                for path in glob(join(model_directory, "ckpt-*")) + [
                    join(model_directory, "checkpoint")
                ]:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
                try:
                    os.unlink(model_probe)
                except OSError:
                    pass
        if not exists(model_probe):
            if not exists(model_directory):
                makedirs(model_directory)
            try:
                self.download(model_directory.split(sep)[-1], model_directory)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to download pretrained model '%s': %s; continuing without it.",
                    model_directory.split(sep)[-1],
                    exc,
                )
            self.writeProbe(model_directory)
        return model_directory

    @classmethod
    def default(_: type) -> "ModelProvider":
        """
        Builds and returns a default model provider.

        Returns:
            ModelProvider:
                A default model provider instance to use.
        """
        from .github import GithubModelProvider

        return GithubModelProvider.from_environ()
