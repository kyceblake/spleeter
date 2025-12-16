#!/usr/bin/env python
# coding: utf8

""" Unit testing for Separator class. """

__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"

from os import makedirs
from os.path import join
from tempfile import TemporaryDirectory

import numpy as np

from spleeter.__main__ import evaluate
from spleeter.audio.adapter import AudioAdapter

EXPECTED_INSTRUMENTS = ("vocals", "drums", "bass", "other")
EXPECTED_METRICS = ("SDR", "SAR", "SIR", "ISR")


def generate_fake_eval_dataset(path):
    """
    Generate fake evaluation dataset
    """
    aa = AudioAdapter.default()
    n_songs = 2
    fs = 44100
    duration = 3
    n_channels = 2
    rng = np.random.RandomState(seed=0)
    for song in range(n_songs):
        song_path = join(path, "test", f"song{song}")
        makedirs(song_path, exist_ok=True)
        for instr in ["mixture", "vocals", "bass", "drums", "other"]:
            filename = join(song_path, f"{instr}.wav")
            data = rng.rand(duration * fs, n_channels) - 0.5
            aa.save(filename, data, fs)


def test_evaluate():
    with TemporaryDirectory() as dataset:
        with TemporaryDirectory() as evaluation:
            generate_fake_eval_dataset(dataset)
            metrics = evaluate(
                adapter="spleeter.audio.ffmpeg.FFMPEGProcessAudioAdapter",
                output_path=evaluation,
                params_filename="spleeter:4stems",
                mus_dir=dataset,
                mwf=False,
                verbose=False,
            )
            assert set(metrics.keys()) == set(EXPECTED_INSTRUMENTS)
            for instrument in EXPECTED_INSTRUMENTS:
                assert set(metrics[instrument].keys()) == set(EXPECTED_METRICS)
                for m in EXPECTED_METRICS:
                    values = metrics[instrument][m]
                    assert len(values) == 2
                    assert np.isfinite(values).all()
