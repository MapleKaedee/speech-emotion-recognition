import unittest
from unittest.mock import patch

import numpy as np
import torch

from utils import get_transcription_waveform, transcribe_audio, transcribe_audio_segments


class RecordingASRPipeline:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.output


class WhisperTranscriptionTests(unittest.TestCase):
    @patch("utils.load_audio")
    def test_transcription_waveform_is_full_sanitized_resampled_and_normalized(self, load_audio):
        load_audio.return_value = (
            torch.tensor([[0.0, float("nan"), -0.25, 0.5]], dtype=torch.float32),
            8000,
        )

        waveform = get_transcription_waveform("rekaman.wav")

        self.assertEqual(waveform.dtype, np.float32)
        self.assertEqual(waveform.shape, (8,))
        self.assertTrue(np.isfinite(waveform).all())
        self.assertAlmostEqual(float(np.max(np.abs(waveform))), 1.0, places=5)

    def test_transcribe_audio_uses_stable_indonesian_decoding(self):
        asr = RecordingASRPipeline({"text": "ini kok gitu sih"})
        waveform = np.zeros(16000, dtype=np.float32)

        transcript = transcribe_audio(asr, waveform)

        self.assertEqual(transcript, "ini kok gitu sih")
        kwargs = asr.calls[0][1]
        self.assertEqual(kwargs["chunk_length_s"], 30)
        self.assertEqual(
            kwargs["generate_kwargs"],
            {
                "language": "indonesian",
                "task": "transcribe",
                "num_beams": 5,
                "temperature": (0.0, 0.2, 0.4, 0.6),
                "compression_ratio_threshold": 1.35,
                "logprob_threshold": -1.0,
                "condition_on_prev_tokens": False,
            },
        )

    def test_transcribe_segments_uses_same_decoding_configuration(self):
        asr = RecordingASRPipeline(
            {
                "text": "ini kok gitu sih",
                "chunks": [
                    {"text": "ini kok gitu sih", "timestamp": (0.0, 1.5)},
                ],
            }
        )
        waveform = np.zeros(16000, dtype=np.float32)

        transcript, segments = transcribe_audio_segments(asr, waveform)

        self.assertEqual(transcript, "ini kok gitu sih")
        self.assertEqual(segments[0]["text"], "ini kok gitu sih")
        self.assertTrue(asr.calls[0][1]["return_timestamps"])
        self.assertEqual(
            asr.calls[0][1]["generate_kwargs"]["temperature"],
            (0.0, 0.2, 0.4, 0.6),
        )


if __name__ == "__main__":
    unittest.main()
