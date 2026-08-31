"""TTS model wrapper for Stage 1 baseline.

Encapsulates loading and inference for:
  - microsoft/speecht5_tts  (acoustic model)
  - microsoft/speecht5_hifigan  (vocoder)

The model is loaded once and reused across synthesis calls.
"""

import logging
import time
from typing import Optional

import torch
import numpy as np
from datasets import load_dataset
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

from src.config import Config

logger = logging.getLogger(__name__)


def resolve_device(device_cfg: str) -> torch.device:
    """Resolve the device string from configuration.

    Parameters
    ----------
    device_cfg:
        One of ``"auto"``, ``"cpu"``, or ``"cuda"``.

    Returns
    -------
    torch.device
        The resolved device.
    """
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device auto-selected: %s", device)
    elif device_cfg in ("cpu", "cuda"):
        device = torch.device(device_cfg)
    else:
        logger.warning(
            "Unknown device config %r — falling back to cpu.", device_cfg
        )
        device = torch.device("cpu")
    return device


class TTSModel:
    """Wrapper around SpeechT5 acoustic model and HiFi-GAN vocoder.

    Usage
    -----
    ::

        model = TTSModel(config)
        model.load()
        waveform, timings = model.synthesize("Hello world!")

    Parameters
    ----------
    config:
        Project :class:`~src.config.Config` instance.
    """

    def __init__(self, config: Config, checkpoint_path: Optional[str] = None) -> None:
        self._config = config
        self._checkpoint_path = checkpoint_path
        self._device: Optional[torch.device] = None
        self._processor: Optional[SpeechT5Processor] = None
        self._acoustic_model: Optional[SpeechT5ForTextToSpeech] = None
        self._vocoder: Optional[SpeechT5HifiGan] = None
        self._speaker_embedding: Optional[torch.Tensor] = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load processor, acoustic model, vocoder, and speaker embedding.

        This method is idempotent — calling it more than once is a no-op.

        Raises
        ------
        RuntimeError
            If a CUDA device is requested but not available.
        """
        if self._loaded:
            logger.debug("Model already loaded — skipping.")
            return

        device_cfg = self._config.inference.device
        self._device = resolve_device(device_cfg)

        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was explicitly requested but is not available on this system."
            )

        default_model = self._config.model.get("name", self._config.model.get("base_model", "microsoft/speecht5_tts"))
        model_name: str = self._checkpoint_path or default_model
        vocoder_name: str = self._config.model.get("vocoder", "microsoft/speecht5_hifigan")

        logger.info("Loading processor from %s …", model_name)
        try:
            self._processor = SpeechT5Processor.from_pretrained(model_name)
        except Exception:
            # Fallback to base model processor if checkpoint only saved model weights
            self._processor = SpeechT5Processor.from_pretrained(default_model)

        logger.info("Loading acoustic model from %s …", model_name)
        self._acoustic_model = (
            SpeechT5ForTextToSpeech.from_pretrained(model_name)
            .to(self._device)
            .eval()
        )

        logger.info("Loading vocoder from %s …", vocoder_name)
        self._vocoder = (
            SpeechT5HifiGan.from_pretrained(vocoder_name)
            .to(self._device)
            .eval()
        )

        self._speaker_embedding = self._load_speaker_embedding()

        self._loaded = True
        logger.info("Model stack loaded on %s.", self._device)

    def synthesize(self, text: str) -> tuple[np.ndarray, dict]:
        """Run acoustic model + vocoder and return a waveform.

        Parameters
        ----------
        text:
            Preprocessed, tokenizer-ready text.

        Returns
        -------
        waveform : np.ndarray
            1-D float32 array containing the raw audio samples.
        timings : dict
            Timing breakdown with keys:

            * ``acoustic_model_time_seconds``
            * ``vocoder_time_seconds``
        """
        self._assert_loaded()

        # ---- Tokenisation ----
        inputs = self._processor(
            text=text,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self._device)

        # ---- Acoustic model ----
        t0 = time.perf_counter()
        with torch.no_grad():
            speech = self._acoustic_model.generate_speech(
                input_ids,
                self._speaker_embedding,
                vocoder=self._vocoder,
            )
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # generate_speech with vocoder produces the final waveform directly.
        # Timings are combined here; the vocoder runs inside generate_speech.
        combined_time = t1 - t0

        waveform: np.ndarray = speech.cpu().numpy().astype(np.float32)

        timings = {
            "acoustic_model_time_seconds": combined_time,
            "vocoder_time_seconds": 0.0,  # Fused into generate_speech
        }

        logger.debug(
            "Synthesis done in %.3f s, waveform shape %s.",
            combined_time,
            waveform.shape,
        )
        return waveform, timings

    def get_device(self) -> str:
        """Return the active device as a human-readable string.

        Returns
        -------
        str
            E.g. ``"cpu"`` or ``"cuda:0"``.
        """
        self._assert_loaded()
        return str(self._device)

    @property
    def is_loaded(self) -> bool:
        """Whether the model stack has been loaded."""
        return self._loaded

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_speaker_embedding(self) -> torch.Tensor:
        """Fetch a pre-computed speaker x-vector embedding.

        Uses the CMU Arctic x-vectors dataset hosted on Hugging Face.

        Returns
        -------
        torch.Tensor
            Shape ``(1, 512)``, on the configured device.
        """
        dataset_name: str = self._config.model.speaker_embeddings_dataset
        split: str = self._config.model.speaker_embeddings_split
        index: int = self._config.model.speaker_embeddings_index

        logger.info(
            "Loading speaker embedding from %s[%s][%d] …",
            dataset_name,
            split,
            index,
        )
        try:
            embeddings_dataset = load_dataset(
                dataset_name,
                split=split,
            )
            embedding_np = np.array(
                embeddings_dataset[index]["xvector"], dtype=np.float32
            )
        except Exception as exc:
            logger.warning(
                "datasets.load_dataset failed (%s). Falling back to direct extraction from Hugging Face Hub.",
                exc,
            )
            import io
            import zipfile
            from huggingface_hub import hf_hub_download

            zip_path = hf_hub_download(
                repo_id=dataset_name,
                filename="spkrec-xvect.zip",
                repo_type="dataset",
            )
            with zipfile.ZipFile(zip_path) as z:
                npy_files = sorted(
                    [n for n in z.namelist() if n.endswith(".npy")]
                )
                if index >= len(npy_files):
                    raise IndexError(
                        f"Speaker embedding index {index} out of range (total {len(npy_files)} files)."
                    )
                with z.open(npy_files[index]) as f:
                    embedding_np = np.load(io.BytesIO(f.read())).astype(np.float32)

        embedding = torch.tensor(embedding_np).unsqueeze(0).to(self._device)
        logger.debug("Speaker embedding shape: %s", embedding.shape)
        return embedding

    def _assert_loaded(self) -> None:
        """Raise :class:`RuntimeError` if the model has not been loaded yet."""
        if not self._loaded:
            raise RuntimeError(
                "Model has not been loaded. Call TTSModel.load() first."
            )
