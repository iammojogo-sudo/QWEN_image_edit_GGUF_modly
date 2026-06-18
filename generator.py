import os
import random
import sys
import threading
import time
import uuid
from io import BytesIO
from pathlib import Path

from services.generators.base import BaseGenerator, smooth_progress

# keep stdout clean for the runner protocol
_print = print
def print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    _print(*args, **kwargs)

# 4-bit GGUF transformer weights (diffusers-loadable) ...
GGUF_REPO = "calcuis/qwen-image-edit-gguf"
GGUF_FILE = "qwen-image-edit-iq4_nl.gguf"
# ... paired with the "decoder" bundle that holds the VAE, text encoder, tokenizer,
# processor, scheduler and the transformer config the GGUF weights plug into.
DECODER_REPO = "callgg/image-edit-decoder"

# Both repos are public Apache-2.0 mirrors — no gate, no token needed.


def _int(val, default):
    try:
        return int(val)
    except:
        return default


def _float(val, default):
    try:
        return float(val)
    except:
        return default


class QwenImageEditGenerator(BaseGenerator):
    MODEL_ID     = "qwen_image_edit_gguf"
    DISPLAY_NAME = "Qwen-Image-Edit (GGUF) Image Edit"
    VRAM_GB      = 8

    def is_downloaded(self):
        gguf_ok = (self.model_dir / GGUF_FILE).exists()
        decoder_ok = (self.model_dir / "model_index.json").exists()
        # The Qwen2 tokenizer needs tokenizer/merges.txt. Guard against a partial
        # download missing it so it gets re-fetched instead of failing at load.
        tok_ok = (self.model_dir / "tokenizer" / "merges.txt").exists()
        return gguf_ok and decoder_ok and tok_ok

    # ------------------------------------------------------------------ loading
    def load(self):
        mode = getattr(self, "_mem_mode", None) or "auto"

        if self._model is not None and getattr(self, "_loaded_mode", None) == mode:
            return
        if self._model is not None:
            self.unload()

        if not self.is_downloaded():
            self._download_weights()

        import torch
        from diffusers import (
            QwenImageEditPipeline,
            QwenImageTransformer2DModel,
            GGUFQuantizationConfig,
        )

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.bfloat16 if self._device == "cuda" else torch.float32

        resolved = self._resolve_mode(mode)
        print("[Qwen] loading (%s) GGUF=%s" % (resolved, GGUF_FILE))

        try:
            transformer = QwenImageTransformer2DModel.from_single_file(
                str(self.model_dir / GGUF_FILE),
                quantization_config=GGUFQuantizationConfig(compute_dtype=self._dtype),
                torch_dtype=self._dtype,
                config=str(self.model_dir),
                subfolder="transformer",
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to load the GGUF transformer (%s). This usually means diffusers is "
                "too old for Qwen GGUF single-file loading — update it in the extension venv "
                "(pip install -U diffusers), then retry." % e
            )

        # Optional 4-bit text encoder so the 7B encoder fits alongside the transformer
        # in 'balanced' mode. Best-effort: any failure falls back to the default encoder.
        text_encoder = None
        if resolved == "balanced":
            text_encoder = self._load_4bit_text_encoder(torch)

        kwargs = dict(torch_dtype=self._dtype, local_files_only=True)
        if text_encoder is not None:
            kwargs["text_encoder"] = text_encoder

        pipe = QwenImageEditPipeline.from_pretrained(
            str(self.model_dir),
            transformer=transformer,
            **kwargs,
        )

        if self._device == "cuda":
            if resolved == "max_speed":
                pipe.to("cuda")
            elif resolved == "balanced":
                pipe.enable_model_cpu_offload()
            else:  # low_vram -> per-layer streaming, smallest footprint
                pipe.enable_sequential_cpu_offload()
            try:
                pipe.vae.enable_tiling()
                pipe.vae.enable_slicing()
            except Exception:
                pass
        else:
            print("[Qwen] CUDA not available — CPU only, this will be impractically slow")
            pipe.to("cpu")

        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        self._model = pipe
        self._loaded_mode = mode
        print("[Qwen] ready on %s" % self._device)

    def _resolve_mode(self, mode):
        mode = (mode or "auto").strip().lower()
        if mode != "auto":
            return mode
        try:
            import torch
            if not torch.cuda.is_available():
                return "low_vram"
            _free_b, total_b = torch.cuda.mem_get_info()
            total_gb = total_b / (1024 ** 3)
            if total_gb >= 24:
                return "balanced"
            if total_gb >= 16:
                return "balanced"
            return "low_vram"
        except Exception:
            return "low_vram"

    def _load_4bit_text_encoder(self, torch):
        try:
            from transformers import BitsAndBytesConfig as TfBnb
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as TEClass
            except Exception:
                from transformers import AutoModelForCausalLM as TEClass

            cfg = TfBnb(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            te = TEClass.from_pretrained(
                str(self.model_dir),
                subfolder="text_encoder",
                quantization_config=cfg,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            return te
        except Exception as e:
            print("[Qwen] 4-bit text encoder unavailable (%s); using default encoder" % e)
            return None

    def unload(self):
        self._model = None
        self._device = None
        self._dtype = None
        self._loaded_mode = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # --------------------------------------------------------------- generation
    def _to_pil(self, image_in):
        from PIL import Image
        if image_in is None:
            raise ValueError("no input image — connect an Image node to the Edit node")
        if isinstance(image_in, Image.Image):
            return image_in.convert("RGB")
        if isinstance(image_in, (bytes, bytearray)):
            return Image.open(BytesIO(bytes(image_in))).convert("RGB")
        if hasattr(image_in, "__fspath__") or isinstance(image_in, str):
            return Image.open(image_in).convert("RGB")
        if hasattr(image_in, "read"):
            return Image.open(image_in).convert("RGB")
        raise ValueError("unrecognized image input type: %r" % type(image_in))

    def generate(self, image_bytes, params, progress_cb=None, cancel_event=None):
        import torch

        params = params or {}
        self._mem_mode = params.get("memory_mode") or "auto"

        if self._model is None or getattr(self, "_loaded_mode", None) != self._mem_mode:
            self.load()

        prompt = ""
        for _k in ("prompt", "text", "instruction", "negative_prompt"):
            _v = params.get(_k)
            if _v and str(_v).strip():
                prompt = str(_v).strip()
                break
        if not prompt:
            raise ValueError("no edit instruction — connect a Text node to the Edit node's text input")

        image = self._to_pil(image_bytes)

        steps     = _int(params.get("steps"), 40)
        cfg       = _float(params.get("true_cfg_scale"), 4.0)
        n_images  = max(1, min(_int(params.get("num_images"), 1), 4))
        base_seed = _int(params.get("seed"), 0)

        self._report(progress_cb, 5, "starting up")
        self._check_cancelled(cancel_event)

        if self.outputs_dir:
            out_dir = self.outputs_dir
        else:
            out_dir = self.model_dir.parent.parent.parent / "outputs" / self.MODEL_ID
        out_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        for i in range(n_images):
            self._check_cancelled(cancel_event)

            if base_seed == 0:
                seed = random.randint(1, 2**31 - 1)
            else:
                seed = base_seed + i
            gen = torch.Generator(device="cpu").manual_seed(seed)

            lo = 10 + int(85 * (i / float(n_images)))
            hi = 10 + int(85 * ((i + 1) / float(n_images)))
            label = "editing" if n_images == 1 else "editing %d/%d" % (i + 1, n_images)
            self._report(progress_cb, lo, label)

            stop = threading.Event()
            ticker = None
            if progress_cb:
                ticker = threading.Thread(
                    target=smooth_progress,
                    args=(progress_cb, lo, hi, label, stop),
                    daemon=True,
                )
                ticker.start()

            try:
                with torch.inference_mode():
                    result = self._model(
                        image=image,
                        prompt=prompt,
                        negative_prompt=" ",
                        true_cfg_scale=cfg,
                        guidance_scale=1.0,
                        num_inference_steps=steps,
                        generator=gen,
                    )
                out_img = result.images[0]
            finally:
                stop.set()
                if ticker:
                    ticker.join(timeout=1.0)

            filename = "qwen_edit_%d_%s.png" % (int(time.time()), uuid.uuid4().hex[:8])
            out_path = out_dir / filename
            out_img.save(str(out_path), format="PNG")
            paths.append(str(out_path))
            print("[Qwen] saved %s (seed %d)" % (out_path, seed))

        self._report(progress_cb, 100, "done")

        # single output -> path string (same contract as the t2i node, so the existing
        # Preview node works unchanged); multiple -> list of paths.
        if len(paths) == 1:
            return paths[0]
        return paths

    # ----------------------------------------------------------------- download
    def _auto_download(self):
        self._download_weights()

    def _download_weights(self):
        from huggingface_hub import snapshot_download, hf_hub_download

        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 1) decoder bundle (vae / text encoder / tokenizer / processor / scheduler /
        #    transformer config). Skip its transformer weight files — we supply the GGUF.
        print("[Qwen] downloading decoder bundle from %s" % DECODER_REPO)
        snapshot_download(
            repo_id=DECODER_REPO,
            local_dir=str(self.model_dir),
            ignore_patterns=[
                "transformer/*.safetensors",
                "transformer/*.bin",
                "transformer/*.gguf",
            ],
        )

        # 2) the 4-bit GGUF transformer weights
        print("[Qwen] downloading %s from %s" % (GGUF_FILE, GGUF_REPO))
        hf_hub_download(
            repo_id=GGUF_REPO,
            filename=GGUF_FILE,
            local_dir=str(self.model_dir),
        )

        print("[Qwen] download complete")
