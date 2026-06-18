# Qwen-Image-Edit (GGUF) Image Edit — Modly Extension

Instruction-based image editing with Alibaba's **Qwen-Image-Edit**, loaded as a 4-bit **GGUF** transformer so it runs on consumer GPUs. Give it one image and a text instruction and it returns edited image(s) — strong at novel viewpoints (rotate an object to show its back/side), style changes, object add/remove, and background swaps.

**Apache 2.0 and ungated** — no HuggingFace login, no token, no license click. The download just works.

---

## VRAM reality (read this first)

Qwen-Image-Edit is a 20B model with a separate ~7B text encoder, so even quantized it leans on offloading:

- **~12GB:** use **Low VRAM** mode. It works, but streams weights layer-by-layer from system RAM, so expect **minutes per image**.
- **~6-8GB:** also possible in Low VRAM mode, slower still.
- **16GB+:** **Balanced** mode is faster (4-bit text encoder + model offload).
- **24GB+:** **Max Speed** keeps everything on the GPU.

**System RAM matters as much as VRAM** — the offloaded weights live there. 32GB+ is strongly recommended; with less you may hit out-of-memory or heavy swapping.

---

## Installation

1. Open Modly → **Extensions** → **Install from GitHub** and paste this repo URL.
2. Wait for setup (installs PyTorch, diffusers, the `gguf` loader, and bitsandbytes into an isolated venv).
3. Click **Download** on the Edit Image node. It pulls two ungated repos: the 4-bit GGUF transformer (`calcuis/qwen-image-edit-gguf`) and the decoder bundle (`callgg/image-edit-decoder`, which holds the text encoder + VAE).

If the model fails to load with a `from_single_file` error, your diffusers is too old for Qwen GGUF loading — update it in the extension venv (`pip install -U diffusers`) and retry. This is bleeding-edge; the first run on a given machine sometimes needs that bump.

---

## Usage (Workflows tab)

1. Drag an **Image** node and point it at the image you want to edit.
2. Drag a **Text** node and type your instruction (e.g. `rotate the object 180 degrees to show the back`, `three-quarter side view`, `change the background to a sunset beach`).
3. Drag an **Edit Image** node. Connect the **Image** node into its image input and the **Text** node into its text input.
4. Connect the **Edit Image** output into your **Preview Image** node.
5. Hit **Run**.

Number of Outputs = 1 emits a single image (same as the text-to-image node). Set 2–4 for multiple variations in one run (each uses a different seed); the node then emits a list of images.

### Prompt tips
Qwen-Image-Edit follows explicit instructions well and preserves the subject's identity. Describe the change directly, make one change at a time, and for viewpoints be specific ("rotate 90° left", "view from above").

---

## Parameters

| Parameter | Default | Notes |
|---|---|---|
| Memory Mode | Auto | Loading/offload strategy (see below) |
| Steps | 40 | 40–50 is typical; higher = slower |
| CFG Scale | 4.0 | Prompt adherence (`true_cfg_scale`); ~2.5–4 works well |
| Number of Outputs | 1 | 1–4 variations per run |
| Seed | 0 | 0 = random each run; fixed = reproducible |

### Memory Mode

| Mode | What it does | Rough VRAM |
|---|---|---|
| Auto | < 16GB → Low VRAM, otherwise Balanced | — |
| Low VRAM | Per-layer streaming offload (slow, smallest footprint) | ~6–12GB |
| Balanced | 4-bit text encoder + model offload | ~16GB+ |
| Max Speed | Everything on GPU, no offload | ~24GB+ |

Balanced needs `bitsandbytes` (installed at setup). If it's missing, Balanced falls back to the default encoder; Low VRAM is unaffected.

---

## Notes

- The default transformer is the `qwen-image-edit-iq4_nl.gguf` 4-bit quant. To trade size/quality, swap `GGUF_FILE` in `generator.py` for another quant from the [GGUF repo](https://huggingface.co/calcuis/qwen-image-edit-gguf/tree/main) (smaller Q3/Q2 = lighter/faster but more artifacts).
- Want real speed at low VRAM? **Nunchaku** ships 4-bit Qwen-Image-Edit with optimized CUDA kernels and a diffusers-compatible API (per-layer offload down to ~3GB). It's an extra, version-matched install, so it's left out of the default build — ask if you'd like a Nunchaku variant.

---

## License

Extension code is **MIT** — see [LICENSE](LICENSE). The Qwen-Image-Edit weights it downloads are **Apache 2.0** (Alibaba's Qwen team), which permits commercial use. This is an independent community extension, not affiliated with Alibaba, Hugging Face, or Modly; each user is responsible for complying with the model license and for whatever they generate. Provided "as is", without warranty.
