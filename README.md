# openreframe

**Turn a finished 16:9 edit into vertical — 9:16, 4:5 and 1:1 — shot by shot, with open models on ComfyUI.**

<!-- docs/compare.gif: the original edit on the left, the vertical version on the right -->

Paid "reframe" and "expand video" features work on **one clip at a time**: Luma Reframe and Runway
Expand take 10–30 s, Runway's clip may not contain cuts, and they expand around the centre. Crop-only
auto-reframe tools keep the speaker and throw away the rest of the frame.

openreframe takes **the whole edit**: it finds every cut, decides for each shot who matters and how
much to crop, and lets the model invent only what is missing, **towards the background**. Your sound
comes through untouched.

Built for the [Comfy Developer Platform Challenge](https://blog.comfy.org/p/open-call-comfy-dev-platform-challenge)
on the official [Comfy SDK](https://docs.comfy.org/development/api-development/sdks): the same code runs
on Comfy Cloud or on your own ComfyUI.

## How it decides, shot by shot

| Step | Where | What |
|---|---|---|
| **Cuts** | local, ffmpeg `scdet` | Every hard cut. On our test edit: 23 of 23 cuts, on the exact frame. |
| **Who matters** | ComfyUI: RT-DETR + MediaPipe | People, cars, faces on every frame. Motion only breaks ties: on a tracking shot the background is what moves, and a motion-only crop drifts onto the buildings. |
| **How wide** | local | The narrowest of 3:4, 1:1 or 4:3 that holds 70 % of the action. A close-up gets a tight crop and little invention; a wide shot where the action fills the frame keeps more of it and invents more. Faces stay inside with a margin. |
| **Where to invent** | ComfyUI: Depth Anything 3 | Towards whatever is **far**. Filling the sky above a table works; filling below the table edge makes the model invent what is under it. A face never ends up below 45 % of the height, nor the real image's centre below 60 %, where platforms put captions and buttons. |
| **What to invent** | ComfyUI: Qwen3-VL | A short description of each shot becomes its prompt; one time of day for the whole edit. |
| **Invent it** | ComfyUI: Wan 2.1 VACE 14B | Outpainting with the real pixels as context, 4 steps with a step-distill LoRA. |
| **Put it back** | local | The original pixels are pasted back over the result (16 px feather), shots are joined, the original audio goes on top, and 4:5 and 1:1 are cut from the 9:16 version around the real image. |

Everything is decided per shot. You can see the decisions: `--debug-view` renders the
cuts, boxes, crop window and invented bands over your edit.

## Quick start (Comfy Cloud)

```bash
pip install git+https://github.com/perrofeo/openreframe
export COMFY_API_KEY=...          # https://platform.comfy.org — API access needs a paid Cloud plan
openreframe my_edit.mp4
```

Results land in `my_edit_openreframe/`: `my_edit_9x16.mp4`, `my_edit_4x5.mp4`, `my_edit_1x1.mp4` and
`my_edit_compare.mp4` (original and vertical side by side). The run's state is in `work/plan.json`: an
interrupted run picks up where it stopped and never submits a finished shot twice.

Options: `--ratios 9:16,4:5`, `--size 720x1280` (more detail, more GPU time), `--seed`, `--steps`,
`--redo split` (re-run a step after changing a rule).

## On your own ComfyUI

The SDK talks to a local ComfyUI through [comfy-api-proxy](https://github.com/Comfy-Org/comfy-api-proxy):

```bash
pip install comfy-api-proxy && comfy-api-proxy        # ComfyUI on :8188 -> API on :8189
export COMFY_BASE_URL=http://127.0.0.1:8189
openreframe my_edit.mp4
```

Your ComfyUI needs a recent core (RT-DETR, MediaPipe face, Depth Anything 3 and Wan VACE nodes), the
[ComfyUI-QwenVL](https://github.com/1038lab/ComfyUI-QwenVL) node pack, and these models:

| Model | File | Folder |
|---|---|---|
| Wan 2.1 VACE 14B | `wan2.1_vace_14B_fp16.safetensors` | `diffusion_models` |
| UMT5-XXL | `umt5_xxl_fp16.safetensors` | `text_encoders` |
| Wan 2.1 VAE | `wan_2.1_vae.safetensors` | `vae` |
| lightx2v step distill | `lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors` | `loras` |
| RT-DETR v4 | `rt_detr_v4-x-hgnet_fp16.safetensors` | `diffusion_models` |
| MediaPipe face | `mediapipe_face_fp32.safetensors` | `detection` |
| Depth Anything 3 Small | `depth_anything_3_small.safetensors` | `depthanything3` |
| Qwen3-VL 4B Instruct | downloaded by the node | `LLM/Qwen-VL` |

The 14B video model wants a large GPU; that is what the Cloud path is for.

## Swap the model

Every model call is a plain API-format workflow in [`openreframe/workflows/`](openreframe/workflows):
`detect.json`, `look.json` and `fill_wan_vace.json`. The pipeline only touches named nodes
(`load_video`, `pad`, `vace`, `positive`, `sampler`, `save`...), so you can replace the model behind
them — a different outpainting model, a bigger captioner, other LoRAs — without touching the Python.

## An open stack

Everything that runs is open source or open weights, commercially usable:

| Piece | License |
|---|---|
| Wan 2.1 VACE 14B, UMT5, Wan VAE | Apache-2.0 |
| lightx2v step-distill LoRA | Apache-2.0 |
| RT-DETR v4 | Apache-2.0 |
| MediaPipe face detector | Apache-2.0 |
| Depth Anything 3 Small | Apache-2.0 |
| Qwen3-VL 4B Instruct | Apache-2.0 |
| ComfyUI-QwenVL node pack | GPL-3.0 |
| Comfy SDK | MIT |
| openreframe | Apache-2.0 |

We swapped out three pieces we started with because they were not open: the CausVid LoRA
(non-commercial), insightface's face models (non-commercial research only) and a depth node pack with
no license.

## Honest limits

- **The model invents.** Most of it reads as part of the shot, but check it: in our test it once put a
  traffic light above a crossing with red and green lit at once.
- **Long shots.** v0.1 renders each shot in one window; Wan is at its best up to ~5 s. Longer shots
  still work but can soften. Overlapping windows are next.
- **Hard cuts only.** Dissolves and flashes are not cuts.
- **It reframes, it does not re-edit.** Shot order and timing are yours.

## Credits

Built by Igor Rekarte ([The Rendered](https://www.youtube.com/@the.rendered)). Demo footage: an AI-made
car chase test from The Rendered; all characters are invented.

Thanks to the teams behind ComfyUI, Wan, lightx2v, Depth Anything, Qwen, RT-DETR and MediaPipe.

## License

[Apache-2.0](LICENSE)
