# 9–15× faster GPT-SoVITS inference on Windows + RX 9070 XT (gfx1201)

What the default MIOpen configuration costs on Windows native ROCm, why it costs that, and what one environment variable buys you.

Measured 2026-10-22 on a single machine, same model, same text — only the MIOpen settings differ.

## Results

| | default | `MIOPEN_FIND_MODE=FAST` | speed-up |
|---|---|---|---|
| Load + warm-up | 32.58 s | 8.88 s | 3.7× |
| Short (7 chars) | 6.839 s · RTF 2.81 | 0.710 s · RTF 0.32 | 9.6× |
| Medium (38 chars) | 15.703 s · RTF 1.94 | 1.028 s · RTF 0.12 | 15.3× |
| Long (92 chars) | not measured (>40 s est.) | 2.218 s · RTF 0.11 | — |
| First chunk (cold process, no warm-up) | — | 5.717 s | — |
| First chunk (after warm-up, steady) | — | **0.120 s** | 48× |
| VRAM | — | 1.36 GB | — |

RTF = synth time / audio duration. 0.11 means one second of speech costs 0.11 s of compute.

## Environment

| |
|---|
| GPU: AMD Radeon RX 9070 XT (gfx1201, RDNA4) |
| OS: Windows 11 x64, CPython 3.12 |
| PyTorch: `2.9.0+rocmsdk20251116`, HIP `7.1.52802-561cc400e1` |
| Backend: `gsv-tts-lite` (a faster GPT-SoVITS inference implementation) |
| Model: GPT-SoVITS V2, ~1.36 GB VRAM |

Windows ROCm ships PyTorch support, not the whole ROCm stack: <https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/windows/install-pytorch.html>

## Why it is slow

On Windows native ROCm, MIOpen runs a **full solver search for every new convolution shape**, and the result **is not cached** — you can see `Invalid elapsed time detected in EvaluateInvokers` in the log, and the next inference with the same shape searches all over again.

Same report upstream, on gfx1100: <https://github.com/ROCm/TheRock/issues/7958>

TTS inference produces many shapes, and sentence length keeps changing them, so nearly every utterance pays the full search cost. The symptom is odd: GPU fans spin, utilisation stays low, but a single sentence takes ten-plus seconds.

## Fix

Set **before** the process starts (before `import torch`):

```powershell
$env:MIOPEN_FIND_MODE        = "FAST"
$env:MIOPEN_USER_DB_PATH     = "D:\\ai\\miopen_db"
$env:MIOPEN_CUSTOM_CACHE_DIR = "D:\\ai\\miopen_db"
```

`MIOPEN_FIND_MODE=FAST` is the one that matters. The two path variables are belt-and-braces: if the default cache location is not writable the cache silently stops working and you are back to full searches.

In Python, before importing torch:

```python
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
os.environ.setdefault("MIOPEN_USER_DB_PATH", r"D:\ai\miopen_db")
os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", r"D:\ai\miopen_db")
```

## Reproduce

```powershell
python benchmark.py --sovits model.pth --gpt model.ckpt --ref ref.wav --ref-text "..." --label fast
```

Run it twice: once with `FAST`, once with a clean `MIOPEN_CUSTOM_CACHE_DIR` and no `FAST`. The second run must use a fresh cache directory or it will hit the previous run's cached kernels and the difference disappears.

## Other traps

1. **numpy version conflict.** `pip install gsv-tts-lite` pulls numpy 2.x, and `pyopenjtalk` crashes on it. Pin back with `pip install "numpy==1.26.4"`.
2. **py3langid misdetects under numpy 1.26.** It wants numpy≥2; after the downgrade its language detection misfires and aborts whole utterances. Override the segmenter for Chinese-only use:
   ```python
   from gsv_tts.LangSegment import LangSegment
   LangSegment.getTexts = staticmethod(lambda text: [{"lang": "zh", "text": text}])
   ```
3. **A venv's `python.exe` re-execs the base interpreter.** One logical run shows up as two PIDs (venv launcher → base python). Check `ParentProcessId` before concluding two services are fighting over the GPU.
4. **Warm-up is mandatory, not an optimisation.** A freshly started process spends **5.717 s** on its first chunk — MIOpen is building the cache for that whole set of convolution shapes. Once one sentence has been synthesised, the same sentence comes back in 0.120 s, and a different (similar-length) sentence in 0.121 s.

| State | First chunk |
|---|---|
| Cold process, no warm-up | 5.717 s |
| After warm-up, same sentence | 0.120 s |
| After warm-up, different sentence | 0.121 s |

In other words: **after warm-up the first chunk is effectively zero** — but that bill has to be paid once at startup. Synthesise a throwaway sentence (e.g. "嗯。") right after loading the models; never make it the first thing a user hears.

That also explains the 8.88 s "load + warm-up" above: 3.14 s of pure loading plus 5.7 s of warm-up.

## Limits

- Long sentences are still bounded by decode steps: 92 chars ≈ 20 s of audio takes 2.2 s. For lower latency, cut at sentence level and pipeline rather than trying to make one call faster.
- This is the Windows native ROCm path. Linux + ROCm behaves differently (MIOpen caching usually works there).
- Numbers are specific to this card and these versions. Absolute values will move; the order of magnitude should hold.

## References

- [ROCm/TheRock#7958](https://github.com/ROCm/TheRock/issues/7958) — MIOpen find results never cached on Windows
- [ComfyUI#10448](https://github.com/Comfy-Org/ComfyUI/pull/10448) — default MIOPEN_FIND_MODE=FAST for AMD
- [AMD — PyTorch on Windows via PIP](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/windows/install-pytorch.html)
- [chinokikiss/GSV-TTS-Lite](https://github.com/chinokikiss/GSV-TTS-Lite)
- [Code-Amadeus/Amadeus — Windows ROCm sidecar](https://github.com/Code-Amadeus/Amadeus/tree/main/tools/rocm_sidecar)

## License

MIT. Numbers are measured; verify before trusting.
