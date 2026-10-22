# -*- coding: utf-8 -*-
"""Benchmark GPT-SoVITS inference on a local GPU.

两次运行才能看出 MIOpen 的差别：一次带 FAST，一次不带、并且用干净的缓存目录。
环境变量必须在 import torch 之前设置，所以这里不接受运行中途切换。

    # 默认（慢）
    $env:MIOPEN_FIND_MODE = ""            # 或者干脆 unset
    $env:MIOPEN_CUSTOM_CACHE_DIR = "D:\tmp\miopen_clean"
    python benchmark.py --sovits ... --gpt ... --ref ref.wav --ref-text "..." --label default

    # FAST
    $env:MIOPEN_FIND_MODE = "FAST"
    python benchmark.py --sovits ... --gpt ... --ref ref.wav --ref-text "..." --label fast
"""
import argparse
import json
import os
import time

TEXTS = [
    ("短句", "嗯，我知道了。"),
    ("中句", "这句话大概三十个字左右，用来看看合成的耗时是不是稳定在一个可以接受的范围内。"),
    ("长句", "夜色沉下来的时候，整座设施只剩下通风管道的低鸣，她站在观测窗前，把额头轻轻贴上冰凉的玻璃，"
             "看着外面那座永远不会有人走上去的舷梯，想着明天还要再做一次同样的检查，同样的记录，同样的回答。"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sovits", required=True, help="SoVITS 权重 .pth")
    ap.add_argument("--gpt", required=True, help="GPT 权重 .ckpt")
    ap.add_argument("--ref", required=True, help="参考音频 wav")
    ap.add_argument("--ref-text", default="", help="参考音频对应的文本")
    ap.add_argument("--ref-lang", default="ja", help="参考音频语言，默认 ja")
    ap.add_argument("--lang", default="zh", help="合成文本语言，默认 zh")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    # 提前落位，赶在 torch 初始化之前
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

    t = time.time()
    import numpy as np
    from gsv_tts import TTS

    tts = TTS(use_bert=True, sovits_cache=[50, 55])
    tts.load_gpt_model(args.gpt)
    tts.load_sovits_model(args.sovits)
    load = time.time() - t

    # 预热：第一个没见过的卷积形状要付一次初始化成本，不然短句的数字会被它污染
    warm_t = time.time()
    list(tts.infer_stream(spk_audio_path=args.ref, prompt_audio_path=args.ref,
                          prompt_audio_text=args.ref_text, text="嗯", text_language=args.lang,
                          prompt_language=args.ref_lang, temperature=0.9))
    warm = time.time() - warm_t

    rows = []
    for name, text in TEXTS:
        t = time.time()
        parts, sr = [], 32000
        for clip in tts.infer_stream(spk_audio_path=args.ref, prompt_audio_path=args.ref,
                                     prompt_audio_text=args.ref_text, text=text,
                                     text_language=args.lang, prompt_language=args.ref_lang,
                                     temperature=0.9, stream_chunk=25, overlap_len=5):
            a = np.asarray(clip.audio_data, dtype="float32")
            if a.ndim > 1:
                a = a.mean(axis=1)
            parts.append(a)
            sr = clip.samplerate
        dt = time.time() - t
        dur = float(np.concatenate(parts).shape[0]) / sr if parts else 0.0
        rows.append({"case": name, "chars": len(text), "synth_s": round(dt, 3),
                     "audio_s": round(dur, 2), "rtf": round(dt / dur, 3) if dur else None})

    result = {"label": args.label, "load_s": round(load, 2), "warmup_s": round(warm, 2),
              "find_mode": os.environ.get("MIOPEN_FIND_MODE", "(default)"), "rows": rows}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    for r in rows:
        print("%s %d chars synth=%.3fs audio=%.2fs RTF=%s" % (r["case"], r["chars"], r["synth_s"], r["audio_s"], r["rtf"]))
    print("load=%.2fs warmup=%.2fs mode=%s -> %s" % (load, warm, result["find_mode"], args.out))


if __name__ == "__main__":
    main()
