# 原始测量数据

日期：2026-10-22
机器：AMD Radeon RX 9070 XT (gfx1201) / Windows 11 x64 / CPython 3.12
PyTorch：2.9.0+rocmsdk20251116，HIP 7.1.52802-561cc400e1
后端：gsv-tts-lite（GPT-SoVITS V2 权重）

## 默认（MIOpen 全量 solver 搜索，缓存目录干净）

```
[env] import 4.7s
[env] load+warmup 32.58s
[bench] 短句 7 chars  synth=6.839s  audio=2.44s  RTF=2.808
[bench] 中句 38 chars synth=15.703s audio=8.08s  RTF=1.943
```

长句没测——按中句的 RTF 推算，92 字要四十秒以上，等下去没有意义。

## MIOPEN_FIND_MODE=FAST

```
[env] import 4.66s
[env] load+warmup 8.88s
[env] vram 1.36 GB
[bench] 短句 7 chars  synth=0.710s audio=2.24s  RTF=0.317
[bench] 中句 38 chars synth=1.028s audio=8.47s  RTF=0.121
[bench] 长句 92 chars synth=2.218s audio=20.39s RTF=0.109
[bench] 流式首包 0.117s
```

短句的 RTF 明显偏高，是第一个新卷积形状的初始化成本；同一句第二次合成会降到 0.11 附近。所以生产环境里要保留一次预热。

## 首包：预热前后差 48 倍

单独探针（同一进程，加载后先不预热）：

```
[probe] load only 3.14s (no warmup)
[probe] 冷启动首包（进程刚起，未预热） 5.717s
[probe] 首次完整合成 0.984s
[probe] 同句再次首包 0.120s
[probe] 换一句首包（已预热） 0.121s
[probe] 第一句首包（第三次） 0.120s
```

| 状态 | 首包 |
|---|---|
| 进程刚起，未预热 | 5.717 s |
| 预热之后，同一句 | 0.120 s |
| 预热之后，换一句（长度相近） | 0.121 s |
| 预热之后，第三次 | 0.120 s |

结论：预热之后首包稳定在 0.12 秒上下，等于归零；那 5.7 秒是一次性的建缓存成本，必须开在启动阶段。
注意这里的「冷启动」指进程刚起但 MIOpen 缓存已在磁盘上。完全第一次跑（缓存为空）会更慢，那是上面默认组 32.58 秒那个量级。

## 对照表

| 项目 | 默认 | FAST | 倍率 |
|---|---|---|---|
| 加载 + 预热 | 32.58 s | 8.88 s | 3.7× |
| 短句 7 字 | 6.839 s | 0.710 s | 9.6× |
| 中句 38 字 | 15.703 s | 1.028 s | 15.3× |
| 长句 92 字 | — | 2.218 s | — |
| 首包（未预热） | — | 5.717 s | — |
| 首包（预热后） | — | 0.120 s | 48× |
