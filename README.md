# 在 Windows + RX 9070 XT（gfx1201）上把 GPT-SoVITS 推理加速 9–15 倍

一份实测记录：Windows 原生 ROCm 跑 GPT-SoVITS / GSV-TTS-Lite 时，默认配置有多慢、为什么慢、以及一个环境变量能快多少。

实测日期 2026-10-22。**同一台机器、同一个模型、同一段文本，只改 MIOpen 的环境变量。**

## 结论先行

| 项目 | 默认 | `MIOPEN_FIND_MODE=FAST` | 提升 |
|---|---|---|---|
| 加载 + 预热 | 32.58 s | 8.88 s | 3.7× |
| 短句（7 字） | 6.839 s｜RTF 2.81 | 0.710 s｜RTF 0.32 | 9.6× |
| 中句（38 字） | 15.703 s｜RTF 1.94 | 1.028 s｜RTF 0.12 | 15.3× |
| 长句（92 字） | 未测（预计 >40 s） | 2.218 s｜RTF 0.11 | — |
| 首包（进程刚起，未预热） | — | 5.717 s | — |
| 首包（预热之后，稳态） | — | **0.120 s** | 48× |
| 显存占用 | — | 1.36 GB | — |

RTF = 合成耗时 / 音频时长，越小越快。0.11 意味着生成 1 秒语音只要 0.11 秒算力。

## 环境

| | |
|---|---|
| GPU | AMD Radeon RX 9070 XT（gfx1201，RDNA4） |
| 系统 | Windows 11 x64 |
| Python | CPython 3.12 |
| PyTorch | `2.9.0+rocmsdk20251116` |
| HIP | `7.1.52802-561cc400e1` |
| 推理后端 | `gsv-tts-lite`（GPT-SoVITS 的高性能推理实现） |
| 模型 | GPT-SoVITS V2（SoVITS + GPT，约 1.36 GB 显存） |

Windows 上的 AMD 支持只覆盖 PyTorch 部分，不是完整 ROCm 栈：<https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/windows/install-pytorch.html>

## 为什么慢

MIOpen 在 Windows 原生 ROCm 上，对**每个新的卷积形状**都会跑一次全量 solver 穷举，而且这次穷举的结果**没有被缓存下来**——日志里能看到 `Invalid elapsed time detected in EvaluateInvokers`，下一次推理同样的形状，再从头搜一遍。

ROCm 仓库里有同样的报告（gfx1100）：

> <https://github.com/ROCm/TheRock/issues/7958>
> "find results never cached, full solver search re-runs on every inference"

TTS 推理的卷积形状多、句子长短又会让形状变化，于是几乎每句话都在付这份穷举成本。表现就是：GPU 风扇在转、利用率不高、但一句话要等十几秒。

## 解法

三个环境变量，写在启动脚本里，**进程启动前**设置：

```powershell
$env:MIOPEN_FIND_MODE        = "FAST"                       # 只走快速启发式，不穷举
$env:MIOPEN_USER_DB_PATH     = "D:\\ai\\miopen_db"          # 缓存写到固定可写目录
$env:MIOPEN_CUSTOM_CACHE_DIR = "D:\\ai\\miopen_db"
```

`MIOPEN_FIND_MODE=FAST` 是关键。后两个是保险：默认缓存路径可能落在没有写权限的地方，缓存失效就退化成默认行为。

PowerShell 里用 `setdefault` 语义（Python 侧）：

```python
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
os.environ.setdefault("MIOPEN_USER_DB_PATH", r"D:\ai\miopen_db")
os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", r"D:\ai\miopen_db")
```

必须在 `import torch` 之前设置。

## 这不是 9070 XT 专属

`MIOPEN_FIND_MODE` 是 MIOpen 自己的环境变量，不挑型号。官方文档里的取值：

| 值 | 含义 |
|---|---|
| `NORMAL` / 1 | 全量搜索，benchmark 所有 solver |
| `FAST` / 2 | 用 FindDb 命中，没有就立即回退 |
| `HYBRID` / 3 | 命中就用，没命中才全量搜索 |
| `DYNAMIC_HYBRID` / 5 | **默认值**，跳过非动态 kernel |

任何走 MIOpen 的 AMD 卡都能设，RDNA2 / 3 / 4 都行。ComfyUI 甚至在 [#10448](https://github.com/Comfy-Org/ComfyUI/pull/10448) 里直接对所有 AMD 卡无条件设成 FAST，没做型号判断：

```python
if is_amd():
    if os.getenv('MIOPEN_FIND_MODE') is None:
        os.environ['MIOPEN_FIND_MODE'] = "FAST"
```

**收益大小取决于那张卡上 find 缓存是不是也失效。** 本文这台是 gfx1201；同一个毛病在 gfx1100（RDNA3）上也有报告（TheRock#7958），所以不是新架构独有的。但具体快多少，只有拿 `benchmark.py` 在你自己卡上跑两组才知道——可能 15 倍，也可能只是小幅改善。

**代价**：FAST 跳过穷举，可能选到次优 kernel，稳态性能理论上不如全量搜索找到的最优解。在这台机器上这个权衡没有讨论余地——默认模式一句 15 秒，根本不能用。如果你的卡上默认模式已经够快，就别急着改。

## 复现

```powershell
python benchmark.py --models D:\models\my_sovits.pth,D:\models\my_gpt.ckpt --ref D:\refs\sample.wav
```

脚本会分别测「加载」「预热」「短句」「中句」「长句」，并把两轮（FAST / 默认）的结果打进 `results.json`。想单独量首包，见 `results.md` 里的 `_probe_first` 记录；那个脚本会分别在「未预热」「预热后同句」「预热后换句」三种状态下取第一个 chunk 的时间。默认那一轮请用干净的 `MIOPEN_CUSTOM_CACHE_DIR`，否则会命中上一轮留下的缓存、看不出差异。

## 另外几个坑

**1. numpy 版本打架。** `pip install gsv-tts-lite` 会顺手把 numpy 升到 2.x，而 `pyopenjtalk`（中日文分词用）在 numpy 2.x 下会崩。必须钉回：

```powershell
pip install "numpy==1.26.4"
```

**2. py3langid 在 numpy 1.26 下误判语言。** `gsv-tts-lite` 依赖的 `py3langid 0.4` 要求 numpy≥2，降级之后它的语言检测会错乱，把整句判成非中文然后直接中止。绕过办法是覆盖它的分段函数：

```python
from gsv_tts.LangSegment import LangSegment
LangSegment.getTexts = staticmethod(lambda text: [{"lang": "zh", "text": text}])
```

只跑中文时这样最省事；多语言场景需要另想办法。

**3. venv 的 python.exe 会 re-exec 基础解释器。** 任务管理器里同一次逻辑运行会显示两个 PID（venv launcher → base python，父子关系）。排查「怎么跑了两个进程」时先看 ParentProcessId，否则会误判成两份服务在抢显卡。

**4. 预热不是优化项，是必需项。** 进程刚起来、什么都没合成过的时候，第一个 chunk 要 **5.717 秒**——MIOpen 要为这一批卷积形状建缓存。合成过一句之后，同一句再回来只要 0.120 秒，换一句新的也还是 0.121 秒。

| 状态 | 首包 |
|---|---|
| 进程刚起，未预热 | 5.717 s |
| 预热之后，同一句 | 0.120 s |
| 预热之后，换一句 | 0.121 s |

也就是说：**预热过之后首包基本归零**，但这笔账必须开在启动那一次上。做法是在加载完模型后立刻合成一句废稿（比如「嗯。」），别拿它当实际输出的第一句。

顺带解释上面那个 8.88 s 的「加载 + 预热」是怎么来的：纯加载 3.14 s，预热那一句 5.7 s，加起来正好。

## 已知限制

- 长句依然受 decode 步数限制：92 字 ≈ 20 秒音频要 2.2 秒算力。要更快的首响，得在**句子层面**切短，靠并行流水线而不是单句加速。
- 这是 Windows 原生 ROCm 的路径，Linux + ROCm 的表现可能不同（MIOpen 在 Linux 上通常缓存正常）。
- 数字只代表这张卡 + 这套版本。换 ROCm 版本、换模型，绝对值会变，倍率量级应该还在。

## 参考

- [ROCm/TheRock#7958 — MIOpen find results never cached on Windows](https://github.com/ROCm/TheRock/issues/7958)
- [ComfyUI#10448 — Default MIOPEN_FIND_MODE=FAST for AMD](https://github.com/Comfy-Org/ComfyUI/pull/10448)
- [AMD — PyTorch on Windows via PIP](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/windows/install-pytorch.html)
- [chinokikiss/GSV-TTS-Lite](https://github.com/chinokikiss/GSV-TTS-Lite)
- [Code-Amadeus/Amadeus — Windows ROCm sidecar（含 GPT-SoVITS sidecar 的另一个实现）](https://github.com/Code-Amadeus/Amadeus/tree/main/tools/rocm_sidecar)

## License

MIT。数据是实测的，结论自己验证过再信。
