# cuda-lab

端到端工具链验证工程：**git + CMake + Ninja + nvcc + MSVC + CTest** 一条链压通。

同时它也是一个 Python/CUDA 实验场（`.venv` 里装了带 CUDA 12.6 的 PyTorch）。

## 环境（本机实测通过）

| 组件 | 版本 | 路径 |
|---|---|---|
| CUDA / nvcc | 12.6.85 | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6` |
| MSVC | 14.44 (VS Build Tools 2022 17.14) | `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools` |
| Windows SDK | 10.0.26100.0 | `E:\Windows Kits\10` |
| CMake / Ninja | 4.4.3 / 1.13.2 | `E:\Tools\buildchain` |
| GPU | RTX 4060 Laptop 8GB (sm_89) | 驱动 596.36 |

## 构建

```powershell
pwsh -File .\build.ps1            # 配置 + 编译 + 运行
pwsh -File .\build.ps1 -Clean     # 先清 build/
pwsh -File .\build.ps1 -Test      # 额外跑 ctest
```

`build.ps1` 会自动把 MSVC 与 Windows SDK 的 `INCLUDE`/`LIB` 注入当前进程——
这是 `nvcc` 找到宿主编译器 `cl.exe` 的前提，所以不需要先手动跑 `vcvars64.bat`。

## 程序做了什么

`src/main.cu` 在 1M 个元素上跑两个核函数（`vectorAdd`、`saxpy`），
把 GPU 结果与 CPU 参考值逐元素比对，并用 CUDA event 测有效带宽。
全部通过时输出 `ALLOK`（CTest 用这个正则判定通过）。

## Python / PyTorch

```powershell
& .\.venv\Scripts\Activate.ps1
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

torch 来自官方 cu126 轮子索引；其余包走清华 TUNA 镜像（`E:\Tools\pip-cache` 为缓存目录）。
