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

`build.ps1` 用微软官方的 `vcvars64.bat` 导入整份 MSVC + Windows SDK 环境
（实测导入 156 个变量），所以 `cl.exe` / `link.exe` / `rc.exe` 全部就位 ——
不需要你先手动跑一遍 `vcvars64.bat`。脚本同时兼容 **PowerShell 7 与 Windows PowerShell 5.1**（都已实测通过）。

## 程序做了什么

`src/main.cu` 在 1M 个元素上跑两个核函数（`vectorAdd`、`saxpy`），
把 GPU 结果与 CPU 参考值逐元素比对，并用 CUDA event 测有效带宽。
全部通过时输出 `ALLOK`（CTest 用这个正则判定通过）。

本机实测：两个核函数误差均为 `0.000e+00`，有效带宽 **520–690 GB/s**。

## Python / PyTorch

```powershell
& .\.venv\Scripts\Activate.ps1
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

- Python **3.12.14**，由 `uv` 托管在 `E:\Tools\uv-python`
- torch 走官方 cu126 轮子索引（`--index-url https://download.pytorch.org/whl/cu126`）
- 其余包走清华 TUNA 镜像，缓存目录 `E:\Tools\pip-cache`

### 国内网络注意事项（本机已配置好）

这台机器上 `github.com` 被 Steam++（Watt Toolkit）的 hosts 规则指向 `127.0.0.1`，
所以任何**依赖 GitHub Releases 的下载都会卡死**。已绕过：

| 用途 | 设置 |
|---|---|
| uv 下载 Python 解释器 | `UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone` |
| npm / pnpm | `registry.npmmirror.com`（用户原有） |
| pip | 清华 TUNA |
| Playwright / Electron / node-sass | npmmirror binaries 镜像 |

若某天 GitHub 恢复直连（在 Steam++ 里启用 GitHub 加速），这些镜像可以保留，不影响正确性。

