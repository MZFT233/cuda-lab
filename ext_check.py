r"""Build and exercise an inline CUDA extension through torch.utils.cpp_extension.

This is the harshest end-to-end test of the toolchain: it needs Python headers,
cl.exe, nvcc and the MSVC linker to cooperate in a single compile+link step.

Run:  & .\.venv\Scripts\python.exe -X utf8 ext_check.py

Two environment quirks are handled automatically (set CUDALAB_TRACEBACK=1 to see
full tracebacks, CUDALAB_FIX_OEM=0 to disable the code-page workaround):

1. This machine's ANSI/OEM code page is 936, and CPython on Windows ships no 'oem'
   mapping for 936. torch's cpp_extension hardcodes SUBPROCESS_DECODE_ARGS=('oem',)
   and decodes cl.exe's localized help text with it, which raises
   UnicodeDecodeError before the real build even starts. See the patch below.
2. Compiler diagnostics on a code-page-936 machine are GBK, so passing -X utf8
   keeps our own printing sane.
"""
import os
import sys
from pathlib import Path

if not sys.flags.utf8_mode:
    print("WARNING: UTF-8 mode is off. Compiler diagnostics may raise UnicodeDecodeError.", file=sys.stderr)
    print("         Re-run as: python -X utf8 ext_check.py", file=sys.stderr)
    print(file=sys.stderr)

# --- avoid the 'oem' codec ---------------------------------------------------
# torch's cpp_extension decodes compiler output through SUBPROCESS_DECODE_ARGS,
# which on Windows it derives from locale.getpreferredencoding(). This machine's
# ANSI/OEM code page is 936, and CPython on Windows ships no 'oem' mapping for 936,
# so any compiler diagnostic blows up with:
#     UnicodeDecodeError: 'cp1' codec ... decoding with 'oem' codec failed
# Reporting UTF-8 as the preferred encoding before torch is imported makes torch
# decode with plain UTF-8 instead. That never raises on these byte strings, and the
# decoded text is only used for an ASCII version regex.
import locale as _locale

if os.name == "nt" and os.environ.get("CUDALAB_FIX_OEM", "1") == "1":
    _locale.getpreferredencoding = lambda do_setlocale=True: "utf-8"

# --- locate MSVC + Windows SDK, so cpp_extension can find cl.exe -------------
# torch's cpp_extension honours DISTUTILS_USE_SDK / MSSdk, so we can point it at
# the toolset directly instead of requiring vcvars64.bat to have been sourced.
VS_ROOT = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools")
SDK_ROOT = Path(r"E:\Windows Kits\10")

def setup_msvc_env() -> bool:
    msvc_root = VS_ROOT / "VC" / "Tools" / "MSVC"
    if not msvc_root.is_dir():
        return False
    msvc = sorted(msvc_root.iterdir(), key=lambda p: p.name)[-1]
    host = msvc / "bin" / "Hostx64" / "x64"

    os.environ["DISTUTILS_USE_SDK"] = "1"
    os.environ["MSSdk"] = "1"
    os.environ["VSCMD_ARG_TGT_ARCH"] = "x64"

    # Force English (US) diagnostics. This machine's ANSI code page is 936, so
    # cl.exe/link.exe would otherwise emit GBK text that torch's cpp_extension
    # fails to decode (it assumes UTF-8), raising UnicodeDecodeError instead of
    # showing the real compiler error.
    os.environ["VSLANG"] = "1033"

    os.environ["PATH"] = f"{host};" + os.environ.get("PATH", "")

    inc = [str(msvc / "include")]
    lib = [str(msvc / "lib" / "x64")]
    sdk_inc = SDK_ROOT / "Include"
    sdk_lib = SDK_ROOT / "Lib"
    if sdk_inc.is_dir():
        ver = sorted(p.name for p in sdk_inc.iterdir() if p.name.startswith("10."))[-1]
        for sub in ("ucrt", "um", "shared", "winrt"):
            d = sdk_inc / ver / sub
            if d.is_dir():
                inc.append(str(d))
        for sub in ("ucrt/x64", "um/x64"):
            d = sdk_lib / ver / sub
            if d.is_dir():
                lib.append(str(d))
        os.environ["WindowsSdkDir"] = str(SDK_ROOT) + os.sep
        os.environ["WindowsSDKVersion"] = ver + os.sep
        # nvcc receives INCLUDE/LIB verbatim -> forward slashes survive quoting
        os.environ["INCLUDE"] = ";".join(p.replace("\\", "/") for p in inc)
        os.environ["LIB"] = ";".join(p.replace("\\", "/") for p in lib)
        os.environ["LIBPATH"] = os.environ["LIB"]

    os.environ["CUDA_HOME"] = os.environ.get("CUDA_HOME", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6")
    print(f"MSVC      : {msvc.name}")
    print(f"cl.exe    : {host / 'cl.exe'}  exists={(host / 'cl.exe').exists()}")
    return True


print("=" * 68)
print("torch.utils.cpp_extension inline CUDA build test")
print("=" * 68)
if not setup_msvc_env():
    print("FAILED: MSVC toolset not found", file=sys.stderr)
    sys.exit(2)

import torch
from torch.utils.cpp_extension import load_inline
import torch.utils.cpp_extension as _ce

# ---------------------------------------------------------------------------
# Patch the hardcoded 'oem' decode.
#
# torch/utils/cpp_extension.py contains, at module level:
#     SUBPROCESS_DECODE_ARGS = ('oem',) if IS_WINDOWS else ()
# and later does  compiler_info.decode(*SUBPROCESS_DECODE_ARGS)  on the output of
# running cl.exe with no arguments. That output is localized Chinese text encoded
# in this machine's code page 936, and CPython on Windows ships NO 'oem' mapping
# for 936 - so the decode raises:
#     UnicodeDecodeError: 'cp1' codec ... decoding with 'oem' codec failed
#
# The decoded string is only fed to re.search(r'(\d+)\.(\d+)\.(\d+)'), i.e. an
# ASCII version number, so decoding as UTF-8 with 'replace' is both safe and
# sufficient. ('replace' also avoids a second failure mode: nvcc mixes UTF-8 and
# code-page bytes in one stream.)
# ---------------------------------------------------------------------------
CE_DECODE_ARGS = ("utf-8", "replace") if os.name == "nt" else ()

if os.name == "nt":
    _ce.SUBPROCESS_DECODE_ARGS = CE_DECODE_ARGS

# ---------------------------------------------------------------------------
# Decoding shim.
#
# torch's cpp_extension captures compiler output with an explicit utf-8 decode.
# On a machine whose ANSI code page is 936, cl.exe/link.exe emit GBK bytes, so any
# compiler diagnostic raises UnicodeDecodeError before the real message is shown.
# The code page cannot be switched for the child (chcp would do it, but plumbing
# that through torch is fragile), so soften the decode instead: invalid bytes are
# replaced rather than raising. Scoped to this process only.
# ---------------------------------------------------------------------------
import subprocess as _sp

if os.environ.get("CUDALAB_SOFTEN_DECODE", "1") == "1":
    _orig_run = _sp.run

    def _tolerant_run(*args, **kwargs):
        try:
            return _orig_run(*args, **kwargs)
        except UnicodeDecodeError:
            kwargs["errors"] = "replace"
            return _orig_run(*args, **kwargs)

    _sp.run = _tolerant_run
    # torch may have already bound the name into its module namespace
    try:
        import torch.utils.cpp_extension as _ce
        if getattr(_ce, "subprocess", None) is not None:
            _ce.subprocess.run = _tolerant_run
    except Exception:
        pass

print(f"torch     : {torch.__version__}")
print(f"cuda      : {torch.version.cuda}   available={torch.cuda.is_available()}")
print(f"arch list : {torch.cuda.get_arch_list()}")
print()

cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scale_add_kernel(const float* x, float* y, float a, float b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + b;
}

// (a) elementwise, exercises a plain kernel
torch::Tensor scale_add(torch::Tensor x, double a, double b) {
  TORCH_CHECK(x.is_cuda(), "input must be a CUDA tensor");
  auto xc = x.contiguous();
  auto y = torch::empty_like(xc);
  int n = xc.numel();
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  scale_add_kernel<<<blocks, threads>>>(xc.data_ptr<float>(), y.data_ptr<float>(),
                                        (float)a, (float)b, n);
  return y;
}

// (b) reduction with atomics + shared memory, exercises a second kernel shape
__global__ void sum_kernel(const float* x, float* out, int n) {
  extern __shared__ float s[];
  int t = threadIdx.x;
  int i = blockIdx.x * blockDim.x + t;
  s[t] = (i < n) ? x[i] : 0.f;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (t < stride) s[t] += s[t + stride];
    __syncthreads();
  }
  if (t == 0) atomicAdd(out, s[0]);
}

torch::Tensor device_sum(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "input must be a CUDA tensor");
  auto xc = x.contiguous();
  auto out = torch::zeros({}, xc.options());
  int n = xc.numel();
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(xc.data_ptr<float>(), out.data_ptr<float>(), n);
  return out;
}
"""

cpp_src = """
torch::Tensor scale_add(torch::Tensor x, double a, double b);
torch::Tensor device_sum(torch::Tensor x);
"""

build_dir = Path(__file__).parent / "build_ext"
build_dir.mkdir(exist_ok=True)

# -ccbin pins the host compiler: nvcc's own PATH lookup is unreliable here,
# exactly the failure mode that broke the CMake build before we pinned it.
ccbin = str(Path(os.environ["PATH"].split(";")[0]))

print("building extension (this invokes cl.exe + nvcc) ...")
try:
    mod = load_inline(
        name="cudalab_ext",
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=["scale_add", "device_sum"],
        extra_cuda_cflags=["-O2", "-ccbin", ccbin],
        extra_cflags=["/O2"],
        build_directory=str(build_dir),
        verbose=False,
    )
except Exception as exc:  # noqa: BLE001
    if os.environ.get("CUDALAB_TRACEBACK") == "1":
        import traceback

        traceback.print_exc()
    print(f"\nBUILD FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(1)

print("build OK\n")

failures = []


def check(name, ok, detail=""):
    print(f"{name:<26}: {'OK' if ok else 'FAIL'} {detail}")
    if not ok:
        failures.append(name)


x = torch.randn(1 << 20, device="cuda")
y = mod.scale_add(x, 2.5, 1.0)
ref = 2.5 * x + 1.0
check("scale_add correctness", torch.allclose(y, ref, atol=1e-4), f"max|d|={(y - ref).abs().max().item():.2e}")

s = mod.device_sum(x)
check("device_sum correctness", torch.allclose(s, x.sum(), rtol=1e-3), f"gpu={s.item():.3f} cpu={x.sum().item():.3f}")

# a second call proves the module stays usable, not a one-shot
y2 = mod.scale_add(y, 0.0, 7.0)
check("second invocation", bool((y2 == 7.0).all()), "all elements == 7.0")

print()
if failures:
    print("EXTENSION CHECK FAILED:", ", ".join(failures))
    sys.exit(1)
print("EXTENSION CHECK PASSED")
