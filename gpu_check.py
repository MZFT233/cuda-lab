"""GPU sanity check for the cuda-lab environment.

Run:  & .\.venv\Scripts\python.exe gpu_check.py
Exits non-zero if CUDA is unavailable or results are wrong.
"""
import sys
import time

try:
    import torch
except ImportError:
    print("torch is not installed in this interpreter.", file=sys.stderr)
    print("Install it with:", file=sys.stderr)
    print("  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126", file=sys.stderr)
    sys.exit(2)

print(f"torch          : {torch.__version__}")
print(f"cuda build     : {torch.version.cuda}")
print(f"cudnn          : {torch.backends.cudnn.version()}")

if not torch.cuda.is_available():
    print("CUDA is NOT available.", file=sys.stderr)
    sys.exit(1)

dev = torch.cuda.current_device()
props = torch.cuda.get_device_properties(dev)
print(f"device         : {props.name}")
print(f"capability     : sm_{props.major}{props.minor}")
print(f"vram           : {props.total_memory / 1024**3:.1f} GiB")

failures = []


def check(name, ok, detail=""):
    print(f"{name:<15}: {'OK' if ok else 'FAIL'} {detail}")
    if not ok:
        failures.append(name)


# 1) elementwise on GPU
a = torch.randn(1 << 20, device="cuda")
b = torch.randn(1 << 20, device="cuda")
check("add", torch.allclose(a + b, (a.cpu() + b.cpu()).cuda()), f"max|d|={(a + b - (a.cpu() + b.cpu()).cuda()).abs().max():.2e}")

# 2) matmul (uses cuBLAS)
x = torch.randn(2048, 2048, device="cuda")
y = torch.randn(2048, 2048, device="cuda")
z = x @ y
ref = (x.double().cpu() @ y.double().cpu()).float().cuda()
rel = ((z - ref).abs().max() / ref.abs().max()).item()
check("matmul 2048", rel < 1e-3, f"rel_err={rel:.2e}")

# 3) tensor core capable dtype
h = x.half() @ y.half()
check("fp16 matmul", h.shape == (2048, 2048), f"dtype={h.dtype}")

# 4) timing
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    _ = x @ y
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / 50
tflops = 2 * 2048**3 / dt / 1e12
print(f"{'matmul time':<15}: {dt * 1000:.3f} ms  ({tflops:.1f} TFLOPS fp32)")

# 5) memory round-trip
src = torch.arange(4096, dtype=torch.float32, device="cuda")
check("memcpy", torch.equal(src.cpu(), torch.arange(4096, dtype=torch.float32)))

print()
if failures:
    print("FAILED:", ", ".join(failures))
    sys.exit(1)
print("GPU CHECK PASSED")
