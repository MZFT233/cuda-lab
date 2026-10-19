r"""OpenCV smoke test for the cuda-lab environment.

Run:  & .\.venv\Scripts\python.exe cv_demo.py

Checks the parts that usually break in a fresh install:
  - image codecs actually encode/decode (PNG, JPEG)
  - core processing ops produce expected results
  - BGR<->RGB convention (the classic OpenCV/PyTorch mismatch)
  - zero-copy-ish interop with a CUDA tensor
  - no CUDA build (reported, not hidden)
"""
import sys

import numpy as np

try:
    import cv2
except ImportError:
    print("cv2 is not installed. Install with:", file=sys.stderr)
    print("  uv pip install --python .venv\\Scripts\\python.exe opencv-python", file=sys.stderr)
    sys.exit(2)

failures = []


def check(name, ok, detail=""):
    print(f"{name:<30}: {'OK' if ok else 'FAIL'} {detail}")
    if not ok:
        failures.append(name)


print(f"cv2            : {cv2.__version__}")
print(f"numpy          : {np.__version__}")

# --- does this build have CUDA? (expected: no, for the pip wheel) ------------
build = cv2.getBuildInformation()
has_cuda = any(line.strip().startswith("CUDA:") and "YES" in line for line in build.splitlines())
print(f"CUDA build     : {'YES' if has_cuda else 'NO (pip wheel is CPU-only; OpenCL may still work)'}")
print()

# --- 1. synthesize an image --------------------------------------------------
h, w = 480, 640
img = np.zeros((h, w, 3), np.uint8)
img[:] = (30, 30, 30)
cv2.circle(img, (320, 240), 120, (0, 200, 255), -1)
cv2.rectangle(img, (40, 40), (200, 160), (255, 80, 0), -1)
cv2.putText(img, "cuda-lab", (60, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
check("synthesize image", img.shape == (480, 640, 3) and img.dtype == np.uint8, f"shape={img.shape}")

# --- 2. codecs round-trip ----------------------------------------------------
for ext, enc in (("png", ".png"), ("jpeg", ".jpg")):
    ok, buf = cv2.imencode(enc, img)
    if not ok:
        check(f"{ext} encode", False)
        continue
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if ext == "png":
        identical = np.array_equal(decoded, img)          # lossless
    else:
        identical = np.mean(np.abs(decoded.astype(int) - img.astype(int))) < 6   # lossy
    check(f"{ext} encode/decode", identical, f"{len(buf)} bytes")

# --- 3. processing ops -------------------------------------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
check("cvtColor", gray.shape == (h, w) and gray.ndim == 2, f"mean={gray.mean():.1f}")

blur = cv2.GaussianBlur(img, (9, 9), 0)
check("GaussianBlur", blur.shape == img.shape, f"delta={np.abs(blur.astype(int) - img.astype(int)).mean():.2f}")

edges = cv2.Canny(gray, 80, 160)
check("Canny", edges.shape == (h, w) and edges.max() == 255, f"edge_px={int((edges > 0).sum())}")

_, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
check("threshold", set(np.unique(thresh)).issubset({0, 255}), f"white={int((thresh > 0).sum())}")

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
check("findContours", len(contours) >= 2, f"{len(contours)} contours")

# --- 4. the BGR/RGB trap -----------------------------------------------------
swapped = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
px_bgr = img[240, 320]           # inside the circle: drawing used BGR (0,200,255)
px_rgb = swapped[240, 320]
check("BGR->RGB swap", tuple(px_rgb) == (px_bgr[2], px_bgr[1], px_bgr[0]), f"{tuple(px_bgr)} -> {tuple(px_rgb)}")

# --- 5. feature detection (SIFT ships in the main module since 4.4) ----------
try:
    sift = cv2.SIFT_create()
    kps, desc = sift.detectAndCompute(gray, None)
    check("SIFT", desc is not None and len(kps) > 0, f"{len(kps)} keypoints, desc={None if desc is None else desc.shape}")
except Exception as exc:  # noqa: BLE001
    check("SIFT", False, f"{type(exc).__name__}: {exc}")

# --- 6. torch interop --------------------------------------------------------
try:
    import torch

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    check("cv2 -> tensor", tuple(tensor.shape) == (1, 3, h, w), f"shape={tuple(tensor.shape)}")

    if torch.cuda.is_available():
        gpu = tensor.cuda()
        # a real op on the GPU, then back
        pooled = torch.nn.functional.interpolate(gpu, size=(h // 2, w // 2), mode="bilinear")
        back = pooled.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).byte().cpu().numpy()
        back_bgr = cv2.cvtColor(back, cv2.COLOR_RGB2BGR)
        check("GPU round-trip -> cv2", back_bgr.shape == (h // 2, w // 2, 3), f"shape={back_bgr.shape}, device={gpu.device}")
    else:
        print("GPU round-trip                 : SKIP (torch reports no CUDA device)")
except ImportError:
    print("cv2 -> tensor                  : SKIP (torch not installed)")

print()
if failures:
    print("CV DEMO FAILED:", ", ".join(failures))
    sys.exit(1)
print("CV DEMO PASSED")
