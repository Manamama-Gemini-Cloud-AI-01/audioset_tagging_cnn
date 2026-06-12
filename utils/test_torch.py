#!/usr/bin/env python3
"""
Termux Torch Environment & Binary Conflict Diagnostic
Focuses on: Eigen dominance, wrong env vars, hand-compiled .so conflicts, apt/pip mismatches, LD_PRELOAD issues.
"""

import os, sys, subprocess, glob, re, platform, gc
import torch

print("=== Termux Torch & Binary Diagnostics ===")
print(f"Python: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")
print(f"sys.platform: {sys.platform}")
print(f"Torch version: {torch.__version__}\n")

# === 1. Environment Variables ===
print("=== 1. Relevant Environment Variables ===")
for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", 
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "LD_PRELOAD", 
            "LD_LIBRARY_PATH", "PYTHONPATH", "TERMUX_PREFIX"]:
    print(f"  {var:25} = {os.environ.get(var, 'NOT SET')}")

# === 2. Torch Build Configuration ===
print("\n=== 2. Torch Build Config (Key Flags) ===")
config = torch.__config__.show()
print(config)

eigen_flag = "USE_EIGEN_FOR_BLAS=ON" in config
print(f"\n  🔴 USE_EIGEN_FOR_BLAS=ON detected: {eigen_flag}  ← This is likely hurting multi-threading")

# === 3. Linked Libraries in torch._C ===
print("\n=== 3. Libraries Linked to torch._C.so ===")
try:
    so_path = torch._C.__file__
    print(f"torch._C path: {so_path}")
    
    ldd_cmd = f"ldd {so_path} 2>/dev/null || echo 'ldd not available'"
    ldd_out = subprocess.getoutput(ldd_cmd)
    print(ldd_out)
    
    # Highlight suspicious libs
    suspicious = [line for line in ldd_out.splitlines() 
                  if any(x in line.lower() for x in ["blas", "openblas", "eigen", "gomp", "termux", "custom"])]
    if suspicious:
        print("\n  Highlighted BLAS/OMP related libs:")
        for line in suspicious:
            print(f"    {line.strip()}")
except Exception as e:
    print(f"Error inspecting .so: {e}")

# === 4. Conflicting Installations (pip vs apt) ===
print("\n=== 4. Conflicting Torch / BLAS Installations ===")
locations = {
    "pip_torch": "/data/data/com.termux/files/usr/lib/python*/site-packages/torch*",
    "apt_torch": "/data/data/com.termux/files/usr/lib/python*/dist-packages/torch*",
    "openblas": "/data/data/com.termux/files/usr/lib/libopenblas*",
    "custom_so": "/data/data/com.termux/files/usr/lib/python*/site-packages/*custom*",
}

for name, pattern in locations.items():
    found = glob.glob(pattern, recursive=True)
    if found:
        print(f"  {name:12} : {len(found)} items found")
        for p in found[:6]:  # limit output
            print(f"    {p}")
        if len(found) > 6:
            print(f"    ... and {len(found)-6} more")

# === 5. Hand-compiled / Suspicious .so files ===
print("\n=== 5. Potentially Hand-Compiled or Problematic .so Files ===")
so_files = glob.glob("/data/data/com.termux/files/usr/lib/python*/site-packages/**/*.so", recursive=True)
custom_sos = [f for f in so_files if any(k in f.lower() for k in ["custom", "recompiled", "manual", "eigen", "blas"])]
for f in custom_sos[:10]:
    print(f"    {f}")
if len(custom_sos) > 10:
    print(f"    ... and {len(custom_sos)-10} more")

# === 6. Quick Threading Rescue Test ===
print("\n=== 6. Quick Threading Test (with forced settings) ===")
def quick_matmul_test(threads):
    torch.set_num_threads(threads)
    A = torch.randn(512, 512)
    B = torch.randn(512, 512)
    import time
    t0 = time.perf_counter()
    for _ in range(50):
        _ = torch.mm(A, B)
    return time.perf_counter() - t0

print("  Threads | Time (s)")
for t in [1, 2, 4, 8]:
    elapsed = quick_matmul_test(t)
    print(f"  {t:7} | {elapsed:.3f}")

# Restore
torch.set_num_threads(8)

print("\n=== Diagnostic Complete ===")
print("Key things to look for:")
print("• USE_EIGEN_FOR_BLAS=ON + poor thread scaling → main performance killer")
print("• Multiple torch/openblas installations → ABI conflicts")
print("• LD_PRELOAD with termux-exec → can interfere with threading/memory")
print("• Hand-compiled .so files with different compiler flags")

