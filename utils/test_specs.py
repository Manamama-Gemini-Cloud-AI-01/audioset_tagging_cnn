#!/usr/bin/env python3
"""
torch_env_diag.py — Environment Diagnostic for Termux vs Proot Performance Gap
================================================================================
Tests the four hypotheses identified from the timing logs:
  H1. BLAS threading: how many cores PyTorch actually saturates
  H2. Memory allocator: page fault pressure during large tensor churn
  H3. Thread scheduling: involuntary preemption rate under load
  H4. Per-op throughput: raw GFLOP/s for the CNN hot-path operations

Run identically in both environments:
    python torch_env_diag.py | tee diag_$(uname -r | cut -d- -f2).txt

Then diff the two output files.
"""

import os, sys, time, platform, subprocess, gc, json, textwrap
import threading

# ── graceful import ────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import numpy as np
except ImportError as e:
    sys.exit(f"[FATAL] Missing dependency: {e}\n  pip install torch numpy")

SEPARATOR = "=" * 70

def section(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)

def read_proc(path, default="unavailable"):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return default

def shell(cmd, default="unavailable"):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return default

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0: Environment Identity
# ══════════════════════════════════════════════════════════════════════════════
section("0. ENVIRONMENT IDENTITY")

print(f"  Python         : {sys.version.split()[0]}")
print(f"  Platform       : {platform.platform()}")
print(f"  uname -r       : {shell('uname -r')}")
print(f"  sys.platform   : {sys.platform}")
print(f"  Torch version  : {torch.__version__}")
print(f"  Torch config   :")
for line in torch.__config__.show().splitlines():
    print(f"    {line}")

# Detect environment type
uname = shell("uname -r").lower()
env_type = "proot-debian" if "proot" in uname else "termux-native" if "android" in shell("uname -a").lower() else "unknown"
print(f"\n  Detected env   : {env_type}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — H1: BLAS Threading
# What BLAS/threading backend is actually linked, and how many threads fire
# ══════════════════════════════════════════════════════════════════════════════
section("1. H1 — BLAS THREADING BACKEND")

print(f"  torch.get_num_threads()          : {torch.get_num_threads()}")
print(f"  torch.get_num_interop_threads()  : {torch.get_num_interop_threads()}")
print(f"  OMP_NUM_THREADS (env)            : {os.environ.get('OMP_NUM_THREADS', 'not set')}")
print(f"  MKL_NUM_THREADS (env)            : {os.environ.get('MKL_NUM_THREADS', 'not set')}")

# Check which BLAS torch actually links
blas_info = torch.__config__.show()
for keyword in ["OpenBLAS", "MKL", "Eigen", "BLIS", "Accelerate", "LAPACK", "blas"]:
    if keyword.lower() in blas_info.lower():
        print(f"  BLAS keyword found             : {keyword}")

# Check shared libraries linked to torch's C extension
so_path = shell("python -c \"import torch._C; print(torch._C.__file__)\"")
print(f"  torch._C path                  : {so_path}")
if so_path != "unavailable":
    ldd_out = shell(f"ldd {so_path} 2>/dev/null || readelf -d {so_path} 2>/dev/null | grep NEEDED")
    blas_libs = [l for l in ldd_out.splitlines()
                 if any(x in l.lower() for x in ["blas", "lapack", "mkl", "openblas", "eigen", "gomp", "omp"])]
    if blas_libs:
        print("  BLAS/OMP linked libs:")
        for lib in blas_libs:
            print(f"    {lib.strip()}")
    else:
        print("  BLAS/OMP linked libs           : none detected via ldd/readelf")

# Measure actual thread concurrency during a matmul
# We time the same workload at thread counts 1, 2, 4, 8
print("\n  Thread scaling test (512x512 float32 matmul, 200 reps):")
print(f"  {'Threads':>8}  {'Time(s)':>9}  {'Speedup':>8}  {'Eff%':>6}")

A = torch.randn(512, 512)
B = torch.randn(512, 512)
baseline_time = None

for n_threads in [1, 2, 4, 6, 8]:
    torch.set_num_threads(n_threads)
    # warmup
    for _ in range(5):
        _ = torch.mm(A, B)
    torch.set_num_threads(n_threads)
    t0 = time.perf_counter()
    for _ in range(200):
        _ = torch.mm(A, B)
    elapsed = time.perf_counter() - t0
    if baseline_time is None:
        baseline_time = elapsed
    speedup = baseline_time / elapsed
    efficiency = speedup / n_threads * 100
    print(f"  {n_threads:>8}  {elapsed:>9.3f}  {speedup:>8.2f}x  {efficiency:>5.0f}%")

# Restore default thread count to the number of visible cores
try:
    torch.set_num_threads(len(os.sched_getaffinity(0)))
except Exception:
    torch.set_num_threads(4)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — H2: Memory Allocator Pressure
# Simulate the chunk churn from the script: allocate/free large tensors
# in a loop and measure the page fault delta
# ══════════════════════════════════════════════════════════════════════════════
section("2. H2 — MEMORY ALLOCATOR & PAGE FAULT PRESSURE")

def read_pagefaults():
    """Read minor+major page faults for this process from /proc/self/status."""
    raw = read_proc("/proc/self/status")
    faults = {}
    for line in raw.splitlines():
        if line.startswith("VmRSS"):
            faults["rss_kb"] = int(line.split()[1])
        if line.startswith("VmPeak"):
            faults["peak_kb"] = int(line.split()[1])
    return faults

def read_rusage_faults():
    """Get minor/major faults from /proc/self/stat fields."""
    try:
        parts = read_proc("/proc/self/stat").split()
        # fields 9,10,11,12 are minflt, cminflt, majflt, cmajflt
        return int(parts[9]), int(parts[11])
    except Exception:
        return 0, 0

# Simulate 12 chunks of 5.76M float32 samples (what the script processes)
CHUNK_SAMPLES = 5_760_000
CHUNK_DTYPE = torch.float32
N_CHUNKS = 12

print(f"  Simulating {N_CHUNKS} chunks × {CHUNK_SAMPLES/1e6:.2f}M float32 samples")
print(f"  (mirrors the script's 3-min chunk loop over a 34-min file)")

min_faults_before, maj_faults_before = read_rusage_faults()
gc.collect()
t0 = time.perf_counter()

alloc_times = []
access_times = []

for i in range(N_CHUNKS):
    ta = time.perf_counter()
    chunk = torch.zeros(1, CHUNK_SAMPLES, dtype=CHUNK_DTYPE)
    alloc_times.append(time.perf_counter() - ta)

    # Force actual page access (like the CNN reading every element)
    tb = time.perf_counter()
    _ = chunk.sum()
    access_times.append(time.perf_counter() - tb)

    del chunk
    gc.collect()

total_elapsed = time.perf_counter() - t0
min_faults_after, maj_faults_after = read_rusage_faults()

print(f"\n  Total churn time           : {total_elapsed:.3f}s")
print(f"  Avg alloc time / chunk     : {sum(alloc_times)/N_CHUNKS*1000:.2f} ms")
print(f"  Avg first-access / chunk   : {sum(access_times)/N_CHUNKS*1000:.2f} ms")
print(f"  Minor page faults (delta)  : {min_faults_after - min_faults_before:,}")
print(f"  Major page faults (delta)  : {maj_faults_after - maj_faults_before:,}")
print(f"  (High minor faults = allocator releases pages between chunks)")

# Also test: does keeping memory resident (no del) change access time?
print("\n  Retention test: access time WITH vs WITHOUT prior del:")
chunk_a = torch.zeros(1, CHUNK_SAMPLES, dtype=CHUNK_DTYPE)
_ = chunk_a.sum()  # fault it in
t0 = time.perf_counter()
_ = chunk_a.sum()  # should be cache-warm
warm_time = time.perf_counter() - t0

del chunk_a
gc.collect()
chunk_b = torch.zeros(1, CHUNK_SAMPLES, dtype=CHUNK_DTYPE)
t0 = time.perf_counter()
_ = chunk_b.sum()  # cold: pages may need re-faulting
cold_time = time.perf_counter() - t0
del chunk_b

print(f"    Warm (retained)  : {warm_time*1000:.3f} ms")
print(f"    Cold (after del) : {cold_time*1000:.3f} ms")
print(f"    Ratio cold/warm  : {cold_time/warm_time:.2f}x  (>1 = allocator releasing pages)")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — H3: Thread Scheduling Quality
# Measure involuntary context switches during a sustained parallel workload
# ══════════════════════════════════════════════════════════════════════════════
section("3. H3 — THREAD SCHEDULING QUALITY")

def read_ctx_switches():
    raw = read_proc("/proc/self/status")
    vol, invol = 0, 0
    for line in raw.splitlines():
        if "voluntary_ctxt_switches" in line and "nonvoluntary" not in line:
            vol = int(line.split()[1])
        if "nonvoluntary_ctxt_switches" in line:
            invol = int(line.split()[1])
    return vol, invol

try:
    n_cores = len(os.sched_getaffinity(0))
except Exception:
    n_cores = 4
print(f"  Visible cores (sched_getaffinity) : {n_cores}")
print(f"  /proc/cpuinfo processor count     : {shell('grep -c processor /proc/cpuinfo')}")
print(f"  CPU affinity mask                 : {shell('taskset -p $$ 2>/dev/null | grep -o 0x.*')}")

torch.set_num_threads(n_cores)

# Run a sustained parallel matmul workload and sample context switches
vol_before, invol_before = read_ctx_switches()
t0 = time.perf_counter()

M = torch.randn(1024, 1024)
for _ in range(100):
    M = torch.mm(M, M.T)
    M = M / M.norm()  # keep values sane

elapsed = time.perf_counter() - t0
vol_after, invol_after = read_ctx_switches()

print(f"\n  Sustained parallel matmul (1024×1024 × 100 reps):")
print(f"    Elapsed                   : {elapsed:.3f}s")
print(f"    Voluntary ctx switches    : {vol_after - vol_before:,}")
print(f"    Involuntary ctx switches  : {invol_after - invol_before:,}")
ratio = (invol_after - invol_before) / max(1, (vol_after - vol_before))
print(f"    Invol/Vol ratio           : {ratio:.2f}  (>>1 = threads getting preempted = Android scheduler friction)")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — H4: CNN Hot-Path Throughput
# Reproduce the actual operations from Cnn14_DecisionLevelMax:
# Conv2d (the dominant op), BatchNorm, ReLU, and Mel filterbank
# ══════════════════════════════════════════════════════════════════════════════
section("4. H4 — CNN HOT-PATH OP THROUGHPUT")

torch.set_num_threads(n_cores)

def bench(label, fn, warmup=3, reps=20):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    elapsed = time.perf_counter() - t0
    per_rep = elapsed / reps * 1000
    print(f"  {label:<45} : {per_rep:>8.2f} ms/rep")
    return per_rep

print(f"  Using {n_cores} threads. Input mimics one 3-min chunk through the CNN.\n")

# Mel spectrogram (torchaudio MelSpectrogram equivalent via manual filterbank)
# Cnn14 input: (batch=1, time=5.76M samples) → mel (1, 1, 64, ~562 frames)
try:
    import torchaudio
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=32000, n_fft=1024, hop_length=320,
        n_mels=64, f_min=50, f_max=14000
    )
    audio_chunk = torch.randn(1, 5_760_000)
    bench("MelSpectrogram (5.76M samples → 64×18000)", lambda: mel_transform(audio_chunk))
    del audio_chunk
except Exception as e:
    print(f"  MelSpectrogram skipped: {e}")

# Conv2d layers matching Cnn14 architecture
# Block 1: (1,64,T) → 64 filters, 3×3
x1 = torch.randn(1, 1, 64, 1800)   # (batch, ch, mels, time_frames)
conv1 = nn.Conv2d(1, 64, 3, padding=1)
bench("Conv2d block1  (1→64ch, 64×1800 input)", lambda: conv1(x1))

# Block 2: 64→128
x2 = torch.randn(1, 64, 32, 900)
conv2 = nn.Conv2d(64, 128, 3, padding=1)
bench("Conv2d block2  (64→128ch, 32×900 input)", lambda: conv2(x2))

# Block 3: 128→256
x3 = torch.randn(1, 128, 16, 450)
conv3 = nn.Conv2d(128, 256, 3, padding=1)
bench("Conv2d block3  (128→256ch, 16×450 input)", lambda: conv3(x3))

# Block 4: 256→512 (deepest, most expensive)
x4 = torch.randn(1, 256, 8, 225)
conv4 = nn.Conv2d(256, 512, 3, padding=1)
bench("Conv2d block4  (256→512ch, 8×225 input)", lambda: conv4(x4))

# BatchNorm (often overlooked but touches every element)
# FIX: x1 has shape (1, 1, 64, 1800) -> only 1 channel, but BatchNorm2d(64)
# expects 64 channels (its running_mean/running_var have 64 elements).
# Use a tensor shaped like conv1's *output* (64 channels) instead, since
# that's the realistic point in the CNN where a 64-channel BN actually runs.
x1_bn = torch.randn(1, 64, 64, 1800)
bn = nn.BatchNorm2d(64)
bench("BatchNorm2d    (64ch, 64×1800)", lambda: bn(x1_bn))
del x1_bn

# Full mini-CNN (stacked, like a real forward pass)
class MiniCnn14(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),
        )
    def forward(self, x):
        return self.net(x)

mini = MiniCnn14()
x_in = torch.randn(1, 1, 64, 1800)
bench("MiniCnn14 forward (full 3-block pass)", lambda: mini(x_in))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Allocator Identity
# What allocator is torch using, and what does /proc/self/maps look like
# ══════════════════════════════════════════════════════════════════════════════
section("5. ALLOCATOR & RUNTIME IDENTITY")

print(f"  libc allocator guess:")
maps = read_proc("/proc/self/maps")
for keyword in ["jemalloc", "scudo", "tcmalloc", "ptmalloc", "libc.so", "libc_malloc"]:
    if keyword in maps:
        print(f"    Found in /proc/self/maps : {keyword}")

# Check LD_PRELOAD for allocator overrides
print(f"  LD_PRELOAD                 : {os.environ.get('LD_PRELOAD', 'not set')}")

# Malloc stats if available
malloc_stats = shell("python -c \"import ctypes; libc=ctypes.CDLL('libc.so.6'); libc.malloc_stats()\" 2>&1")
if "Arena" in malloc_stats or "total" in malloc_stats.lower():
    print(f"  malloc_stats output:\n{textwrap.indent(malloc_stats[:500], '    ')}")

# Check if Transparent Huge Pages are enabled (affects large tensor mapping)
thp = read_proc("/sys/kernel/mm/transparent_hugepage/enabled", "unavailable (no THP on Android)")
print(f"  Transparent HugePages      : {thp}")

# OOM score (Android may be aggressive about reclaiming pages)
oom_score = read_proc("/proc/self/oom_score", "unavailable")
oom_adj = read_proc("/proc/self/oom_score_adj", "unavailable")
print(f"  OOM score / adj            : {oom_score} / {oom_adj}")
print(f"  (High oom_score_adj on Android = kernel more likely to reclaim pages → more faults)")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Summary & Verdict
# ══════════════════════════════════════════════════════════════════════════════
section("6. SUMMARY")

print("""
  Compare the two output files on these key numbers:

  H1 (BLAS threads):
    • Thread scaling test: does speedup plateau at 1.7x (Termux) or 4x (Proot)?
    • BLAS keyword + linked libs: OpenBLAS vs Eigen vs nothing

  H2 (Allocator pressure):
    • Minor page faults delta: should be ~14x higher in Termux
    • Cold/warm access ratio: >2x in Termux = allocator releasing pages

  H3 (Scheduler):
    • Involuntary ctx switch count during matmul workload
    • Invol/Vol ratio: should be much higher in Termux

  H4 (Raw throughput):
    • Conv2d block4 ms/rep: directly measures CNN hot-path FLOP/s
    • MiniCnn14 forward: end-to-end comparison

  H5 (Allocator identity):
    • Termux: likely 'scudo' (Android default since API 30)
    • Proot:  likely 'ptmalloc2' (glibc)
    • OOM adj: Termux process likely has high adj → aggressive page reclaim
""")

print(f"  Run completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Save output:  python torch_env_diag.py | tee diag_$(uname -r | cut -d- -f2).txt")
