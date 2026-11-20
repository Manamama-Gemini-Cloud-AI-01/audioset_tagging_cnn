# HDF5-backed dashboard — reference notes

**Status: deferred (ad calendas graecas).** Not planned work. Read this if a single
recording ever produces an HTML file larger than about 1 GB, and only then. The notes
exist so that the reasoning is not re-derived from scratch — and so that the mistakes
recorded at the bottom are not repeated.

---

## Why this is shelved

The current `interactive_dashboard.html` is **correct**. It applies an absolute 0.05 floor
to all 527 classes, keeps the explosion visible, and runs in one click. Its only fault is
size: the event matrix is serialised as JSON text (~9.9 bytes per score) next to ~4.85 MB
of embedded Plotly.

That fault does not bite at the scale in use. It bites at roughly 10 hours of audio, which
is exactly where a browser would also refuse to *draw* the result. Measured on the Elijah
clip (20.2 s, 5 Hz, 527 classes): 102 rows, 0.5 MB of JSON, 5.4 MB total HTML, HDF5 0.35 MB.

So the change buys nothing today, and cannot buy anything on its own tomorrow. Shelve it.

---

## The contract that must not change

Any replacement must show **the same classes by the same rule**, not merely the same data:

1. **Inclusion is an absolute floor, never a ranking.** Peak ≥ 0.05 is visible; everything
   else stays searchable. `top_k`, "strongest N", percentile cuts — all forbidden.
2. **All 527 classes remain reachable** from the UI, not a truncated subset.
3. **The floor is the only gate.** If it needs a second gate, that gate is wrong.

Evidence from the clip, because the numbers are the argument:

| selection rule | what appears |
|---|---|
| `index < 10` (the original bug) | Speech, Male speech, Conversation… — no events at all |
| top 5 by peak (the attempted "fix") | Dog .641, Animal .604, Domestic animals .559, Hiccup .438, Bow-wow .357 |
| peak ≥ 0.05 (correct) | 18 classes, **including Explosion .060 at 16.0 s** and Train .125 at 17.6 s |

Explosion ranks 18th of 527. Fireworks (19th), Firecracker (21st), Burst (25th) follow.
Any "top N" with N < 18 deletes the event under investigation; N = 18 is just the floor
written badly.

---

## Preferred design: load it all

Start here. It is the simplest thing that preserves the contract exactly.

1. Server reads `full_event_log.h5` with `h5py` and sends the **whole matrix as raw
   little-endian `float32`** — ~4 bytes per cell instead of ~9.9 for JSON text.
2. Browser: `new Float32Array(buffer)`, then `matrix.subarray(i, i + rows)` per class —
   *views into one buffer*, so memory stays at 1× and no per-trace copies exist.
3. Reuse the existing dashboard JS unchanged: 0.05 floor, 527-class search, `hoversort`,
   click-to-seek.
4. Create a Plotly trace **only** for classes above the floor, plus any class the user
   ticks. Do not instantiate 527 traces holding points nobody draws.

No `h5wasm`, no `WORKERFS`, no file picker, no windowed fetching. Serve `plotly.js` from the
installed `plotly` package, bind to `127.0.0.1`, serve media with byte ranges. Roughly
120 lines.

---

## Arithmetic: where "load it all" dies

`cells = hours × 3600 × 5 Hz × 527 classes`

| duration | cells | float32 | JSON text | Plotly points if all traces exist |
|---|---|---|---|---|
| 20 s | 54 k | 0.2 MB | 0.5 MB | 54 k |
| 1 h | 9.5 M | 38 MB | 94 MB | 9.5 M |
| 10 h | 95 M | 380 MB | 940 MB | 95 M |
| 100 h | 950 M | 3.8 GB | 9.4 GB | 950 M |

Two independent walls, both browser-side:

- **Heap.** Chrome kills the tab long before a 3.8 GB `ArrayBuffer`, and the buffer is not
  the only consumer.
- **Plotly.** `legendonly` traces still hold their points. 95 M points is a hung tab, not a
  slow one.

The refinement above (traces only for classes above the floor) cuts the point count by
~30× and pushes load-all to roughly **10 hours** — the same threshold as the current HTML.
Past that, no in-page design works.

---

## The rule that matters

> **The time axis may change. The class axis must not.**

- Small files: nothing changes at all; load everything and behave as today.
- Long files: window the *frames* (initial span, overview, on-demand detail). Never window
  the *class list*. Never let a frames-budget leak into a class budget.

Cap **cells (frames × classes)** per request, not the number of classes. When a window is
too long for all 527 curves, fall back to a per-frame top-K rollup *for display only*,
while the class list, the default set and the hover membership keep coming from the
precomputed index (labels + per-class peak + RLE event spans — kilobytes, not megabytes).

---

## Anti-patterns observed (do not repeat)

| anti-pattern | consequence |
|---|---|
| `visible: index < 10` | shows the dictionary's first ten words; no events |
| "show the 5 strongest classes" | deletes Explosion (rank 18) |
| max 16 classes at once | the dog family alone takes 7 of 16 slots |
| capping classes because the window is long | a rendering limit silently becomes a semantic filter |
| validating on "the chart renders" | the chart rendered correctly and said the wrong thing |

Related: `self_discoveries/the_dog_was_right.md` in
`manamama-deepseek-digital-garden` — the same failure in a narrative form, including why
per-class confidence is not a common scale and why ranking invites causal misreading.

---

## Acceptance test (write it before the code)

On `Elijah_Mikaelson_-_Wybuch_w_po.mp4`, with no clicks after load:

1. Explosion, Fireworks, Firecracker, Burst/pop and Train are all drawn and identifiable.
2. The 16.0 s onset is visibly distinct, not a flat band under a 0.64 background.
3. The default set is exactly `peak >= 0.05` — 18 classes — not a ranked subset.
4. With the sidebar floor lowered to 0.02, the count becomes 28 and the blast family stays.

If any of these fails, it is broken regardless of how the page looks.

---

## What was tried and rejected (2026-11-20)

- **Analyzer dedup — kept.** `framewise_ui` removed; one max-pooled `framewise_output`
  at `--ui_fps`, `float32`, no threshold. 480 KB → 348 KB on the clip, identical peaks
  (per-class correlation 0.994). This is what `master` contains.
- **Analyzer rewrite — reverted.** Streamed FFmpeg→HDF5 with staged publication, bounded
  overviews and disk checks. Larger change than the problem justified; not exercised on a
  long file before being committed.
- **Loopback viewer — abandoned.** The transport worked (h5py slices, byte-range media,
  auto-open); the default view violated the contract above. Preserved on branch
  `viewer-experiment-d4b1e62` as a reference for the endpoints, **not** as something to
  merge as-is.
