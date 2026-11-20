# Plan: Open an HDF5-backed dashboard automatically

## User workflow

1. Run `bash audio_me.sh /mnt/HP_P7_Data/Videos/Elijah_Mikaelson_-_Wybuch_w_po.mp4` as usual.
2. After analysis completes successfully, the wrapper starts a lightweight Python viewer, waits for it to answer a readiness check, then opens its localhost URL in the default browser. If browser launch is unavailable, print the URL. Provide `--no-open` for unattended runs and a clear way to stop the viewer.
3. To review later, run a small launcher such as `bash view_audio.sh /path/to/result_directory`. It opens the existing results without importing the model, running inference, or regenerating outputs. The local server must stay up while the dashboard is in use.

## Storage and viewer

- Keep `full_event_log.h5` as the sole persistent store for the full event matrix. The HTML is a separate reusable viewer, with Plotly served locally and no inline event matrix. Do not embed the HDF5 file or generate a second full-resolution format. Keep the source media separate and serve it with byte-range support for seeking.
- Bind the server to `127.0.0.1` on an available port. Serve the viewer, media, and data endpoints from the same origin. Open the selected result's HDF5 read-only via `h5py`; expose only the intended artifacts, not an arbitrary filesystem or directory listing. No file picker, CDN, browser HDF5 parser, or network access required.
- On open, show a bounded initial time window and the full recording's navigation range. On pan/zoom or class selection, request a bounded HDF5 slice. Debounce requests and discard stale responses. For wide views, return a bounded peak-preserving overview; if an overview index is persisted, keep it small and derived, not a second copy of the matrix.
- In detail mode, return every stored `float32` score for the selected time window and classes without extra rounding, thresholds, or quantization. Use stored timestamps where applicable and handle legacy files with `framewise_ui` explicitly. This preserves the stored grid; it does not restore frames already discarded by upstream max-pooling. Cap response size and ask users to narrow the time range or class selection rather than silently degrading detail.

## Keep the analysis itself bounded

A thin dashboard alone does not make a 100-hour run safe. Before enabling automatic opening by default:

1. Inspect `audio_me.sh` and its supported environments. Integrate the launcher only after a successful, validated result, including when an existing result is reused. Define explicit server lifecycle and cleanup behavior; do not leave a new orphan server on every invocation.
2. Remove the full-duration event RAM mirror: write each produced chunk to HDF5, grow the dataset as needed, and never drop rows because a duration estimate was low. Audit full reads of events and `stft_viz`, delta generation, static plots, and optional video renders. Use bounded chunks, bounded overview images, and streaming summaries with state carried across chunk boundaries. If an optional output cannot yet be produced within budget, disable it for large runs with an explicit explanation.
3. Before conversion or inference, check free space on the actual output and scratch filesystems. Estimate peak space including temporary media and optional artifacts with a safety margin; HDF5 compression is content-dependent. Check during long runs, fail clearly before exhausting space where possible, and distinguish incomplete outputs from finished results so a rerun cannot mistake a partial HDF5 file for a valid cache.
4. Replace the dense HTML export with the viewer. Implement metadata, detail-window, and overview endpoints with explicit request bounds. Start with in-process `h5py` slicing; benchmark realistic chunk layouts before changing the schema further.

## Acceptance checks

- On the Elijah sample, the wrapper opens the correct dashboard by itself; playback/seeking and zoom work. A later launcher invocation opens it from the saved `.h5` alone without inference (media is still needed for playback).
- Detail values, including faint scores, match HDF5 slices exactly; test chunk boundaries and an underestimated duration. Test legacy and current result formats.
- With a long logical/synthetic recording, measure peak analyzer/server/browser memory, temporary disk growth, response sizes, and zoom latency. Verify that neither browser startup nor zoom reads the whole file, and that the viewer size does not grow with recording duration. Run a physical multi-GB benchmark only after checking space and obtaining approval.
- Test insufficient disk, interrupted inference, missing/invalid results, browser-launch failure, reopening, and cleanup. Confirm that no dense HTML is produced and that the server is loopback-only.

*Status: plan only. Viewer and long-run safeguards not yet implemented.*
