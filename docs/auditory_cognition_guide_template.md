# AI Analysis Guide — audioset_tagging_cnn output folder

**Version:** 3.1.0
**Audience:** an AI assistant handed one output folder, usually with **no server running**.
**Job:** answer *"what happened in this media, and when?"* from `summary_events.csv`. **No server is required**, and no code for a quick pass — but a few lines of stdlib Python are worth it when the CSV is long.

> v2 was built around a live Shapash dashboard that is normally not started. v3.1 puts the static CSV first, separates a **summary** read from a **forensic** read, and keeps Shapash as a short appendix.

---

## 1. What is in the folder

| File | What it is | Use it? |
| --- | --- | --- |
| `summary_events.csv` | Timestamped sound events above a 0.02 peak floor. | **Read first.** |
| `detailed_events_delta_ai_attention_friendly.json` | Per-class events (no 3-cap) + attack/decay traces, RLE-compressed. | Every occurrence of one class; metaphor test. |
| `full_event_log.h5` | Raw probability matrix, `(frames × 527)` classes. | Only to hunt *faint* events. |
| `interactive_dashboard.html` | Self-contained Plotly view of all 527 classes. | Manual browsing. |
| `eventogram.png` | Static picture: spectrogram + the top-10 loudest classes. | Visual gloss only. |
| `*_eventogram_dynamic*.mp4` | Eventogram video (± overlaid on the source). | Only if run with the flag. |
| `auditory_cognition_guide_template.md` | This file. | — |

The `.h5` is the complete forensic record: `framewise_output`, `labels`, `timestamps`, `stft_viz`, plus an `effective_ui_fps` attribute.

---

## 2. Two modes — pick one and say which

| Mode | Question | Rows used |
| --- | --- | --- |
| **Summary** (§3) | "Give me the headline events." | top ~8% (peak ≥ 0.30) |
| **Forensic** (§4) | "What is actually in this audio?" | **all** rows, clustered |

The audio is often the material itself (a game, a field recording, a scene to describe). Then forensic mode is the correct one, and summary mode is only its abstract.

---

## 3. Summary mode — "Timeline of confident events" (peak ≥ 0.30)

**Input:** `summary_events.csv`. Columns:

```
sound_class, start_time_seconds, end_time_seconds, duration_seconds, peak_probability, average_probability
```

**Recipe (no code):**

1. Read the CSV.
2. Keep every row with `peak_probability >= 0.30`.
3. Format each as `mm:ss–mm:ss  <sound_class>  peak=0.xx`.
4. Sort by `start_time_seconds`.

`0.30` is the working "confident" line for `Cnn14_DecisionLevelMax`. Call the result a *0.30 summary* — do not present it as "the events".

```
00:40–00:42  Cough                     peak=0.37
02:26–02:29  Police car (siren)        peak=0.39
02:57–03:01  Vehicle                   peak=0.80
22:35–26:30  Speech                    peak=0.90
```

Then add one or two lines of interpretation (phases, likely on-screen cause), and state the **baseline texture** first: the class with the longest `duration_seconds` rows spread across the file is the bed (e.g. "Speech" spanning minutes ⇒ talk-heavy, everything else is background). A row's `average_probability` near its peak = a sustained sound; far below = a spike.

---

## 4. Forensic mode — read the whole CSV, clustered (minimal coding is fine)

The CSV is **unfiltered**: every class whose peak reaches 0.02. A 55-minute clip gave **607 rows across 226 classes**, and a ≥0.30 summary keeps only ~8% of them. The rows you drop are not filler — rare and low-signal events live there, and the *supporting* members of a sound family are almost always below 0.30 (e.g. "Explosion" faint, and the "Boom / Burst / Gunshot" that co-fire with it fainter still).

A plain read fails at this length: 600+ rows is too long to eyeball, and the file is sorted by start time, so one physical event is scattered across several near-identical rows. **Cluster by time instead** — stdlib only:

```python
import csv, collections
rows = list(csv.DictReader(open('summary_events.csv')))
BUCKET = 10                                  # seconds
scenes = collections.defaultdict(list)
for r in rows:
    t = int(float(r['start_time_seconds']) // BUCKET) * BUCKET
    scenes[t].append((float(r['peak_probability']), r['sound_class']))
for t in sorted(scenes):
    ev = sorted(scenes[t], reverse=True)
    if len(ev) >= 3:                         # 3+ classes in one window ⇒ a scene
        print(f"{t//60:02d}:{t%60:02d}  " + ", ".join(f"{c} {p:.2f}" for p, c in ev[:8]))
```

Read the output as **scenes**, not rows: sibling classes firing in the same window describe *one* physical event, and its nature is in the *mix*.

- `Explosion + Boom + Burst + Gunshot + Artillery fire` → one loud broadband bang.
- `Wild animals + Roaring cats + Chirp, tweet + Bird` → an outdoor wildlife stretch.
- `Electronic music + Trance + Techno + Dubstep` → the radio's genre.
- `Cough + Throat clearing + Hiccup` clustered → one person at the microphone.

Two cautions:

- **Do not split this CSV with `awk -F,` or `cut -d,`.** Many class names contain commas (`"Chicken, rooster"`, `"Outside, urban or manmade"`), so those fields are quoted and column tools misread the file. Use the `csv` module.
- A 3+ class cluster is *not* proof of 3+ sounds. Low-peak families are frequently one loud transient spraying near-synonyms (§8).

---

## 5. Five facts about the data you must know

1. **The CSV's inclusion floor is a peak of 0.02.** Anything below never appears, even if it truly occurred. The CSV maps *prominent* sounds, not a full census.
2. **Blocks** are cut at onset/offset `0.01` with a **0.5 s minimum duration**, and are labelled by the block's *peak*. A steady 20-minute drone is a single row.
3. **Only the top 3 blocks per class are exported.** A class showing 3 rows may have had 30. **Never read a row count as an occurrence count, and never compare classes by row count.** Rank by `peak_probability`.
4. **Times are absolute** seconds from the start of the media (leading silence included).
5. **The delta JSON has the opposite trade-off.** It carries **every** block of a class above 0.05 (no 3-cap — here `Speech` has 164 events, `Music` 195, against 3 each in the CSV) but over **fewer** classes (144 vs 226) and without `average_probability`. So: *CSV = wide but shallow, JSON = deep but narrow.*

---

## 6. Reading a peak number

| Peak | Meaning |
| --- | --- |
| ≥ 0.60 | Near-certain — trust the label. |
| 0.30 – 0.60 | Confident event — keep the label, mild hedging is fine. |
| 0.10 – 0.30 | Plausible — texture or a hint; verify before asserting. |
| < 0.10 | Usually noise, metaphor, or mic handling. Do not build a claim on it alone. |

A peak is model *confidence*, not loudness: a quiet clean event can outrank a loud ambiguous one.

---

## 7. Getting more detail

Each step costs more than the last — stop at the first one that answers the question.

1. **A class missing from the CSV?** Check `detailed_events_delta_ai_attention_friendly.json` first: it covers a class down to a 0.05 peak and lists *every* occurrence without the 3-cap. It is ~750 KB, so **query it per class with `jq`, never dump it whole**:

   ```bash
   jq -r 'to_entries[] | "\(.value[0].peak_prob)\t\(.key)"' \
     detailed_events_delta_ai_attention_friendly.json | sort -rn | head -20   # what it covers

   jq -r '.["Telephone"][] | "\(.start_time)\t\(.end_time)\t\(.peak_prob)"' \
     detailed_events_delta_ai_attention_friendly.json                        # every phone event
   ```

   `delta_trace` is the attack/decay shape, run-length encoded: `{"skip": N}` = N frames of near-zero change. A big first positive value = a sharp attack; many small values = a sustained sound. That shape is what distinguishes a real event from a metaphor (§8).

2. **Below 0.05, or a class in neither the CSV nor the JSON?** Go to the raw matrix — `full_event_log.h5`, `framewise_output` `(frames × 527)`, no floor at all:

   ```python
   import h5py
   f = h5py.File('full_event_log.h5', 'r')
   labels = [s.decode().split(', ', 1)[-1] for s in f['labels'][:]]
   X = f['framewise_output'][:]        # (frames, 527)
   i = labels.index('Explosion')
   print(X[:, i].max())                # peak for that class over the whole file
   ```

3. **Want the shape over time?** Open `interactive_dashboard.html` in a browser — all 527 classes, searchable. Do not try to parse the HTML as data.

---

## 8. Model quirks (why a label can be wrong)

PANNs was trained on web audio, so it reaches for "semantic metaphors":

- Footsteps → **Animal**
- Mic / cable rubbing, wind → **Thunder** / **Rain**
- Distant or reverberant speech → **Speech** (present but not transcribable)
- Rhythmic machine → **Printer** / **Vacuum**

And: sounds right on the microphone (handling, breathing, a cough) produce the *largest* peaks of all and are device noise, not content. Before writing a confident event into the timeline as fact, check that it has a plausible on-screen cause.

---

## 9. Appendix — Shapash "Acoustic DNA" (only if a server is running)

Usually it is not running; skip this unless a specific label needs explaining. If one is up on `http://localhost:8050`:

```bash
curl -s http://localhost:8050/_dash-layout | jq 'type'   # "object" ⇒ server alive
```

The server explains a chosen class by the other classes that co-occur with it — its "Acoustic DNA", literal or metaphorical. The terminal-first version, when the repo is present:

```bash
python3 scripts/Shapash_visualization/launch_correlations_dashboard.py <PATH_TO_H5> --target "<SOUND_CLASS>"
```

Server numbers are Shapley *contributions* (roughly 10% of the CSV's probabilities) — calibrate with `max` and never compare them directly to CSV peaks. Full detail: `docs/acoustic_dna_terminal_protocol.md`.

---

## 10. Provenance

Generated by `pytorch/audioset_tagging_cnn_inference_6.py` (the inference stage copies this file into every output folder). Edit the repo copy at `docs/auditory_cognition_guide_template.md`; the folder copy is overwritten on each run. Use `--ui_fps 5` — the model only computes a fresh value every 320 ms, so higher FPS adds file size, not information.
