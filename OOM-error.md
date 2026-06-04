When running below, I have safe: 
```
zezen@zezen-HP-Pavilion-x360-Convertible:~/Downloads/GitHub/audioset_tagging_cnn$ free
              total        used        free      shared  buff/cache   available
Mem:        8043472     4470560     1648352      266104     1924560     3040448
Swap:       8388604     2030336     6358268
zezen@zezen-HP-Pavilion-x360-Convertible:~/Downloads/GitHub/audioset_tagging_cnn$ 

```

at: 

```
zezen@zezen-HP-Pavilion-x360-Convertible:~/Downloads/GitHub/audioset_tagging_cnn$ time bash -c 'for file in "/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split/"*; do bash audio_me.sh "$file" --dynamic_eventogram; done'
--- AudioSet Tagging CNN Inference (SED Mode) ---
Model Type: Cnn14_DecisionLevelMax
Checkpoint path: /media/zezen/HP_P7_Data/Temp/AI_models/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth
Input file and parameters: /mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split/output_000.mp3 --dynamic_eventogram
Add  --dynamic_eventogram   or  --static_eventogram  for a video of the graph with a moving marker.

Note: If you see errors with sox / mp3 parsing, then apt install 'sox' and 'libsox-fmt-mp3', test it via 'sox --info /mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split/output_000.mp3 --dynamic_eventogram ' 
Note: Paths provided as command-line arguments should be absolute to avoid ambiguity.
We are checking the real path of the first argument, which may hang if the file be not accessible...

2026-06-04 09:52:58 zezen-HP-Pavilion-x360-Convertible numexpr.utils[13755] INFO NumExpr defaulting to 4 threads.
Eventogrammer, version 6.8.11
Adaptation of: https://github.com/qiuqiangkong/audioset_tagging_cnn

Recent Material Changes:
* Constants Promoted: vis_fps, output_fps, and adaptive_lookahead are now CLI arguments.
* Speed Hack: Persistent Matplotlib figures with Artist Updates.
* Visual Fix: Proper window centering for scrolling eventograms.
* Refactor: Gemini AI rationalized path management and code structure.

Note on the models:
* Cnn14_DecisionLevelMax (Sound Event Detection): Uses Decision-level pooling to maintain
  time resolution. Essential for generating Eventograms and high-res CSV logs.
* Other models: Best for global audio tagging (use the '--audio_tagging' mode).

Performance & Stability:
* Processing ratio: ~15s audio per 1s CPU time (300 GFLOPs, 4-core, no viz).
* Platform Gap: works ~1.7x faster in Prooted Debian than in Termux (Eigen BLAS).
* OOM Safety: Close browsers or restart whole phone if crashes occur in Termux.


Split Suggestion:
If the file is too long, use FFmpeg to segment it first:
mkdir split_input_media && cd split_input_media && \
ffmpeg -i /mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split/output_000.mp3 -c copy -f segment -segment_time 1200 output_%03d.mp4
time bash -c 'for file in *; do bash audio_me.sh "$file" --dynamic_eventogram; done'

Tips & Environment Hacks:
* For 'undefined symbol: torch_library_impl' or 'NotImplementedError':
  Run: pip install -U torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
* If Torchcodec errors (e.g., 'libnvrtc.so.13 not found'), simply remove it:
  Run: pip uninstall torchcodec  (The script will safely fallback to FFmpeg)
* If you see 'NotImplementedError: sys.platform = android' after an update:
  Edit: /data/data/com.termux/files/usr/lib/python3.10/site-packages/torchaudio/_internally_replaced_utils.py
  Change 'if sys.platform == "linux":' to 'if sys.platform == "android":'

Dependency Versions:
MoviePy: 2.1.1
Torchaudio: 2.8.0+cpu
Torchcodec: 0.7.0

Using device: cpu
Using CPU.
Copied AI analysis guide to: /mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split/output_000_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/auditory_cognition_guide_template.md
⏲  🗃️  File duration: 2:46:40
📊  Starting inference in 3.0m chunks. (RAM Avail: 5181 MB)
    Resolution: Disk 100 Hz (Data Frames) | Visualization 5 Hz (UI/RAM)
Chunk at 0m finished. (RAM Avail: 4164 MB)
Chunk at 3m finished. (RAM Avail: 4145 MB)
Chunk at 6m finished. (RAM Avail: 4154 MB)
Chunk at 9m finished. (RAM Avail: 4120 MB)
Chunk at 12m finished. (RAM Avail: 4098 MB)
Chunk at 15m finished. (RAM Avail: 4098 MB)
Chunk at 18m finished. (RAM Avail: 4129 MB)
Chunk at 21m finished. (RAM Avail: 4096 MB)
Chunk at 24m finished. (RAM Avail: 4087 MB)
Chunk at 27m finished. (RAM Avail: 4084 MB)
Chunk at 30m finished. (RAM Avail: 3733 MB)
Chunk at 33m finished. (RAM Avail: 3592 MB)
Chunk at 36m finished. (RAM Avail: 3624 MB)
Chunk at 39m finished. (RAM Avail: 3558 MB)
Chunk at 42m finished. (RAM Avail: 3640 MB)
Chunk at 45m finished. (RAM Avail: 3679 MB)
Chunk at 48m finished. (RAM Avail: 3649 MB)
Chunk at 51m finished. (RAM Avail: 3646 MB)
Chunk at 54m finished. (RAM Avail: 3584 MB)
Chunk at 57m finished. (RAM Avail: 3512 MB)
Chunk at 60m finished. (RAM Avail: 3509 MB)
Chunk at 63m finished. (RAM Avail: 3491 MB)
Chunk at 66m finished. (RAM Avail: 3478 MB)
Chunk at 69m finished. (RAM Avail: 3433 MB)
Chunk at 72m finished. (RAM Avail: 3196 MB)
Chunk at 75m finished. (RAM Avail: 3200 MB)
Chunk at 78m finished. (RAM Avail: 3320 MB)
Chunk at 81m finished. (RAM Avail: 3358 MB)
Chunk at 81m finished. (RAM Avail: 3358 MB)
Chunk at 84m finished. (RAM Avail: 3385 MB)
Chunk at 87m finished. (RAM Avail: 3338 MB)
Chunk at 90m finished. (RAM Avail: 3324 MB)
Chunk at 93m finished. (RAM Avail: 3266 MB)
Chunk at 96m finished. (RAM Avail: 3171 MB)
Chunk at 99m finished. (RAM Avail: 3065 MB)
Chunk at 102m finished. (RAM Avail: 3113 MB)
Chunk at 105m finished. (RAM Avail: 3112 MB)
Chunk at 108m finished. (RAM Avail: 3068 MB)
Chunk at 111m finished. (RAM Avail: 2941 MB)
Chunk at 114m finished. (RAM Avail: 2826 MB)
Chunk at 117m finished. (RAM Avail: 2784 MB)
Chunk at 120m finished. (RAM Avail: 2737 MB)
Chunk at 123m finished. (RAM Avail: 2702 MB)
Chunk at 126m finished. (RAM Avail: 2709 MB)
Chunk at 129m finished. (RAM Avail: 2668 MB)
Chunk at 132m finished. (RAM Avail: 2656 MB)
Chunk at 135m finished. (RAM Avail: 2607 MB)
Chunk at 138m finished. (RAM Avail: 2521 MB)
Chunk at 141m finished. (RAM Avail: 2492 MB)
Chunk at 144m finished. (RAM Avail: 2518 MB)
Chunk at 147m finished. (RAM Avail: 2446 MB)
Chunk at 150m finished. (RAM Avail: 2368 MB)
Chunk at 153m finished. (RAM Avail: 2322 MB)
Chunk at 156m finished. (RAM Avail: 2323 MB)
Chunk at 159m finished. (RAM Avail: 2288 MB)
Chunk at 162m finished. (RAM Avail: 2090 MB)
Chunk at 165m finished. (RAM Avail: 2178 MB)
Aggregation complete. Internal Viz resolution: 5 Hz (Data Frames)
Final analysis duration: 10011.00s
Saved eventogram PNG to: /mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split/output_000_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/eventogram.png
📊  Generating AI-friendly event summary files…
Saved summary events CSV to: /mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split/output_000_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/summary_events.csv
📊  Generating detailed AI-friendly event delta JSON (RLE optimized)…
Saved AI-friendly delta JSON to: /mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split/output_000_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/detailed_events_delta_ai_attention_friendly.json
📊  Generating interactive Plotly dashboard…
Saved interactive dashboard to: /mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split/output_000_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/interactive_dashboard.html
🎞  Initializing video rendering pipeline (30 FPS)…
📊  Precomputing 50055 windows (Adaptive=False)…
MoviePy - Building video /mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split/output_000_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/output_000_audioset_tagging_cnn_eventogram_dynamic.mp4.
MoviePy - Writing audio in /mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split/output_000_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/temp_render_audio.mp3
chunk:  50%|████████████████████                    | 110407/220501 [01:40<01:34, 1164.01it/s, now=None]
chunk:  50%|████████████████████                    | 110877/220501 [01:41<01:35, 1148.64it/s, now=None]
chunk:  50%|████████████████████▏                   | 111219/220501 [01:41<01:37, 1118.63it/s, now=None]
chunk:  51%|████████████████████▏                   | 111448/220501 [01:41<01:37, 1114.69it/s, now=None]
chunk:  51%|████████████████████▎                   | 111676/220501 [01:41<01:37, 1119.74it/s, now=None]
chunk:  51%|████████████████████▎                   | 112261/220501 [01:42<01:33, 1161.21it/s, now=None]
chunk:  55%|█████████████████████▊                  | 120227/220501 [01:49<01:32, 1079.92it/s, now=None]
chunk:  55%|█████████████████████▊                  | 120575/220501 [01:49<01:28, 1128.67it/s, now=None]
chunk:  93%|█████████████████████████████████████▎  | 205814/220501 [03:05<00:13, 1117.57it/s, now=None]
chunk:  94%|█████████████████████████████████████▍  | 206295/220501 [03:06<00:12, 1171.62it/s, now=None]
MoviePy - Done.                                                                                         
MoviePy - Writing video /mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split/output_000_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/output_000_audioset_tagging_cnn_eventogram_dynamic.mp4

frame_index:   0%|                                      | 98/300330 [00:03<3:27:57, 24.06it/s, now=None]
frame_index:   0%|                                     | 107/300330 [00:04<3:21:14, 24.86it/s, now=None]
````


But it OOMs if I tried to process whole:

```
zezen@zezen-HP-Pavilion-x360-Convertible:/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split$ mediainfo "/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg.mp3" 
General
Complete name                            : /mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg.mp3
Format                                   : MPEG Audio
File size                                : 103 MiB
Duration                                 : 10 h 1 min
Overall bit rate mode                    : Constant
Overall bit rate                         : 24.0 kb/s
Writing library                          : Lavf58.29.100

Audio
Format                                   : MPEG Audio
Format version                           : Version 2
Format profile                           : Layer 3
Duration                                 : 10 h 1 min
Bit rate mode                            : Constant
Bit rate                                 : 24.0 kb/s
Channel(s)                               : 1 channel
Sampling rate                            : 16.0 kHz
Frame rate                               : 27.778 FPS (576 SPF)
Compression mode                         : Lossy
Stream size                              : 103 MiB (100%)


zezen@zezen-HP-Pavilion-x360-Convertible:/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split$ 

```

so I split it via :

`time ffmpeg -i "/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg.mp3" -c copy -f segment -segment_time 10000 output_%03d.mp3
`


See discussion below: 




> From: https://claude.ai/chat/8f655951-5f33-4b5c-a613-3682cb1c2aae

# you asked

message time: 2026-06-04 09:37:33

What is syntax error with my loop here? : 

zezen@zezen-HP-Pavilion-x360-Convertible:~/Downloads/GitHub/audioset_tagging_cnn$ bash audio_me.sh 
--- AudioSet Tagging CNN Inference (SED Mode) ---
Model Type: Cnn14_DecisionLevelMax
Checkpoint path: /media/zezen/HP_P7_Data/Temp/AI_models/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth
Input file and parameters: 
Add  --dynamic_eventogram   or  --static_eventogram  for a video of the graph with a moving marker.
Note: If you see errors with sox / mp3 parsing, then apt install 'sox' and 'libsox-fmt-mp3', test it via 'sox --info  ' 
Note: Paths provided as command-line arguments should be absolute to avoid ambiguity.
We are checking the real path of the first argument, which may hang if the file be not accessible...
realpath: '': No such file or directory
ERROR: The input file has not been provided or has not been found at '' 
zezen@zezen-HP-Pavilion-x360-Convertible:~/Downloads/GitHub/audioset_tagging_cnn$ for file in "/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split" do; bash audio_me.sh "$file" --dynamic_eventogram   done 
bash: syntax error near unexpected token `bash'
zezen@zezen-HP-Pavilion-x360-Convertible:~/Downloads/GitHub/audioset_tagging_cnn$ file  "/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split" 
/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split: directory
zezen@zezen-HP-Pavilion-x360-Convertible:~/Downloads/GitHub/audioset_tagging_cnn$

---

# claude response

The issue is `do;` — the semicolon is in the wrong place. It should come **after** the `do`, not before it. Also, `done` needs a semicolon or newline before it when written on one line.

**Your broken syntax:**
```bash
for file in "..." do; bash audio_me.sh "$file" --dynamic_eventogram done
```

**Fixed:**
```bash
for file in "/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split"; do bash audio_me.sh "$file" --dynamic_eventogram; done
```

The rules for a one-liner `for` loop are:
```bash
for var in <items>; do <command>; done
```

- Semicolon goes **after** the item list (before `do`)
- Semicolon goes **after** each command (before `done`)

---

Also worth noting: since `file` resolves to a **directory**, you likely want to loop over the files *inside* it instead:

```bash
for file in "/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split"/*; do
    bash audio_me.sh "$file" --dynamic_eventogram
done
```

That `/*` glob expands to all files within that directory, which is probably what you intended.

---

# you asked

message time: 2026-06-04 09:41:48

Yes. This works: `for file in "/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split/"*; do bash audio_me.sh "$file" --dynamic_eventogram ;  done 
`. 
How can I `time`  it? Ref: 

zezen@zezen-HP-Pavilion-x360-Convertible:~/Downloads/GitHub/audioset_tagging_cnn$ time for file in "/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split/"*; do bash audio_me.sh "$file" --dynamic_eventogram ;  done 
bash: syntax error near unexpected token `do'
zezen@zezen-HP-Pavilion-x360-Convertible:~/Downloads/GitHub/audioset_tagging_cnn$ time (for file in "/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split/"*; do bash audio_me.sh "$file" --dynamic_eventogram ;  done )
bash: syntax error near unexpected token `('
zezen@zezen-HP-Pavilion-x360-Convertible:~/Downloads/GitHub/audioset_tagging_cnn$

---

# claude response

The issue is that `time` in bash can't directly time compound commands from the command line like that. Use curly braces or a subshell with `{ }`:

```bash
time { for file in "/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split/"*; do bash audio_me.sh "$file" --dynamic_eventogram; done; }
```

Note the **space after `{`** and **semicolon before `}`** — both are required.

---

Alternatively, the cleanest approach is to just wrap it in a subshell using `bash -c`:

```bash
time bash -c 'for file in "/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split/"*; do bash audio_me.sh "$file" --dynamic_eventogram; done'
```

---

The reason `time (...)` and `time for ...` fail at the terminal is that `time` is a shell **keyword**, not a command, and it only accepts simple commands or `{ }` blocks — not subshells `()` or flow-control keywords like `for` directly after it in interactive bash.

---

# you asked

message time: 2026-06-04 10:03:09

Hm. While indeed (only!) this works: `time bash -c 'for file in "/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split/"*; do bash audio_me.sh "$file" --dynamic_eventogram; done'` and I know why it works, but `time` here is not a keyword. Look: 

```
zezen@zezen-HP-Pavilion-x360-Convertible:/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split$ alias time
alias time='/usr/bin/time -v '
zezen@zezen-HP-Pavilion-x360-Convertible:/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split$ time ls
'output_000_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn'   output_002.mp3
 output_000.mp3                                                          output_003.mp3
 output_001.mp3
	Command being timed: "ls --color=auto"
	User time (seconds): 0.00
	System time (seconds): 0.00
	Percent of CPU this job got: 66%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:00.00
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 2504
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 3
	Minor (reclaiming a frame) page faults: 129
	Voluntary context switches: 3
	Involuntary context switches: 2
	Swaps: 0
	File system inputs: 280
	File system outputs: 0
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0
zezen@zezen-HP-Pavilion-x360-Convertible:/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split$ 

```

Yet it had not worked without that `bash -c` trick.

---

# claude response

You're absolutely right, good catch! Since `time` is aliased to `/usr/bin/time -v`, it's an **external command** — not the bash keyword. And external commands can only take a single executable + its arguments, not shell constructs like `for` loops.

So the constraint isn't about `time` being a keyword here — it's that **no external command can directly execute a shell loop**. The `bash -c` trick works because you're passing the loop to `bash` as a string argument, and `bash` is the executable that `/usr/bin/time -v` actually runs.

It's the same reason why this would also work:

```bash
time bash -c 'ls'
```
vs the alias just doing:
```bash
/usr/bin/time -v ls
```

Both are equivalent — `bash -c` is just the way to wrap a compound shell construct into something an external command can invoke.

So to summarize: **the `bash -c` wrapper is always required when using `/usr/bin/time` (or any external command) with loops/pipelines**, since those constructs only exist inside a shell, not as standalone executables. The built-in `time` keyword doesn't have this limitation because it operates *within* the shell's own parsing.

---

# you asked

message time: 2026-06-04 10:04:38

Good news: after your advice (and my splitting the original via ffmpeg) this works: 

```
zezen@zezen-HP-Pavilion-x360-Convertible:~/Downloads/GitHub/audioset_tagging_cnn$ time bash -c 'for file in "/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split/"*; do bash audio_me.sh "$file" --dynamic_eventogram; done'
--- AudioSet Tagging CNN Inference (SED Mode) ---
Model Type: Cnn14_DecisionLevelMax
Checkpoint path: /media/zezen/HP_P7_Data/Temp/AI_models/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth
Input file and parameters: /mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split/output_000.mp3 --dynamic_eventogram
Add  --dynamic_eventogram   or  --static_eventogram  for a video of the graph with a moving marker.

Note: If you see errors with sox / mp3 parsing, then apt install 'sox' and 'libsox-fmt-mp3', test it via 'sox --info /mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split/output_000.mp3 --dynamic_eventogram ' 
Note: Paths provided as command-line arguments should be absolute to avoid ambiguity.
We are checking the real path of the first argument, which may hang if the file be not accessible...

2026-06-04 09:52:58 zezen-HP-Pavilion-x360-Convertible numexpr.utils[13755] INFO NumExpr defaulting to 4 threads.
Eventogrammer, version 6.8.11
Adaptation of: https://github.com/qiuqiangkong/audioset_tagging_cnn

Recent Material Changes:
* Constants Promoted: vis_fps, output_fps, and adaptive_lookahead are now CLI arguments.
* Speed Hack: Persistent Matplotlib figures with Artist Updates.
* Visual Fix: Proper window centering for scrolling eventograms.
* Refactor: Gemini AI rationalized path management and code structure.

Note on the models:
* Cnn14_DecisionLevelMax (Sound Event Detection): Uses Decision-level pooling to maintain
  time resolution. Essential for generating Eventograms and high-res CSV logs.
* Other models: Best for global audio tagging (use the '--audio_tagging' mode).

Performance & Stability:
* Processing ratio: ~15s audio per 1s CPU time (300 GFLOPs, 4-core, no viz).
* Platform Gap: works ~1.7x faster in Prooted Debian than in Termux (Eigen BLAS).
* OOM Safety: Close browsers or restart whole phone if crashes occur in Termux.

Split Suggestion:
If the file is too long, use FFmpeg to segment it first:
mkdir split_input_media && cd split_input_media && \
ffmpeg -i /mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split/output_000.mp3 -c copy -f segment -segment_time 1200 output_%03d.mp4
time bash -c 'for file in *; do bash audio_me.sh "$file" --dynamic_eventogram; done'

Tips & Environment Hacks:
* For 'undefined symbol: torch_library_impl' or 'NotImplementedError':
  Run: pip install -U torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
* If Torchcodec errors (e.g., 'libnvrtc.so.13 not found'), simply remove it:
  Run: pip uninstall torchcodec  (The script will safely fallback to FFmpeg)
* If you see 'NotImplementedError: sys.platform = android' after an update:
  Edit: /data/data/com.termux/files/usr/lib/python3.10/site-packages/torchaudio/_internally_replaced_utils.py
  Change 'if sys.platform == "linux":' to 'if sys.platform == "android":'

Dependency Versions:
MoviePy: 2.1.1
Torchaudio: 2.8.0+cpu
Torchcodec: 0.7.0

Using device: cpu
Using CPU.
Copied AI analysis guide to: /mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split/output_000_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/auditory_cognition_guide_template.md
⏲  🗃️  File duration: 2:46:40
📊  Starting inference in 3.0m chunks. (RAM Avail: 5181 MB)
    Resolution: Disk 100 Hz (Data Frames) | Visualization 5 Hz (UI/RAM)
Chunk at 0m finished. (RAM Avail: 4164 MB)
Chunk at 3m finished. (RAM Avail: 4145 MB)
Chunk at 6m finished. (RAM Avail: 4154 MB)
Chunk at 9m finished. (RAM Avail: 4120 MB)
Chunk at 12m finished. (RAM Avail: 4098 MB)
Chunk at 15m finished. (RAM Avail: 4098 MB)
Chunk at 18m finished. (RAM Avail: 4129 MB)
Chunk at 21m finished. (RAM Avail: 4096 MB)
Chunk at 24m finished. (RAM Avail: 4087 MB)
Chunk at 27m finished. (RAM Avail: 4084 MB)
Chunk at 30m finished. (RAM Avail: 3733 MB)
Chunk at 33m finished. (RAM Avail: 3592 MB)
Chunk at 36m finished. (RAM Avail: 3624 MB)
Chunk at 39m finished. (RAM Avail: 3558 MB)
Chunk at 42m finished. (RAM Avail: 3640 MB)
Chunk at 45m finished. (RAM Avail: 3679 MB)
Chunk at 48m finished. (RAM Avail: 3649 MB)
Chunk at 51m finished. (RAM Avail: 3646 MB)
Chunk at 54m finished. (RAM Avail: 3584 MB)
Chunk at 57m finished. (RAM Avail: 3512 MB)
Chunk at 60m finished. (RAM Avail: 3509 MB)
Chunk at 63m finished. (RAM Avail: 3491 MB)
Chunk at 66m finished. (RAM Avail: 3478 MB)
Chunk at 69m finished. (RAM Avail: 3433 MB)
Chunk at 72m finished. (RAM Avail: 3196 MB)
Chunk at 75m finished. (RAM Avail: 3200 MB)
Chunk at 78m finished. (RAM Avail: 3320 MB)
Chunk at 81m finished. (RAM Avail: 3358 MB)
Chunk at 84m finished. (RAM Avail: 3385 MB)
Chunk at 87m finished. (RAM Avail: 3338 MB)
Chunk at 90m finished. (RAM Avail: 3324 MB)
Chunk at 93m finished. (RAM Avail: 3266 MB)
Chunk at 96m finished. (RAM Avail: 3171 MB)
Chunk at 99m finished. (RAM Avail: 3065 MB)
Chunk at 102m finished. (RAM Avail: 3113 MB)

```

---

# claude response

Great, it's running! A nearly 3-hour audio file being processed in 3-minute chunks — that's going to take a while. Based on the performance note in the script's own output:

> *Processing ratio: ~15s audio per 1s CPU time*

So for **2h 46m 40s** (= 10,000s of audio):
- ~10,000s ÷ 15 = **~667 seconds ≈ ~11 minutes** per file (no viz)
- With `--dynamic_eventogram` the rendering overhead will add more on top of that

You can watch it progress and when it finishes you'll get the `time` output at the bottom showing the real wall-clock duration.

One tip for next time — if you want to monitor all files in the loop progressing, you could add a simple counter:

```bash
time bash -c '
i=0; total=$(ls "/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split/" | wc -l)
for file in "/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split/"*; do
  i=$((i+1))
  echo "=== Processing file $i / $total: $(basename "$file") ==="
  bash audio_me.sh "$file" --dynamic_eventogram
done'
```

That way you can see at a glance which file out of how many you're on, rather than inferring it from the chunk timestamps.

---

# you asked

message time: 2026-06-04 10:06:00

I rephrase. After me doing: `time ffmpeg -i "/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg.mp3" -c copy -f segment -segment_time 10000 output_%03d.mp3`  this file is being processed (indirectly): 

```
zezen@zezen-HP-Pavilion-x360-Convertible:/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split$ mediainfo "/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg.mp3" 
General
Complete name                            : /mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg.mp3
Format                                   : MPEG Audio
File size                                : 103 MiB
Duration                                 : 10 h 1 min
Overall bit rate mode                    : Constant
Overall bit rate                         : 24.0 kb/s
Writing library                          : Lavf58.29.100

Audio
Format                                   : MPEG Audio
Format version                           : Version 2
Format profile                           : Layer 3
Duration                                 : 10 h 1 min
Bit rate mode                            : Constant
Bit rate                                 : 24.0 kb/s
Channel(s)                               : 1 channel
Sampling rate                            : 16.0 kHz
Frame rate                               : 27.778 FPS (576 SPF)
Compression mode                         : Lossy
Stream size                              : 103 MiB (100%)


zezen@zezen-HP-Pavilion-x360-Convertible:/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split$ 


```

thanks to your `time  bash -c` trick

---

# claude response

Ah, so it's actually a **10-hour** file, not 2h 46m! The `2:46:40` shown earlier was just the first chunk (`output_000.mp3`) after splitting with `-segment_time 10000` seconds — which is indeed ~2h 46m per chunk.

So you have roughly:
- 10 hours total ÷ ~2h 46m per chunk = **~4 chunks** (`output_000` through `output_003`)
- At ~15s audio per 1s CPU time: 36,000s audio ÷ 15 ≈ **~40 minutes total CPU time** (without viz overhead)
- With `--dynamic_eventogram` rendering, likely **longer**

You can check how many chunks were created:
```bash
ls "/mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg_split/"*.mp3 | wc -l
```

And the loop is already chugging through them all automatically thanks to the `*` glob — so nothing to do but wait for the `time` output at the end!

---

# you asked

message time: 2026-06-04 10:07:54

#!/usr/bin/env python3
# See: origin https://github.com/qiuqiangkong/audioset_tagging_cnn, heavily modified.
# Uses two .py files that should be in the same folder for imports: pytorch_utils.py and models.py.

# AI EXECUTION CONTEXT WARNING:
# This script is designed to be run from the PROJECT ROOT directory
# (the parent directory of this `pytorch` folder).
#
# MANDATORY ACTION: The Current Working Directory (CWD) MUST be the project root.
# This is because imported modules (like `config.py`) use relative paths
# (e.g., 'metadata/class_labels_indices.csv') that are resolved from the CWD.
#
# Correct execution example:
# `cd /path/to/audioset_tagging_cnn/`
# `python3 pytorch/audioset_tagging_cnn_inference_6.py /path/to/video.mp4 --checkpoint_path ...`
#
# Note: Paths provided as command-line arguments should be absolute to avoid ambiguity.



import os
import sys
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#Torch and torchaudio and coverage are version sensitive.  Use apt for that if you can. Be careful with below over apt parallel install then: python -m pip install torch torchaudio torchcodec --upgrade --extra-index-url https://download.pytorch.org/whl/cpu

import torch
# Handle version-sensitive imports: Torchaudio is essential for tensor-land processing and CUDA efficiency.
try:
    import torchaudio
except (OSError, ImportError) as e:
    print(f"\033[1;31mERROR: torchaudio and torch are not compatible ({e}). We stop.\033[0m")
    print("Please synchronize your versions to fix the 'undefined symbol' or import error:")
    print("\033[1;32mpip install -U torch torchaudio torchcodec --extra-index-url https://download.pytorch.org/whl/cpu\033[0m")
    sys.exit(1)

try:
    import torchcodec
except (OSError, ImportError, RuntimeError):
    torchcodec = None
# AI ARCHITECTURAL NOTE: torchaudio is essential for:
# 1. High-level Decoding/Normalization/Resampling to 32kHz (fixed model rate).
# 2. GPU/CUDA efficiency on large files.
# 3. Direct Tensor-land processing to minimize NumPy/Memory overhead in Termux.
# If torchaudio.load fails on Android, prefer the 'android=linux' torchcodec hack
# over full library replacement to preserve these architectural benefits.
import csv
import datetime
import time
import subprocess
import shutil
import moviepy
import warnings
import soundfile as sf
import psutil
#import coverage 

from moviepy import AudioFileClip, VideoClip
import json
import collections
import plotly.offline as pyo
import tempfile # Import tempfile for temporary file handling

# Suppress torchaudio deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

# Add utils directory to sys.path
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../utils'))
from utilities import create_folder, get_filename
from models import *
from pytorch_utils import move_data_to_device
import config

    



def get_media_metadata(input_media_path):
    """
    Extract duration, FPS, resolution, and is_video flag from a media file using FFprobe.
    Handles audio-only and video files universally, prioritizing format duration.
    Returns: (duration, avg_fps, width, height, is_video, r_fps)
    """
    try:
        # Run ffprobe to get format and stream info
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-print_format', 'json', input_media_path],
            capture_output=True, text=True
        )
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            print(f"\033[1;31mError: Failed to parse FFprobe JSON output for {input_media_path}: {e}\033[0m")
            data = {}

        streams = data.get('streams', [])
        format_info = data.get('format', {})
        
        duration, avg_fps, width, height, r_fps = None, None, None, None, None
        is_video = any(s['codec_type'] == 'video' and s.get('codec_name') not in ['mjpeg', 'png'] for s in streams)

        # Try format duration first (most reliable)
        if format_info.get('duration'):
            try:
                duration = float(format_info['duration'])
            except (ValueError, TypeError):
                print(f"\033[1;33mWarning: Invalid format duration in {input_media_path}\033[0m")

        # Check video stream for duration, FPS, and resolution
        video_stream = next((s for s in streams if s['codec_type'] == 'video'), None)
        if video_stream:
            if duration is None and video_stream.get('duration'):
                try:
                    duration = float(video_stream['duration'])
                except (ValueError, TypeError):
                    print(f"\033[1;33mWarning: Invalid video stream duration in {input_media_path}\033[0m")
            
            # Average FPS
            avg_fr = video_stream.get('avg_frame_rate')
            if avg_fr and '/' in avg_fr:
                try:
                    num, den = map(int, avg_fr.split('/'))
                    avg_fps = num / den if den else None
                except (ValueError, TypeError): pass
            
            # Real FPS (for VFR check)
            r_fr = video_stream.get('r_frame_rate')
            if r_fr and '/' in r_fr:
                try:
                    num, den = map(int, r_fr.split('/'))
                    r_fps = num / den if den else None
                except (ValueError, TypeError): pass
            
            width = int(video_stream.get('width', 0)) if video_stream.get('width') else None
            height = int(video_stream.get('height', 0)) if video_stream.get('height') else None
            
            if duration is None and video_stream.get('nb_frames') and avg_fps:
                try:
                    duration = int(video_stream['nb_frames']) / avg_fps
                except (ValueError, TypeError):
                    print(f"\033[1;33mWarning: Invalid nb_frames or fps for duration calculation in {input_media_path}\033[0m")

        # Check audio stream fallback
        if duration is None:
            audio_stream = next((s for s in streams if s['codec_type'] == 'audio'), None)
            if audio_stream and audio_stream.get('duration'):
                try:
                    duration = float(audio_stream['duration'])
                except (ValueError, TypeError):
                    print(f"\033[1;33mWarning: Invalid audio stream duration in {input_media_path}\033[0m")

        # Final Fallback: Direct duration probe
        if duration is None:
            try:
                result = subprocess.run(
                    ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_media_path],
                    capture_output=True, text=True
                )
                duration = float(result.stdout.strip())
            except (subprocess.CalledProcessError, ValueError, TypeError) as e:
                print(f"\033[1;33mWarning: Fallback duration probe failed for {input_media_path}: {e}\033[0m")

        # If duration is still None, exit with error
        if duration is None:
            print(f"\033[1;31mError: Could not determine duration for {input_media_path}. Exiting.\033[0m")
            return None, None, None, None, False, None

        # --- USER INTERFACE & LOGGING ---
        duration_str = str(datetime.timedelta(seconds=int(duration))) if duration else "?"
        print(f"⏲  🗃️  File duration: \033[1;34m{duration_str}\033[0m")
        if avg_fps:
            print(f"🮲  🗃️  Video FPS (avg): \033[1;34m{avg_fps:.3f}\033[0m")
        if width and height:
            print(f"📽  🗃️  Video resolution: \033[1;34m{width}x{height}\033[0m")
            if height > width:
                print("\n\033[1;33m--- WARNING: Portrait Video Detected ---\033[0m")
                print("\nProcessing a portrait video may result in a narrow or distorted overlay.")
                print("For best results, it is recommended to rotate the video to landscape before processing.")
                print("\nYou can use the following ffmpeg command to rotate the video:\n")
                print(f"\033[1;32mffmpeg -i \"{input_media_path}\" -vf \"transpose=1\" \"{os.path.splitext(input_media_path)[0]}_rotated.mp4\"\033[0m\n")
                print("To do so, break the script now (Ctrl+C) and run the command above.")
                print("Otherwise, we are proceeding with the portrait video in 2 seconds...")
                time.sleep(2)
                print("\033[1;33m------------------------------------------\033[0m\n")

        return duration, avg_fps, width, height, is_video, r_fps

    except Exception as e:
        print(f"\033[1;31mError: Metadata probe failed for {input_media_path}: {e}\033[0m")
        return None, None, None, None, False, None

def compute_kl_divergence(p, q, eps=1e-10):
    """Compute KL divergence between two probability distributions."""
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * np.log(p / q))

def get_dynamic_top_events(framewise_output, start_idx, end_idx, top_k=10):
    """Get top k events for a given window of framewise_output."""
    window_output = framewise_output[start_idx:end_idx]
    if window_output.shape[0] == 0:
        return np.array([]), np.array([])
    max_probs = np.max(window_output, axis=0)
    sorted_indexes = np.argsort(max_probs)[::-1][:top_k]
    return window_output[:, sorted_indexes], sorted_indexes

def audio_tagging(args):
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    classes_num = config.classes_num
    labels = config.labels

    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
                  hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
                  classes_num=classes_num)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
    except Exception as e:
        print(f"\033[1;31mError loading model checkpoint: {e}\033[0m")
        return

    if device.type == 'cuda':
        model.to(device)
        print(f'GPU number: {torch.cuda.device_count()}')
        model = torch.nn.DataParallel(model)

    waveform, sr = torchaudio.load(audio_path)
    print(f"Loaded waveform shape: {waveform.shape}, sample rate: {sr}")
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
    waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
    waveform = waveform[None, :]  # Shape: [1, samples] for model input
    
    
    
    waveform = move_data_to_device(waveform, device)

    with torch.no_grad():
        model.eval()
        try:
            batch_output_dict = model(waveform, None)
        except Exception as e:
            print(f"\033[1;31mError during model inference: {e}\033[0m")
            return

    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
    sorted_indexes = np.argsort(clipwise_output)[::-1]

    print('Sound events detection result (time_steps x classes_num): {}'.format(clipwise_output.shape))

    for k in range(10):
        print('{}: {}'.format(np.array(labels)[sorted_indexes[k]], clipwise_output[sorted_indexes[k]]))

    if 'embedding' in batch_output_dict:
        embedding = batch_output_dict['embedding'].data.cpu().numpy()[0]
        print('embedding: {}'.format(embedding.shape))

    return clipwise_output, labels

def sound_event_detection(args):
    # --- PHASE 0: Setup & Environment ---
    output_fps = args.output_fps
    viz_fps = args.vis_fps  # Fixed constant for internal resolution
    adaptive_lookahead = args.adaptive_lookahead
    
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    inference_fps = sample_rate // hop_size # Typically 100
    
    top_k = 10  # Number of top events to track/visualize
    fig_width_px, fig_height_px, dpi = 1280, 480, 100
    
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    
    # Path Consolidation: Clearly define the roles of each media path
    source_media = args.audio_path    # The original file (never changed)
    inference_media = source_media   # The file used for duration/audio loading (may be updated to recovered/sanitized)
    overlay_media = source_media     # The file used for the final video overlay
    
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    else:
        print('Using CPU.')

    classes_num = config.classes_num
    labels = config.labels
    
    audio_dir = os.path.dirname(source_media)
    base_name = get_filename(source_media)
    checkpoint_name = os.path.basename(checkpoint_path)
    output_dir = os.path.join(audio_dir, f'{base_name}_{checkpoint_name}_audioset_tagging_cnn')
    create_folder(output_dir)

    # --- PHASE 1: Dependency Injection (AI Guide) ---
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        guide_src_path = os.path.normpath(os.path.join(script_dir, '..', 'docs', 'auditory_cognition_guide_template.md'))
        guide_dest_path = os.path.join(output_dir, 'auditory_cognition_guide_template.md')
        
        if os.path.exists(guide_src_path):
            shutil.copy(guide_src_path, guide_dest_path)
            print(f'Copied AI analysis guide to: \033[1;34m{guide_dest_path}\033[1;0m')
        else:
            print(f'\033[1;33mWarning: AI analysis guide not found at {guide_src_path}\033[0m')
    except Exception as e:
        print(f'\033[1;33mWarning: Failed to copy AI analysis guide: {e}\033[0m')

    tag_suffix = "_audioset_tagging_cnn"
    fig_path = os.path.join(output_dir, 'eventogram.png')

    # --- PHASE 2: Idempotency Check ---
    csv_path = os.path.join(output_dir, 'full_event_log.csv')
    
    # Efficiently probe all metadata once
    duration, video_fps, video_width, video_height, is_video, r_fps = get_media_metadata(source_media)
    if duration is None: return # Error already printed in probe

    # Define what files we strictly expect to see before skipping
    required_files = [csv_path]
    if args.static_eventogram:
        vid_path = os.path.join(output_dir, f'{base_name}{tag_suffix}_eventogram_static.mp4')
        required_files.append(f"{os.path.splitext(vid_path)[0]}_overlay.mp4" if is_video else vid_path)
            
    if args.dynamic_eventogram:
        vid_path = os.path.join(output_dir, f"{base_name}{tag_suffix}_eventogram_dynamic.mp4")
        required_files.append(f"{os.path.splitext(vid_path)[0]}_overlay.mp4" if is_video else vid_path)
            
    if all(os.path.exists(f) for f in required_files):
        print(f"✅ Skipping {source_media}, all requested outputs already exist in: \033[1;34m{output_dir}\033[1;0m")
        return

    # Check for sufficient disk space
    disk_usage = shutil.disk_usage(audio_dir)
    if disk_usage.free < 1e9:
        print(f"\033[1;31mError: Insufficient disk space ({disk_usage.free / 1e9:.2f} GB free). Exiting.\033[0m")
        return
    
    # --- PHASE 3: Model Loading ---
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
                  hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
                  classes_num=classes_num)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
    except Exception as e:
        print(f"\033[1;31mError loading model checkpoint: {e}\033[0m")
        return

    # --- PHASE 4: Media Integrity & Recovery ---
    # Recovery: Attempt conversion to MP3 if duration detection returned 0
    if duration == 0:
        print(f"\033[1;33mWarning: Duration probe failed for {inference_media}. Attempting MP3 recovery...\033[0m")
        mp3_path = os.path.join(tempfile.gettempdir(), f"{base_name}_recovered.mp3")

        try:
            subprocess.run(['ffmpeg', '-i', inference_media, '-y', mp3_path], check=True, capture_output=True)
            inference_media = mp3_path # Rest of the script uses this recovered file
            duration, video_fps, video_width, video_height, is_video, r_fps = get_media_metadata(inference_media)
            
            if duration is None or duration == 0:
                print(f"\033[1;31mError: Could not determine duration even after recovery. Exiting.\033[0m")
                return
        except Exception as e:
            print(f"\033[1;31mRecovery conversion failed: {e}\033[0m")
            return

    if is_video and (video_width is None or video_height is None):
        video_width, video_height = 1280, 720
        print(f"\033[1;33mWarning: Video dimensions not detected, using default 1280x720.\033[0m")
    
    # --- PHASE 5: VFR Check & Constant Frame Rate Sanitization ---
    temp_video_path = None
    if is_video and r_fps and video_fps:
        if abs(r_fps - video_fps) > 0.01:
            print(f"\033[1;33mDetected VFR ({r_fps:.2f}/{video_fps:.2f}). Re-encoding to CFR for sync...\033[0m")
            temp_video_path = os.path.join(tempfile.gettempdir(), f'temp_cfr_{base_name}.mp4')
            target_fps = video_fps if video_fps > 0 else output_fps
            # --- FFmpeg Compatibility Note ---
            # We use '-vsync 1' (Constant Frame Rate) instead of the newer '-fps_mode cfr'.
            # Rationale: '-fps_mode' was introduced in FFmpeg 5.1 (late 2022). 
            # Many stable systems (like Ubuntu 20.04) still use FFmpeg 4.x, which only 
            # recognizes '-vsync'. While '-vsync' is deprecated in FFmpeg 5.1+, it 
            # remains a functional alias in FFmpeg 6.x/7.x. 
            # If a future FFmpeg version (e.g., v10+) removes '-vsync', switch this back 
            # to: '-fps_mode', 'cfr'
            subprocess.run([
                'ffmpeg', '-loglevel', 'warning', '-i', inference_media, '-r', str(target_fps), '-vsync', '1', '-c:a', 'aac', temp_video_path, '-y'
            ], check=True)
            inference_media = temp_video_path

    # --- PHASE 6: Waveform Loading ---
    waveform, sr = torchaudio.load(inference_media)

    import gc
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
        gc.collect()
        
    waveform = waveform.mean(dim=0, keepdim=True)  # Downmix to mono
    waveform_np = waveform.squeeze(0).numpy() # Move to NumPy for memory-safe chunking
    
    # CRITICAL: Drop torch tensors to free significant RAM immediately (essential for Termux)
    del waveform
    gc.collect()
    waveform = waveform_np 

    # --- PHASE 7: Chunked Model Inference (Memory-Safe) ---
    chunk_duration = 180  # 3 minutes
    chunk_samples = int(chunk_duration * sample_rate)
    vis_downsample = inference_fps // viz_fps
    
    framewise_vis_list = []
    stft_vis_list = []
    
    avail_ram = psutil.virtual_memory().available / (1024 * 1024)
    print(f"📊  Starting inference in {chunk_duration/60}m chunks. (RAM Avail: {avail_ram:.0f} MB)")
    print(f"    Resolution: Disk {inference_fps} Hz (Data Frames) | Visualization {viz_fps} Hz (UI/RAM)")

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time'] + list(labels))
        
        for start in range(0, len(waveform), chunk_samples):
            chunk = waveform[start:start + chunk_samples]
            if len(chunk) < sample_rate // 10: continue
            
            # Step A: Inference
            chunk_tensor = move_data_to_device(torch.from_numpy(chunk[None, :]).float(), device)
            with torch.no_grad():
                model.eval()
                try:
                    batch_output_dict = model(chunk_tensor, None)
                    chunk_out = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
                except Exception as e:
                    print(f"\033[1;31mInference error in chunk: {e}\033[0m"); continue
            
            # Step B: Write high-res results to disk immediately (Lean memory)
            chunk_start_time = start / sample_rate
            csv_downsample = max(1, inference_fps // args.csv_fps)
            for i in range(0, len(chunk_out), csv_downsample):
                timestamp = chunk_start_time + (i / inference_fps)
                writer.writerow([round(timestamp, 3)] + chunk_out[i].tolist())
            
            # Step C: Downsample for RAM-based visualization (Max-pooling preserves short events)
            for i in range(0, len(chunk_out), vis_downsample):
                vis_slice = chunk_out[i : i + vis_downsample]
                if len(vis_slice) > 0:
                    framewise_vis_list.append(np.max(vis_slice, axis=0))

            # Step D: Chunked STFT for visualization background
            chunk_tensor_stft = torch.from_numpy(chunk).to(device)
            chunk_stft = torch.stft(
                chunk_tensor_stft, n_fft=window_size, hop_length=hop_size,
                window=torch.hann_window(window_size).to(device),
                center=True, return_complex=True
            ).abs().cpu().numpy()
            stft_vis_list.append(chunk_stft[:, ::vis_downsample])
            
            avail_ram = psutil.virtual_memory().available / (1024 * 1024)
            print(f"Chunk at {int(chunk_start_time/60)}m finished. (RAM Avail: {avail_ram:.0f} MB)")

    # --- PHASE 8: Result Aggregation & Metadata Consolidation ---
    framewise_output = np.array(framewise_vis_list)
    stft = np.concatenate(stft_vis_list, axis=1)
    del framewise_vis_list, stft_vis_list
    
    # DATA-DRIVEN DURATION: Use real data length as truth (fixes VBR/probe guesses)
    frames_num = len(framewise_output)
    duration = frames_num / viz_fps
    
    print(f'Aggregation complete. Internal Viz resolution: \033[1;34m{viz_fps} Hz (Data Frames)\033[1;0m')
    print(f'Final analysis duration: \033[1;34m{duration:.2f}s\033[1;0m')

    # --- PHASE 9: Static Eventogram Generation (PNG) ---
    sorted_indexes = np.argsort(np.max(framewise_output, axis=0))[::-1]
    top_result_mat = framewise_output[:, sorted_indexes[0:top_k]]
    top_labels = np.array(labels)[sorted_indexes[0:top_k]]

    fig = plt.figure(figsize=(fig_width_px / dpi, fig_height_px / dpi), dpi=dpi)
    
    # Pass 1: Create dummy plot to measure Y-axis label pixel width (Dynamic Margin)
    gs_init = fig.add_gridspec(2, 1, height_ratios=[1, 1], left=0.1, right=1.0, top=0.95, bottom=0.08, hspace=0.05)
    axs_init = [fig.add_subplot(gs_init[0]), fig.add_subplot(gs_init[1])]
    axs_init[1].set_yticks(np.arange(0, top_k)); axs_init[1].set_yticklabels(top_labels, fontsize=14)
    fig.canvas.draw()
    
    max_label_width_px = max(lbl.get_window_extent(renderer=fig.canvas.get_renderer()).width for lbl in axs_init[1].yaxis.get_majorticklabels())
    left_frac = min(0.45, (max_label_width_px + 14) / fig_width_px)
    
    # Pass 2: Final Render with corrected margin
    fig.clear()
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], left=left_frac, right=1.0, top=0.95, bottom=0.08, hspace=0.05)
    axs = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]

    axs[0].matshow(np.log(stft + 1e-10), origin='lower', aspect='auto', cmap='jet')
    axs[0].set_ylabel('Freq bins', fontsize=14); axs[0].set_title('Spectrogram and Eventogram', fontsize=14)
    axs[1].matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)

    tick_interval = max(5, int(duration / 20))
    x_ticks = np.arange(0, frames_num, viz_fps * tick_interval)
    axs[1].xaxis.set_ticks(x_ticks)
    axs[1].xaxis.set_ticklabels([int(t / viz_fps) for t in x_ticks], rotation=45, ha='right', fontsize=10)
    axs[1].set_xlim(0, frames_num); axs[1].set_yticks(np.arange(0, top_k)); axs[1].set_yticklabels(top_labels, fontsize=14)
    axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
    axs[1].set_xlabel('Seconds', fontsize=14); axs[1].xaxis.set_ticks_position('bottom')
    
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0); plt.close(fig)
    print(f'Saved eventogram PNG to: \033[1;34m{fig_path}\033[1;0m')




    # --- PHASE 10: AI-Friendly Summary Generation (Event Block Detection) ---
    print("📊  Generating AI-friendly event summary files…")
    summary_csv_path = os.path.join(output_dir, 'summary_events.csv')
    
    # Event detection heuristics
    onset_threshold, offset_threshold = 0.01, 0.01
    min_event_duration_seconds = 0.5 
    top_n_per_class = 3 # Only export the strongest N events per class

    events_by_class = {label: [] for label in top_labels}

    # Identify continuous blocks of sound for the top classes shown in the eventogram
    for label_idx, label in enumerate(labels):
        if label not in top_labels: continue
        
        in_event, event_start_frame = False, 0
        for frame_index, prob in enumerate(framewise_output[:, label_idx]):
            if not in_event and prob > onset_threshold:
                in_event, event_start_frame = True, frame_index
            elif in_event and prob < offset_threshold:
                in_event = False
                duration_frames = frame_index - event_start_frame
                duration_secs = duration_frames / viz_fps
                
                if duration_secs >= min_event_duration_seconds:
                    event_block_probs = framewise_output[event_start_frame:frame_index, label_idx]
                    events_by_class[label].append({
                        'sound_class': label,
                        'start_time_seconds': round(event_start_frame / viz_fps, 3),
                        'end_time_seconds': round(frame_index / viz_fps, 3),
                        'duration_seconds': round(duration_secs, 3),
                        'peak_probability': float(np.max(event_block_probs)),
                        'average_probability': float(np.mean(event_block_probs))
                    })

    # Consolidate and sort top events chronologically
    top_events = []
    for label, events in events_by_class.items():
        if events:
            top_events.extend(sorted(events, key=lambda x: x['peak_probability'], reverse=True)[:top_n_per_class])
    
    final_sorted_events = sorted(top_events, key=lambda x: x['start_time_seconds'])

    with open(summary_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['sound_class', 'start_time_seconds', 'end_time_seconds', 'duration_seconds', 'peak_probability', 'average_probability']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(final_sorted_events)
    print(f'Saved summary events CSV to: \033[1;34m{summary_csv_path}\033[1;0m')

    # --- PHASE 11: Detailed AI-Friendly Event Delta Map (RLE Compression) ---
    # This format is optimized for AI attention mechanisms: it uses Run-Length Encoding (RLE)
    # and probability deltas to focus on "attacks" and "decays".
    print("📊  Generating detailed AI-friendly event delta JSON (RLE optimized)…")
    json_ai_path = os.path.join(output_dir, 'detailed_events_delta_ai_attention_friendly.json')
    ai_threshold = 0.05
    events_derivative = collections.defaultdict(list)

    def zip_trace(trace):
        """Compress trace by replacing consecutive zeros with a skip marker."""
        zipped = []
        zero_count = 0
        for val in trace:
            if val == 0.0:
                zero_count += 1
            else:
                if zero_count > 0:
                    zipped.append({"skip": zero_count}); zero_count = 0
                zipped.append(val)
        if zero_count > 0: zipped.append({"skip": zero_count})
        return zipped

    for label_idx, label in enumerate(labels):
        current_event = None
        probs_stream = framewise_output[:, label_idx]
        
        for frame_index, prob in enumerate(probs_stream):
            if prob > ai_threshold:
                if current_event is None:
                    current_event = {
                        "start": round(frame_index / viz_fps, 3),
                        "peak": float(prob), "trace": [float(prob)]
                    }
                else:
                    current_event["peak"] = max(current_event["peak"], float(prob))
                    current_event["trace"].append(float(prob))
            elif current_event:
                # Finalize event and calculate momentum (deltas)
                tr = current_event["trace"]
                deltas = [tr[0]] # Anchor point
                for i in range(1, len(tr)): deltas.append(round(tr[i] - tr[i-1], 6))
                
                events_derivative[label].append({
                    "start_time": current_event["start"],
                    "end_time": round(frame_index / viz_fps, 3),
                    "peak_prob": current_event["peak"],
                    "delta_trace": zip_trace(deltas)
                })
                current_event = None
        
        if current_event: # Cleanup trailing event
            tr = current_event["trace"]
            deltas = [tr[0]]
            for i in range(1, len(tr)): deltas.append(round(tr[i] - tr[i-1], 6))
            events_derivative[label].append({
                "start_time": current_event["start"],
                "end_time": round(len(probs_stream) / viz_fps, 3),
                "peak_prob": current_event["peak"],
                "delta_trace": zip_trace(deltas)
            })

    with open(json_ai_path, 'w') as f:
        json.dump({k: v for k, v in events_derivative.items() if v}, f, indent=2)
    print(f'Saved AI-friendly delta JSON to: \033[1;34m{json_ai_path}\033[1;0m')

    # --- PHASE 12: Interactive Plotly Dashboard (Top 50 Events) ---
    print("📊  Generating interactive Plotly dashboard…")
    html_dashboard_path = os.path.join(output_dir, 'interactive_dashboard.html')
    
    # Identify top 50 classes by total popularity
    popularity = np.sum(framewise_output, axis=0)
    top_50_indices = np.argsort(popularity)[::-1][:50]
    
    times = [round(i / viz_fps, 2) for i in range(frames_num)]
    traces_data = []
    for idx in top_50_indices:
        traces_data.append({"name": labels[idx], "y": np.round(framewise_output[:frames_num, idx], 4).tolist()})
    
    json_payload = json.dumps({"times": times, "traces": traces_data})
    plotly_js = pyo.get_plotlyjs()
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Audio Analysis - {base_name}</title>
    <script type="text/javascript">{plotly_js}</script>
    <style>
        body {{ font-family: sans-serif; margin: 20px; background: #fafafa; color: #333; }}
        #plot {{ width: 100%; height: 85vh; background: white; border-radius: 8px; border: 1px solid #ddd; }}
        .header {{ margin-bottom: 15px; border-left: 5px solid #2563eb; padding-left: 15px; }}
        code {{ background: #eee; padding: 2px 5px; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Sound Event Analysis: {base_name}</h1>
        <p>Interactive Plotly Dashboard | Top 50 Classes | Source: <code>full_event_log.csv</code></p>
    </div>
    <div id="plot"></div>
    <script type="text/javascript">
        const data = {json_payload};
        const traces = data.traces.map((t, index) => ({{
            x: data.times, y: t.y, name: t.name, mode: 'lines',
            visible: index < 10 ? true : 'legendonly', line: {{ width: 2 }}
        }}));
        Plotly.newPlot('plot', traces, {{
            title: 'Top 50 Sound Events Momentum',
            xaxis: {{ title: 'Seconds', gridcolor: '#eee' }},
            yaxis: {{ title: 'Probability', range: [0, 1], gridcolor: '#eee' }},
            hovermode: 'x unified', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            margin: {{ t: 50, b: 50, l: 60, r: 20 }}
        }}, {{ responsive: true }});
    </script>
</body>
</html>
"""
    with open(html_dashboard_path, 'w', encoding='utf-8') as f: f.write(html_content)
    print(f'Saved interactive dashboard to: \033[1;34m{html_dashboard_path}\033[1;0m')

    """
    # Auto-open the dashboard using the system 'open' command
    try:
        subprocess.run(['open', html_dashboard_path], check=False)
    except Exception as e:
        print(f"Warning: Could not auto-open dashboard: {e}")
    """

    


    # --- PHASE 13: Video Rendering Pipeline (Data-Synchronous & Cached) ---
    if args.dynamic_eventogram or args.static_eventogram:
        print(f"🎞  Initializing video rendering pipeline ({output_fps} FPS)…")
        
        # Rendering Strategy: Dynamic (Scrolling) vs Static (Marker only)
        if args.dynamic_eventogram:
            output_video_path = os.path.join(output_dir, f"{base_name}{tag_suffix}_eventogram_dynamic.mp4")
            window_frames = int(args.window_duration * viz_fps)
            half_window = window_frames // 2 

            # PRECOMPUTE: Map every data point to its local acoustic window (once per run)
            print(f"📊  Precomputing {frames_num} windows (Adaptive={args.use_adaptive_window})…")
            precomputed_data = []
            for i in range(frames_num):
                start_f, end_f = max(0, i - half_window), min(frames_num, i + half_window)
                
                # Adaptive logic: Try to center the window on acoustic "peaks" rather than strict time
                if args.use_adaptive_window:
                    kl_threshold = 0.5
                    lookahead_f = int(adaptive_lookahead * viz_fps)
                    for offset in range(half_window, half_window + lookahead_f, int(viz_fps)):
                        if start_f - offset >= 0:
                            prev_p = np.mean(framewise_output[start_f-offset:start_f], axis=0)
                            curr_p = np.mean(framewise_output[start_f:start_f+offset], axis=0)
                            if compute_kl_divergence(prev_p, curr_p) > kl_threshold:
                                start_f = max(0, start_f - offset // 2); break
                        if end_f + offset < frames_num:
                            curr_p = np.mean(framewise_output[end_f-offset:end_f], axis=0)
                            next_p = np.mean(framewise_output[end_f:end_f+offset], axis=0)
                            if compute_kl_divergence(curr_p, next_p) > kl_threshold:
                                end_f = min(frames_num, end_f + offset // 2); break
                
                win_out, win_idxs = get_dynamic_top_events(framewise_output, start_f, end_f, top_k)
                if win_out.size == 0:
                    win_out, win_idxs = np.zeros((end_f - start_f, top_k)), sorted_indexes[:top_k]
                precomputed_data.append({'start': start_f, 'end': end_f, 'out': win_out, 'idxs': win_idxs})

            # Setup Persistent Figure for Sprint Speed (Reuses same canvas objects to avoid GC spikes)
            fig_fr = plt.figure(figsize=(fig_width_px/dpi, fig_height_px/dpi), dpi=dpi)
            gs_fr = fig_fr.add_gridspec(2, 1, height_ratios=[1, 1], left=left_frac, right=1.0, top=0.95, bottom=0.08, hspace=0.05)
            axs_fr = [fig_fr.add_subplot(gs_fr[0]), fig_fr.add_subplot(gs_fr[1])]
            
            stft_log = np.log(stft + 1e-10)
            v_min, v_max = np.percentile(stft_log, [1, 99]) # Fix contrast globally

            # Persistent Artists (The "Engine")
            im_spec = axs_fr[0].imshow(np.zeros((stft.shape[0], 2)), origin='lower', aspect='auto', cmap='jet', vmin=v_min, vmax=v_max)
            im_event = axs_fr[1].imshow(np.zeros((top_k, 2)), origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
            marker_lines = [ax.axvline(x=0, color='red', linewidth=2, alpha=0.8) for ax in axs_fr]
            
            axs_fr[0].set_ylabel('Freq bins', fontsize=14); axs_fr[1].set_xlabel('Seconds', fontsize=14)
            axs_fr[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
            axs_fr[0].tick_params(labelbottom=False, labeltop=False, bottom=False, top=False)
            
            last_idxs = {"val": None}

            def draw_strategy(i):
                """High-speed redraw using set_data instead of creating new plots."""
                data = precomputed_data[i]
                s_f, e_f = data['start'], data['end']
                t_start, t_end, t_curr = s_f / viz_fps, e_f / viz_fps, i / viz_fps
                
                im_spec.set_data(stft_log[:, s_f:e_f]); im_spec.set_extent([t_start, t_end, 0, stft.shape[0]])
                im_event.set_data(data['out'].T); im_event.set_extent([t_start, t_end, 0, top_k])
                
                # Only update labels if the "local context" changed (saves CPU)
                if last_idxs["val"] is None or not np.array_equal(last_idxs["val"], data['idxs']):
                    axs_fr[1].set_yticks(np.arange(0.5, top_k + 0.5))
                    axs_fr[1].set_yticklabels(np.array(labels)[data['idxs']][::-1], fontsize=14)
                    last_idxs["val"] = data['idxs']
                
                axs_fr[0].set_title(f'Spectrogram and Eventogram (t={t_curr:.1f}s)', fontsize=14)
                for ax, line in zip(axs_fr, marker_lines):
                    ax.set_xlim(t_start, t_end); line.set_xdata([t_curr, t_curr])
                
                fig_fr.canvas.draw()
                return np.frombuffer(fig_fr.canvas.buffer_rgba(), dtype=np.uint8).reshape((fig_height_px, fig_width_px, 4))[:,:,:3]

        else: # Static Eventogram (Just a moving red line over the PNG)
            output_video_path = os.path.join(output_dir, f'{base_name}{tag_suffix}_eventogram_static.mp4')
            base_img = plt.imread(fig_path)
            if base_img.dtype == np.float32: base_img = (base_img * 255).astype(np.uint8)
            base_img = base_img[:, :, :3]
            h, w, _ = base_img.shape
            x_start, x_end = int(left_frac * w), w

            def draw_strategy(i):
                img = base_img.copy()
                marker_x = int(x_start + (x_end - x_start) * (i / max(frames_num - 1, 1)))
                img[:, max(0, marker_x-1):min(w, marker_x+1)] = [255, 0, 0]
                return img

        # --- SMART CACHE LAYER ---
        # Since video FPS (30) >> data Hz (5), we reuse the last rendered frame
        # for 6 consecutive video frames to save ~80% of CPU rendering time.
        frame_cache = {"last_i": -1, "last_img": None}

        def make_frame(t):
            i = min(int(t * viz_fps), frames_num - 1)
            if i == frame_cache["last_i"]: return frame_cache["last_img"]
            
            img = draw_strategy(i)
            frame_cache.update({"last_i": i, "last_img": img})
            return img

        # Final Compositing & Export
        eventogram_clip = VideoClip(make_frame, duration=duration)
        audio_clip = AudioFileClip(inference_media)
        final_clip = eventogram_clip.with_audio(audio_clip)
        final_clip.fps = output_fps
        
        final_clip.write_videofile(
            output_video_path, codec="libx264", fps=output_fps, 
            threads=os.cpu_count(),
            temp_audiofile=os.path.join(output_dir, "temp_render_audio.mp3")
        )
        print(f"✅ Saved eventogram video to: \033[1;34m{output_video_path}\033[1;0m")
        if args.dynamic_eventogram: plt.close(fig_fr)

        # --- PHASE 14: FFmpeg Overlay (Video-on-Video) ---
        if is_video:
            print("🎬  Overlaying the source media with the created eventogram…")
            final_overlay_path = f"{os.path.splitext(output_video_path)[0]}_overlay.mp4"
            
            # Use pre-probed dimensions (no redundant FFprobe call)
            b_w, b_h = video_width, video_height
            
            # Leaner Scaling: Match eventogram width (1280px) if source is smaller; force even height
            t_w = max(b_w, fig_width_px)
            t_h = int(b_h * t_w / b_w) // 2 * 2 
            
            overlay_cmd = [
                "ffmpeg", "-y", "-i", overlay_media, "-i", output_video_path, "-loglevel", "warning",
                "-filter_complex", (
                    f"[0:v]scale={t_w}:{t_h}[main];" # Scale source video
                    f"[1:v]scale={t_w}:{int(t_h * args.overlay_size)}[ovr];" # Scale eventogram
                    f"[ovr]format=rgba,colorchannelmixer=aa={args.translucency}[ovr_t];" # Set alpha
                    "[main][ovr_t]overlay=x=0:y=H-h[v]" # Stack them at the bottom
                ),
                "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest"
            ]
            if args.bitrate: overlay_cmd.extend(["-b:v", args.bitrate])
            else: overlay_cmd.extend(["-crf", str(args.crf)])
            
            overlay_cmd.append(final_overlay_path)
            
            try:
                subprocess.run(overlay_cmd, check=True)
                print(f"✅ 🎥 Final overlay saved to: \033[1;34m{final_overlay_path}\033[1;0m")
            except subprocess.CalledProcessError as e:
                print(f"\033[1;31mError during FFmpeg overlay: {e}\033[0m")
            
            # Cleanup temporary CFR file if created
            if temp_video_path and os.path.exists(temp_video_path): os.remove(temp_video_path)
        else:
            print("🎧 Source is audio-only, skipping video overlay.")

    print(f"⏲  🗃️  Analysis finished. Input duration: \033[1;34m{duration:.2f}s\033[0m")

if __name__ == '__main__':



    

    parser = argparse.ArgumentParser(
        description='Audio tagging and Sound event detection.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=""" """
    )
    parser.add_argument('audio_path', type=str, help='Path to the media file')
    parser.add_argument('--mode', choices=['audio_tagging', 'sound_event_detection'],
                        default='sound_event_detection', help='Select the processing mode')
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--mel_bins', type=int, default=64)
    parser.add_argument('--fmin', type=int, default=50)
    parser.add_argument('--fmax', type=int, default=14000)
    parser.add_argument('--model_type', type=str, required=True, help='Model architecture type')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to pretrained .pth file')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use CUDA for inference')
    parser.add_argument('--translucency', type=float, default=0.7,
                        help='Overlay translucency (0 to 1)')
    parser.add_argument('--overlay_size', type=float, default=0.2,
                        help='Overlay size as fraction of video height')
    parser.add_argument('--dynamic_eventogram', action='store_true', default=False,
                        help='Generate dynamic eventogram with scrolling window (slower to generate)')
    parser.add_argument('--static_eventogram', action='store_true', default=False,
                        help='Generate static eventogram with a scrolling marker (faster to generate)')                        
    parser.add_argument('--crf', type=int, default=23, help='FFmpeg CRF value (0-51, lower is higher quality)')
    parser.add_argument('--bitrate', type=str, default=None, help='FFmpeg video bitrate (e.g., "2000k" for 2 Mbps)')
    parser.add_argument('--window_duration', type=float, default=30.0,
                        help='Duration of sliding window for the dynamic eventogram (in seconds)')
    parser.add_argument('--use_adaptive_window', action='store_true', default=False,
                        help='Use adaptive window size based on the event boundaries')
    parser.add_argument('--csv_fps', type=int, default=5,
                        help='Data frames per second (Hz) to write to full_event_log.csv. '
                             'Use 100 for full resolution (very large file). '
                             '5 is enough for Shapash + outlier detection.')
    parser.add_argument('--vis_fps', type=int, default=5,
                        help='Data frames per second (Hz) for internal RAM-based visualization data (RAM guard)')
    parser.add_argument('--output_fps', type=int, default=30,
                        help='FPS for the final rendered video output')
    parser.add_argument('--adaptive_lookahead', type=float, default=30.0,
                        help='Max seconds to look ahead/back for acoustic boundaries in adaptive mode')

    # Heuristic to find audio_path for the help/info block
    audio_path_hint = "[audio_path]"
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if not arg.startswith('-'):
                audio_path_hint = arg
                break                        
                        
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
 
    print(f"Eventogrammer, version 6.8.11") 
    print(f"Adaptation of: https://github.com/qiuqiangkong/audioset_tagging_cnn")
    print()

    print(f"Recent Material Changes:")
    print(f"* Constants Promoted: vis_fps, output_fps, and adaptive_lookahead are now CLI arguments.")
    print(f"* Speed Hack: Persistent Matplotlib figures with Artist Updates.")
    print(f"* Visual Fix: Proper window centering for scrolling eventograms.")
    print(f"* Refactor: Gemini AI rationalized path management and code structure.")
    print()

    print(f"Note on the models:")   
    print(f"* Cnn14_DecisionLevelMax (Sound Event Detection): Uses Decision-level pooling to maintain")
    print(f"  time resolution. Essential for generating Eventograms and high-res CSV logs.")
    print(f"* Other models: Best for global audio tagging (use the '--audio_tagging' mode).")
    print()

    print(f"Performance & Stability:") 
    print(f"* Processing ratio: ~15s audio per 1s CPU time (300 GFLOPs, 4-core, no viz).") 
    print(f"* Platform Gap: works ~1.7x faster in Prooted Debian than in Termux (Eigen BLAS).")  
    print(f"* OOM Safety: Close browsers or restart whole phone if crashes occur in Termux.")  
    print()

    print(f"Split Suggestion:") 
    print(f"If the file is too long, use FFmpeg to segment it first:") 
    print(f"mkdir split_input_media && cd split_input_media && \\")
    print(f"ffmpeg -i {audio_path_hint} -c copy -f segment -segment_time 1200 output_%03d.mp4")
    print(f"time bash -c 'for file in ""*; do bash audio_me.sh \"$file\" --dynamic_eventogram; done'")
    print()

    print("Tips & Environment Hacks:")
    print("* For 'undefined symbol: torch_library_impl' or 'NotImplementedError':")
    print("  Run: pip install -U torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu")
    print("* If Torchcodec errors (e.g., 'libnvrtc.so.13 not found'), simply remove it:")
    print("  Run: pip uninstall torchcodec  (The script will safely fallback to FFmpeg)")
    print("* If you see 'NotImplementedError: sys.platform = android' after an update:")
    print(f"  Edit: /data/data/com.termux/files/usr/lib/python{py_ver}/site-packages/torchaudio/_internally_replaced_utils.py")
    print("  Change 'if sys.platform == \"linux\":' to 'if sys.platform == \"android\":'")
    print()

    print(f"Dependency Versions:")
    print(f"MoviePy: {moviepy.__version__}")
    print(f"Torchaudio: {torchaudio.__version__ if torchaudio else 'Not Available'}")
    # May need to be disabled as it errors if installed some weird version 0 dev.
    print(f"Torchcodec: {torchcodec.__version__ if torchcodec else 'Not Available'}")
 
    print()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    audio_path = args.audio_path


    if args.mode == 'audio_tagging':
        audio_tagging(args)
    else:
        sound_event_detection(args)

 
    print()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    audio_path = args.audio_path


    if args.mode == 'audio_tagging':
        audio_tagging(args)
    else:
        sound_event_detection(args)

Yes. Speaking of which: do you see a way to avoid chunking via `ffmpeg` while avoiding OOMs (which happens if smth longer than 3 hours, on my 8 GB RAM + 8 GB swap box)?

---

# claude response

Yes, looking at your code, the chunking is already handled **internally** in PHASE 7 (`chunk_duration = 180` seconds). The FFmpeg pre-split you did was unnecessary — the script already processes in 3-minute chunks and streams results to CSV immediately, keeping RAM flat.

The OOM likely happens not in inference but in **PHASE 13 (video rendering)**, specifically these two culprits:

**1. `precomputed_data` (the biggest offender)**
```python
precomputed_data = []
for i in range(frames_num):
    ...
    precomputed_data.append({'start': ..., 'end': ..., 'out': win_out, 'idxs': win_idxs})
```
For a 10-hour file at `viz_fps=5`: `frames_num = 10*3600*5 = 180,000` frames, each storing a `(window_frames × 10)` float64 array. That's enormous.

**2. `framewise_output` and `stft` arrays**
```python
framewise_output = np.array(framewise_vis_list)  # 180,000 × 527 floats
stft = np.concatenate(stft_vis_list, axis=1)      # freq_bins × 180,000 floats
```
At float32, `framewise_output` alone is `180,000 × 527 × 4 bytes ≈ 360 MB`. The STFT is similar. Fine individually, but combined with `precomputed_data` it compounds fast.

---

**Fixes without FFmpeg pre-splitting:**

The cleanest solution is to **never precompute all windows** — compute them lazily inside `make_frame()` instead:

```python
# DELETE the entire precompute block (the for i in range(frames_num) loop)
# and replace draw_strategy with:

def make_frame(t):
    i = min(int(t * viz_fps), frames_num - 1)
    if i == frame_cache["last_i"]:
        return frame_cache["last_img"]
    
    # Compute on-the-fly instead of from precomputed_data
    s_f = max(0, i - half_window)
    e_f = min(frames_num, i + half_window)
    win_out, win_idxs = get_dynamic_top_events(framewise_output, s_f, e_f, top_k)
    if win_out.size == 0:
        win_out, win_idxs = np.zeros((e_f - s_f, top_k)), sorted_indexes[:top_k]
    
    # ... rest of draw_strategy inline here
```

This trades a one-time RAM spike for slightly more CPU per frame, but since `make_frame` is already cached (`frame_cache`), and `viz_fps=5` while `output_fps=30` means **6 video frames reuse each computation**, the CPU cost is minimal.

The adaptive window logic (`args.use_adaptive_window`) is the only thing that genuinely benefits from precomputation — but you're not using it by default, so this is safe.

---

**Secondary fix:** reduce STFT memory by not storing it at all for audio-only files, since it's only used for the spectrogram backdrop and could be recomputed per-window on demand the same way.

---

# you asked

message time: 2026-06-04 10:09:13

Partly wrong: "
10:07 AM


Yes, looking at your code, the chunking is already handled internally in PHASE 7 (`chunk_duration = 180` seconds)." 

That is: if I run this script on `Complete name                            : /mnt/HP_P7_Data/Audio/Standard Recordings/2/2026-06-03 10.23.35.ogg.mp3
Format                                   : MPEG Audio
File size                                : 103 MiB
Duration                                 : 10 h 1 min
Overall bit rate mode                    : Constant
Overall bit rate                         : 24.0 kb/s
Writing library                          : Lavf58.29.100
`, it OOMs after loading all of it to memory. Why so?

---

# claude response

Good catch. The OOM happens here in **PHASE 6**, before chunking even begins:

```python
waveform, sr = torchaudio.load(inference_media)  # Loads ENTIRE file into RAM at once
```

`torchaudio.load()` decodes the **entire 10-hour file to a raw float32 waveform tensor** before any chunking logic. At 32kHz mono float32:

```
10h * 3600s * 32000 samples * 4 bytes = ~4.6 GB
```

So PHASE 7's chunking is chunking an **already-RAM-resident** numpy array — it never reduces peak RAM, it only reduces the RAM held by the *model inference* step.

---

**The fix** is to use `torchaudio.load()` with `frame_offset` and `num_frames` to load one chunk at a time, and move the waveform array out of memory between chunks:

```python
# PHASE 6: Remove the full load entirely
# DELETE: waveform, sr = torchaudio.load(inference_media)
# DELETE: waveform_np = waveform.squeeze(0).numpy()

# PHASE 7: Load chunk-by-chunk instead
chunk_samples = int(chunk_duration * sample_rate)

with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['time'] + list(labels))
    
    # Get total duration via metadata (already have it) to know iteration count
    total_samples = int(duration * sample_rate)
    
    for start in range(0, total_samples, chunk_samples):
        waveform, sr = torchaudio.load(
            inference_media,
            frame_offset=start,
            num_frames=chunk_samples
        )
        if sr != sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
        chunk = waveform.mean(dim=0).numpy()
        del waveform
        gc.collect()
        
        # ... rest of chunk processing unchanged
```

This keeps peak RAM at **one chunk (~180MB)** instead of the full 4.6 GB. The `frame_offset`/`num_frames` parameters tell torchaudio to seek and decode only that slice — it doesn't read the whole file.
