`shapash` starts server at foreground and needs stopping it as per below. Which is good. The current logic of audio_me.sh however, does not use it. 


Discuss how it can be used. Ref: 

zezen@above-hp2-silver:~/Downloads/GitHub/audioset_tagging_cnn$ python test_server_1.py 
--- Launching Server 1 on Port 8050 ---
2026-05-15 13:05:58 above-hp2-silver numexpr.utils[73834] INFO NumExpr defaulting to 8 threads.
Dash is running on http://0.0.0.0:8050/

2026-05-15 13:06:01 above-hp2-silver dash.dash[73834] INFO Dash is running on http://0.0.0.0:8050/

 * Serving Flask app 'shapash.webapp.smart_app'
 * Debug mode: off
2026-05-15 13:06:01 above-hp2-silver werkzeug[73834] INFO WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8050
 * Running on http://10.118.7.36:8050
2026-05-15 13:06:01 above-hp2-silver werkzeug[73834] INFO Press CTRL+C to quit
2026-05-15 13:06:09 above-hp2-silver werkzeug[73834] INFO 127.0.0.1 - - [15/May/2026 13:06:09] "GET / HTTP/1.1" 200 -
2026-05-15 13:06:09 above-hp2-silver werkzeug[73834] INFO 127.0.0.1 - - [15/May/2026 13:06:09] "GET /assets/material-icons.css?m=1772266833.9364164 HTTP/1.1" 200 -
2026-05-15 13:06:09 above-hp2-silver werkzeug[73834] INFO 127.0.0.1 - - [15/May/2026 13:06:09] "GET /_dash-component-suites/dash/deps/react@18.v4_0_0m1772266904.3.1.min.js HTTP/1.1" 200 -
2026-05-15 13:06:09 above-hp2-silver werkzeug[73834] INFO 127.0.0.1 - - [15/May/2026 13:06:09] "GET /_dash-component-suites/dash/deps/react-dom@18.v4_0_0m1772266904.3.1.min.js HTTP/1.1" 200 -
2026-05-15 13:06:09 above-hp2-silver werkzeug[73834] INFO 127.0.0.1 - - [15/May/2026 13:06:09] "GET /_dash-component-suites/dash/deps/polyfill@7.v4_0_0m1772266904.12.1.min.js HTTP/1.1" 200 -
2026-05-15 13:06:09 above-hp2-silver werkzeug[73834] INFO 127.0.0.1 - - [15/May/2026 13:06:09] "GET /assets/style.css?m=1772266833.9374154 HTTP/1.1" 200 -
2026-05-15 13:06:09 above-hp2-silver werkzeug[73834] INFO 127.0.0.1 - - [15/May/2026 13:06:09] "GET /_dash-component-suites/dash/deps/prop-types@15.v4_0_0m1772266904.8.1.min.js HTTP/1.1" 200 -
2026-05-15 13:06:09 above-hp2-silver werkzeug[73834] INFO 127.0.0.1 - - [15/May/2026 13:06:09] "GET /_dash-component-suites/dash_bootstrap_components/_components/dash_bootstrap_components.v1_7_1m1772266833.min.js HTTP/1.1" 200 -
2026-05-15 13:06:09 above-hp2-silver werkzeug[73834] INFO 127.0.0.1 - - [15/May/2026 13:06:09] "GET /assets/main.js?m=1772266833.9364164 HTTP/1.1" 200 -
2026-05-15 13:06:09 above-hp2-silver werkzeug[73834] INFO 127.0.0.1 - - [15/May/2026 13:06:09] "GET /assets/jquery.js?m=1772266833.9364164 HTTP/1.1" 200 -
2026-05-15 13:06:09 above-hp2-silver werkzeug[73834] INFO 127.0.0.1 - - [15/May/2026 13:06:09] "GET /_dash-component-suites/dash_daq/dash_daq.v0_6_0m1772266833.min.js HTTP/1.1" 200 -
2026-05-15 13:06:09 above-hp2-silver werkzeug[73834] INFO 127.0.0.1 - - [15/May/2026 13:06:09] "GET /_dash-component-suites/dash/dcc/dash_core_components.v4_0_0m1772266904.js HTTP/1.1" 200 -
2026-05-15 13:06:09 above-hp2-silver werkzeug[73834] INFO 127.0.0.1 - - [15/May/2026 13:06:09] "GET /_dash-component-suites/dash/dash-renderer/build/dash_renderer.v4_0_0m1772266904.min.js HTTP/1.1" 200 -
2026-05-15 13:06:09 above-hp2-silver werkzeug[73834] INFO 127.0.0.1 - - [15/May/2026 13:06:09] "GET /_dash-component-suites/dash/dcc/dash_core_components-shared.v4_0_0m1772266904.js HTTP/1.1" 200 -
2026-05-15 13:06:09 above-hp2-silver werkzeug[73834] INFO 127.0.0.1 - - [15/May/2026 13:06:09] "GET /_dash-component-suites/dash/html/dash_html_components.v4_0_0m1772266904.min.js HTTP/1.1" 200 -
2026-05-15 13:06:09 above-hp2-silver werkzeug[73834] INFO 127.0.0.1 - - [15/May/2026 13:06:09] "GET /_dash-component-suites/dash/dash_table/bundle.v7_0_0m1772266904.js HTTP/1.1" 200 -
2026-05-15 13:06:10 above-hp2-silver werkzeug[73834] INFO 127.0.0.1 - - [15/May/2026 13:06:10] "GET /_dash-dependencies HTTP/1.1" 200 -


vs: 

zezen@above-hp2-silver:~/Downloads/GitHub/audioset_tagging_cnn$  bash audio_me.sh /home/zezen/Videos/Videos_on_P7/Something_very_weird.webm
--- AudioSet Tagging CNN Inference (SED Mode) ---
Model Type: Cnn14_DecisionLevelMax
Checkpoint path: /media/zezen/HP_P7_Data/Temp/AI_models/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth
Input file and parameters: /home/zezen/Videos/Videos_on_P7/Something_very_weird.webm
Add  --dynamic_eventogram   or  --static_eventogram  for a video of the graph with a moving marker.

Note: If you see errors with sox / mp3 parsing, then apt install 'sox' and 'libsox-fmt-mp3', test it via 'sox --info /home/zezen/Videos/Videos_on_P7/Something_very_weird.webm ' 
Note: Paths provided as command-line arguments should be absolute to avoid ambiguity.
We are checking the real path of the first argument, which may hang if the file be not accessible...

2026-05-15 12:27:22 above-hp2-silver numexpr.utils[12357] INFO NumExpr defaulting to 8 threads.
Eventogrammer, version 6.8.9
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
ffmpeg -i /home/zezen/Videos/Videos_on_P7/Something_very_weird.webm -c copy -f segment -segment_time 1200 output_%03d.mp4

Tips & Environment Hacks:
* For 'undefined symbol: torch_library_impl' or 'NotImplementedError':
  Run: pip install -U torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
* If you see 'NotImplementedError: sys.platform = android' after an update:
  Edit: /data/data/com.termux/files/usr/lib/python3.12/site-packages/torchaudio/_internally_replaced_utils.py
  Change 'if sys.platform == "linux":' to 'if sys.platform == "android":'

Dependency Versions:
MoviePy: 2.1.2
Torchaudio: 2.11.0+cpu
Torchcodec: 0.11.1+cpu

Using device: cpu
Using CPU.
Copied AI analysis guide to: /home/zezen/Videos/Videos_on_P7/Something_very_weird_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/auditory_cognition_guide_template.md
⏲  🗃️  File duration: 0:19:01
🮲  🗃️  Video FPS (avg): 23.976
📽  🗃️  Video resolution: 1280x720
📊  Starting inference in 3.0m chunks. (RAM Avail: 3817 MB)
    Resolution: Disk 100 Hz (Data Frames) | Visualization 5 Hz (UI/RAM)
Chunk at 0m finished. (RAM Avail: 3747 MB)
Chunk at 3m finished. (RAM Avail: 3773 MB)
Chunk at 6m finished. (RAM Avail: 3767 MB)
Chunk at 9m finished. (RAM Avail: 3752 MB)
Chunk at 12m finished. (RAM Avail: 3702 MB)
Chunk at 15m finished. (RAM Avail: 3555 MB)
Chunk at 18m finished. (RAM Avail: 3385 MB)
Aggregation complete. Internal Viz resolution: 5 Hz (Data Frames)
Final analysis duration: 1143.00s
Saved eventogram PNG to: /home/zezen/Videos/Videos_on_P7/Something_very_weird_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/eventogram.png
📊  Generating AI-friendly event summary files…
Saved summary events CSV to: /home/zezen/Videos/Videos_on_P7/Something_very_weird_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/summary_events.csv
📊  Generating detailed AI-friendly event delta JSON (RLE optimized)…
Saved AI-friendly delta JSON to: /home/zezen/Videos/Videos_on_P7/Something_very_weird_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/detailed_events_delta_ai_attention_friendly.json
📊  Generating interactive Plotly dashboard…
Saved interactive dashboard to: /home/zezen/Videos/Videos_on_P7/Something_very_weird_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/interactive_dashboard.html
⏲  🗃️  Analysis finished. Input duration: 1143.00s
	Command being timed: "python /home/zezen/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py --model_type=Cnn14_DecisionLevelMax --checkpoint_path=/media/zezen/HP_P7_Data/Temp/AI_models/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth --cuda /home/zezen/Videos/Videos_on_P7/Something_very_weird.webm"
	User time (seconds): 160.96
	System time (seconds): 14.09
	Percent of CPU this job got: 280%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 1:02.30
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 3813172
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 728
	Minor (reclaiming a frame) page faults: 6811655
	Voluntary context switches: 34487
	Involuntary context switches: 3557
	Swaps: 0
	File system inputs: 188072
	File system outputs: 88
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0

📊  Inference complete. Launching Shapash Correlations Dashboard...
2026-05-15 12:28:22 above-hp2-silver numexpr.utils[13728] INFO NumExpr defaulting to 8 threads.
loading /media/zezen/HP_P7_Data/Videos/Something_very_weird_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/full_event_log.csv
explaining target: Speech (avg prob: 0.4525)
analyzing 29 features for Speech
training model with 20 estimators...
compiling shapash on 219 samples...
INFO: Shap explainer type - <shap.explainers._tree.TreeExplainer object at 0x75a345df0fb0>
computing global importance...

--- TOP ACOUSTIC WEIGHTS (PREDICTORS FOR SPEECH) ---
Male speech, man speaking     0.280397
Tambourine                    0.138650
Conversation                  0.123844
Inside, large room or hall    0.078379
Electronic music              0.035562
Inside, small room            0.031252
Drip                          0.031149
Speech synthesizer            0.028353
Snare drum                    0.027822
Animal                        0.022880
dtype: float64
--------------------------------------------------

saving acoustic brain to: /media/zezen/HP_P7_Data/Videos/Something_very_weird_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/shapash_brain_speech.pkl
initializing webapp...
--------------------------------------------------
DASHBOARD STARTING (Explaining Speech)...
Open your browser at: http://localhost:8050
--------------------------------------------------

real	0m16.244s
user	0m17.179s
sys	0m1.246s


(which ends asap)


