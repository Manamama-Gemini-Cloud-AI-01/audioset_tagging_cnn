After some code rationalization, this now happens. You may check some GEMINI.md type files about the project history. 

Ref: 

~/.../GitHub/audioset_tagging_cnn $ bash audio_me.sh ~/storage/music/It_is_realme.mp3 
--- AudioSet Tagging CNN Inference (SED Mode) ---
Model Type: Cnn14_DecisionLevelMax
Checkpoint path: /storage/emulated/0/LLMs/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth
Input file and parameters: /data/data/com.termux/files/home/storage/music/It_is_realme.mp3
Add  --dynamic_eventogram   or  --static_eventogram  for a video of the graph with a moving marker.

Note: If you see errors with sox / mp3 parsing, then apt install 'sox' and 'libsox-fmt-mp3', test it via 'sox --info /data/data/com.termux/files/home/storage/music/It_is_realme.mp3 ' 
Note: Paths provided as command-line arguments should be absolute to avoid ambiguity.
We are checking the real path of the first argument, which may hang if the file be not accessible...

WARNING: linker: Warning: "/data/data/com.termux/files/usr/lib/python3.13/site-packages/optree.libs/libc++_shared-d523468d.so" unused DT entry: unknown processor-specific (type 0x70000001 arg 0x0) (ignoring)
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
ffmpeg -i /data/data/com.termux/files/home/storage/music/It_is_realme.mp3 -c copy -f segment -segment_time 1200 output_%03d.mp4

Tips & Environment Hacks:
* For 'undefined symbol: torch_library_impl' or 'NotImplementedError':
  Run: pip install -U torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
* If you see 'NotImplementedError: sys.platform = android' after an update:
  Edit: /data/data/com.termux/files/usr/lib/python3.13/site-packages/torchaudio/_internally_replaced_utils.py
  Change 'if sys.platform == "linux":' to 'if sys.platform == "android":'

Dependency Versions:
MoviePy: 2.1.2
Torchaudio: 2.11.0
Torchcodec: 0.11.0a0

Using device: cpu
Using CPU.
Copied AI analysis guide to: /data/data/com.termux/files/home/storage/music/It_is_realme_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/auditory_cognition_guide_template.md
⏲  🗃️  File duration: 0:01:20
📽  🗃️  Video resolution: 900x900
📊  Starting inference in 3.0m chunks. (RAM Avail: 1424 MB)
    Resolution: Disk 100 Hz (Data Frames) | Visualization 5 Hz (UI/RAM)
Chunk at 0m finished. (RAM Avail: 1884 MB)
Aggregation complete. Internal Viz resolution: 5 Hz (Data Frames)
Final analysis duration: 80.60s
Saved eventogram PNG to: /data/data/com.termux/files/home/storage/music/It_is_realme_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/eventogram.png
📊  Generating AI-friendly event summary files…
Saved summary events CSV to: /data/data/com.termux/files/home/storage/music/It_is_realme_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/summary_events.csv
📊  Generating detailed AI-friendly event delta JSON (RLE optimized)…
Saved AI-friendly delta JSON to: /data/data/com.termux/files/home/storage/music/It_is_realme_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/detailed_events_delta_ai_attention_friendly.json
📊  Generating interactive Plotly dashboard…
Saved interactive dashboard to: /data/data/com.termux/files/home/storage/music/It_is_realme_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/interactive_dashboard.html
⏲  🗃️  Analysis finished. Input duration: 80.60s
	Command being timed: "python /data/data/com.termux/files/home/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py --model_type=Cnn14_DecisionLevelMax --checkpoint_path=/storage/emulated/0/LLMs/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth --cuda /data/data/com.termux/files/home/storage/music/It_is_realme.mp3"
	User time (seconds): 75.55
	System time (seconds): 18.44
	Percent of CPU this job got: 162%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:57.83
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 2629804
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 1130
	Minor (reclaiming a frame) page faults: 1610194
	Voluntary context switches: 16525
	Involuntary context switches: 341651
	Swaps: 0
	File system inputs: 2220672
	File system outputs: 20488
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0

📊  Launching Shapash Correlations Dashboard...
Note: The dashboard explains ONE specific sound class (the target).
Each .pkl file is a targeted 'Acoustic Brain', not a universal model for all classes.
loading /storage/emulated/0/Music/It_is_realme_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/full_event_log.csv
Test: explaining the most popular target class: Music (avg prob: 0.6023)
Note: Each .pkl 'brain' is specific to ONE target class. It is NOT universal for all classes.
analyzing 74 features for Music
training model with 20 estimators...
compiling shapash on 204 samples...
INFO: Shap explainer type - <shap.explainers._tree.TreeExplainer object at 0x7666ef2900>
computing global importance...
saving acoustic brain to: /storage/emulated/0/Music/It_is_realme_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/shapash_brain_music.pkl

--- TOP ACOUSTIC WEIGHTS (PREDICTORS FOR MUSIC) ---
Rhythm and blues        0.138259
Jingle bell             0.103979
Jingle, tinkle          0.068529
Independent music       0.054159
Dubstep                 0.038417
Soul music              0.038386
Middle Eastern music    0.033421
Acoustic guitar         0.031185
Country                 0.028919
Christian music         0.028646
dtype: float64
--------------------------------------------------

initializing webapp...
Traceback (most recent call last):
  File "/data/data/com.termux/files/home/Downloads/GitHub/audioset_tagging_cnn/scripts/Shapash_visualization/launch_correlations_dashboard.py", line 139, in <module>
    main()
    ~~~~^^
  File "/data/data/com.termux/files/home/Downloads/GitHub/audioset_tagging_cnn/scripts/Shapash_visualization/launch_correlations_dashboard.py", line 119, in main
    xpl.init_app(title_story=f"Acoustic Explanation: {target}")
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: SmartExplainer.init_app() got an unexpected keyword argument 'title_story'

~/.../GitHub/audioset_tagging_cnn $ 


