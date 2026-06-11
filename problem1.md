It works well, see attached image and below.  

I need: 

* move the legend (labels: 'Music, Speech, Animal ...') to bottom of the graph. Reason: when watched on mobile, the legend takes half the width. 

* Add the resulting .png , here:  below the graph. Reason: I can then compare the png with the graph. 


Ref: 
```
zezen@above-hp2-silver:~/Downloads/GitHub/audioset_tagging_cnn$ git remote -v
origin	https://github.com/Manamama-Gemini-Cloud-AI-01/audioset_tagging_cnn.git (fetch)
origin	https://github.com/Manamama-Gemini-Cloud-AI-01/audioset_tagging_cnn.git (push)
upstream	https://github.com/qiuqiangkong/audioset_tagging_cnn.git (fetch)
upstream	https://github.com/qiuqiangkong/audioset_tagging_cnn.git (push)
zezen@above-hp2-silver:~/Downloads/GitHub/audioset_tagging_cnn$ git branch -v
  master          7a24315 Update performance baseline in GEMINI.md
* test-torchcodec 944fe04 Merge branch 'test-torchcodec' of https://github.com/Manamama-Gemini-Cloud-AI-01/audioset_tagging_cnn into test-torchcodec
zezen@above-hp2-silver:~/Downloads/GitHub/audioset_tagging_cnn$ bash audio_me.sh "/home/zezen/Videos/Videos_on_P7/Watch_me/2/WALL-E_CLIP_COMPILAT.mp4" --no-shapash
--- AudioSet Tagging CNN Inference (SED Mode) ---
Model Type: Cnn14_DecisionLevelMax
Checkpoint path: /media/zezen/HP_P7_Data/Temp/AI_models/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth
Input file and parameters: /home/zezen/Videos/Videos_on_P7/Watch_me/2/WALL-E_CLIP_COMPILAT.mp4 --no-shapash
Add  --dynamic_eventogram   or  --static_eventogram  for a video of the graph with a moving marker.

Note: If you see errors with sox / mp3 parsing, then apt install 'sox' and 'libsox-fmt-mp3', test it via 'sox --info /home/zezen/Videos/Videos_on_P7/Watch_me/2/WALL-E_CLIP_COMPILAT.mp4 --no-shapash ' 
Note: Paths provided as command-line arguments should be absolute to avoid ambiguity.
We are checking the real path of the first argument, which may hang if the file be not accessible...

2026-06-11 18:51:49 above-hp2-silver numexpr.utils[266714] INFO NumExpr defaulting to 8 threads.
Eventogrammer, version 6.12.3
Adaptation of: https://github.com/qiuqiangkong/audioset_tagging_cnn
Recent Material Changes:
* Media player added to plotly graph. 
* H5py used instead of CSV to save disk space.
* We convert all files to MP3
* Load: Memory-safe chunked decoding (OOM Fix for 10h+ files).
* Constants Promoted: vis_fps, output_fps, and adaptive_lookahead are now CLI arguments.

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
ffmpeg -i /home/zezen/Videos/Videos_on_P7/Watch_me/2/WALL-E_CLIP_COMPILAT.mp4 -c copy -f segment -segment_time 1200 output_%03d.mp4
time bash -c 'for file in *; do bash audio_me.sh "$file" --dynamic_eventogram; done'

Tips & Environment Hacks:
* For 'undefined symbol' (e.g. torch_from_blob), 'Invalid argument' crashes, or SIGABRT:
  Root Cause: ABI Collision. You likely have a mix of 'apt' (Debian) and 'pip' (Official) Torch packages.
  The C++ backend becomes a 'hollow shell'—importing the Python part works, but touching the C++ core kills the process.
  Fix: Uninstall ALL apt versions and 'pip install -U' the whole stack together to ensure binary alignment:
  Run: pip install -U torch torchaudio torchcodec --extra-index-url https://download.pytorch.org/whl/cpu
* If Torchcodec errors (e.g., 'libnvrtc.so.13 not found'):
  Root Cause: CUDA Ghosting. Standard wheels link to NVIDIA libs even on CPU-only setups.
  Fix: Use the --extra-index-url above to force CPU-only binaries, or: pip uninstall torchcodec
* If you see 'NotImplementedError: sys.platform = android' after an update:
  Edit: /data/data/com.termux/files/usr/lib/python3.12/site-packages/torchaudio/_internally_replaced_utils.py
  Change 'if sys.platform == "linux":' to 'if sys.platform == "android":'
* If some coverage numba error: do 'apt remove python3-coverage'. Be careful with the below python modules if they have parallel apt based install versions, use one or the other then: 'python -m pip install torch torchaudio torchcodec --upgrade --extra-index-url https://download.pytorch.org/whl/cpu ' : do check these against their apt versions
* If: 'LibsndfileError: File contains data in an unimplemented format', then 'git clone https://github.com/libsndfile/libsndfile.git' and install it in e.g. Termux. Or run in Proot.

Dependency Versions:
Torch: 2.12.0+cpu
Torchaudio: 2.11.0+cpu
Torchcodec: 0.14.0+cpu

Using device: cpu
Using CPU.
Copied AI analysis guide to: /home/zezen/Videos/Videos_on_P7/Watch_me/2/WALL-E_CLIP_COMPILAT_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/auditory_cognition_guide_template.md
🎬  Re-encoding input to CBR MP3 for seeking stability...
⏲  🗃️  File duration: 0:11:16
📊  Starting decoupled inference in 3.0m chunks. (RAM Avail: 1872 MB)
    Native SR: 44100 Hz | Inference SR: 32000 Hz
💡 Aesthetic Decoupling: Aggregating 2x into 2500 RAM columns.
DEBUG: Native Chunk Samples=7938000, Native SR=44100, Chunk Seconds=180.0
DEBUG: Range [0.0, 180.0], Raw Samples Shape: torch.Size([1, 5759198])
DEBUG: Processed Chunk Shape: torch.Size([1, 5759198])
Chunk at 0m finished. (RAM Avail: 1884 MB)
DEBUG: Range [180.0, 360.0], Raw Samples Shape: torch.Size([1, 5760000])
DEBUG: Processed Chunk Shape: torch.Size([1, 5760000])
Chunk at 3m finished. (RAM Avail: 1928 MB)
DEBUG: Range [360.0, 540.0], Raw Samples Shape: torch.Size([1, 5760000])
DEBUG: Processed Chunk Shape: torch.Size([1, 5760000])
Chunk at 6m finished. (RAM Avail: 1824 MB)
DEBUG: Range [540.0, 720.0], Raw Samples Shape: torch.Size([1, 4354370])
DEBUG: Processed Chunk Shape: torch.Size([1, 4354370])
Chunk at 9m finished. (RAM Avail: 1956 MB)
Aggregation complete. Internal Viz resolution: 2.5000 Hz (Data Frames)
Final analysis duration: 676.40s
Saved eventogram PNG to: /home/zezen/Videos/Videos_on_P7/Watch_me/2/WALL-E_CLIP_COMPILAT_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/eventogram.png
📊  Generating AI-friendly event summary files…
Saved summary events CSV to: /home/zezen/Videos/Videos_on_P7/Watch_me/2/WALL-E_CLIP_COMPILAT_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/summary_events.csv
📊  Generating detailed AI-friendly event delta JSON (RLE optimized)…
Saved AI-friendly delta JSON to: /home/zezen/Videos/Videos_on_P7/Watch_me/2/WALL-E_CLIP_COMPILAT_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/detailed_events_delta_ai_attention_friendly.json
📊  Generating interactive Plotly dashboard…
Saved interactive dashboard to: /home/zezen/Videos/Videos_on_P7/Watch_me/2/WALL-E_CLIP_COMPILAT_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/interactive_dashboard.html
⏲  🗃️  Analysis finished. Input duration: 676.40s

	Command being timed: "python /home/zezen/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py --model_type=Cnn14_DecisionLevelMax --checkpoint_path=/media/zezen/HP_P7_Data/Temp/AI_models/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth --cuda /home/zezen/Videos/Videos_on_P7/Watch_me/2/WALL-E_CLIP_COMPILAT.mp4 --no-shapash"
	User time (seconds): 144.90
	System time (seconds): 15.73
	Percent of CPU this job got: 232%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 1:08.99
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 2361252
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 2974
	Minor (reclaiming a frame) page faults: 4920866
	Voluntary context switches: 125699
	Involuntary context switches: 2875
	Swaps: 0
	File system inputs: 1208824
	File system outputs: 21720
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0

⏭️  Skipping Shapash dashboard launch (--no-shapash specified).

zezen@above-hp2-silver:~/Downloads/GitHub/audioset_tagging_cnn$ open /home/zezen/Videos/Videos_on_P7/Watch_me/2/WALL-E_CLIP_COMPILAT_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/interactive_dashboard.html
zezen@above-hp2-silver:~/Downloads/GitHub/audioset_tagging_cnn$ Opening in existing browser session.
```

and all works as of now. 
