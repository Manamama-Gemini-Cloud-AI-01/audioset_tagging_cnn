One audio file parses (the more tricky one, larger) , the other does not. Before you work on code, do read the relevant materials in this repo. 


Ref: 
```

zezen@above-hp2-silver:~/Downloads/GitHub/audioset_tagging_cnn$ bash audio_me.sh "/home/zezen/Music/Music_on_Data/Recorded/20260429T190053.mp3"
--- AudioSet Tagging CNN Inference (SED Mode) ---
Model Type: Cnn14_DecisionLevelMax
Checkpoint path: /media/zezen/HP_P7_Data/Temp/AI_models/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth
Input file and parameters: /home/zezen/Music/Music_on_Data/Recorded/20260429T190053.mp3
Add  --dynamic_eventogram   or  --static_eventogram  for a video of the graph with a moving marker.

Note: If you see errors with sox / mp3 parsing, then apt install 'sox' and 'libsox-fmt-mp3', test it via 'sox --info /home/zezen/Music/Music_on_Data/Recorded/20260429T190053.mp3 ' 
Note: Paths provided as command-line arguments should be absolute to avoid ambiguity.
Note: if no torchaudio backends, do find a patch file, usually called speechbrain.patch, and patch speechbrain itself. 
2026-05-01 09:25:47 above-hp2-silver numexpr.utils[52382] INFO NumExpr defaulting to 8 threads.
Eventogrammer, version 6.7.6. Material Changes:  * 50x RAM Optimization: Internal RAM structures (Eventogram/Spectrogram) are now max-pooled to 2 FPS. * Improved idempotency. 
Notes: The processing time ratio now is: 15 second of the orignal duration takes 1 seconds to process on a regular 300 GFLOPs, 4 core CPU.
If the file is too long, use e.g. this to split:
mkdir split_input_media && cd split_input_media && ffmpeg -i /home/zezen/Music/Music_on_Data/Recorded/20260429T190053.mp3 -c copy -f segment -segment_time 1200 output_%03d.mp4
This script is an adaptation of: https://github.com/qiuqiangkong/audioset_tagging_cnn so see there if something be amiss.
Note on the models:
* `Cnn14_DecisionLevelMax_mAP=0.385.pth` (Sound Event Detection): This model uses Decision-level pooling. It calculates classification probabilities for every small segment of time first, and only then takes the maximum probability to represent the whole clip. Resolution: Cnn14_DecisionLevelMax is specifically designed for Sound Event Detection (SED). Because it maintains the time dimension through the classifier, it can output the framewise_output used by the inference script to generate the 'Eventograms' and the CSV logs.
* The other models are good for audio tagging: use the '--audio_tagging' switch for that mode.

Note on speed: works 1.7 times faster in Prooted Debian than in Termux, see the comments why so.
Note on the out of memory crashes: close all other programs in Droid, especially the browser. Or just restart whole phone. Or do it in Recovery.
Tips: 'undefined symbol: torch_library_impl' or 'NotImplementedError':
This is often a version mismatch between torch and torchaudio, simply run:
pip install -U torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install -U torchcodec --extra-index-url https://download.pytorch.org/whl/cpu
In e.g. Prooted Debian under Termux, torchcodec has dependencies on NVIDA .so files and the script errors, so you may need to git clone and pip install it from scratch (which works without a hitch) then.
If you see 'NotImplementedError: sys.platform = android' after an update:
1. Edit: /data/data/com.termux/files/usr/lib/python3.12/site-packages/torchaudio/_internally_replaced_utils.py
2. Change 'if sys.platform == "linux":' to 'if sys.platform == "android":' - it works.
Using moviepy version: 2.1.2
Using torchaudio version: 2.11.0+cpu
Using torchcodec version: 0.11.1+cpu

Using device: cpu
Using CPU.
Copied AI analysis guide to: /home/zezen/Music/Music_on_Data/Recorded/20260429T190053_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/auditory_cognition_guide_template.md
/usr/lib/python3/dist-packages/paramiko/pkey.py:81: CryptographyDeprecationWarning: TripleDES has been moved to cryptography.hazmat.decrepit.ciphers.algorithms.TripleDES and will be removed from cryptography.hazmat.primitives.ciphers.algorithms in 48.0.0.
  "cipher": algorithms.TripleDES,
/usr/lib/python3/dist-packages/paramiko/transport.py:254: CryptographyDeprecationWarning: TripleDES has been moved to cryptography.hazmat.decrepit.ciphers.algorithms.TripleDES and will be removed from cryptography.hazmat.primitives.ciphers.algorithms in 48.0.0.
  "class": algorithms.TripleDES,
⏲  🗃️  File duration: 2:40:42
Loaded waveform shape: torch.Size([1, 154972224]), sample rate: 16000
Processed waveform shape: (309944448,)
📊  Processing in 3.0m chunks. CSV: 100 FPS | RAM/Vis: 2 FPS
Chunk at 0m done. RAM usage: 361 vis-frames cached.
Chunk at 3m done. RAM usage: 722 vis-frames cached.
Chunk at 6m done. RAM usage: 1083 vis-frames cached.
Chunk at 9m done. RAM usage: 1444 vis-frames cached.
Chunk at 12m done. RAM usage: 1805 vis-frames cached.
Chunk at 15m done. RAM usage: 2166 vis-frames cached.
Chunk at 18m done. RAM usage: 2527 vis-frames cached.
Chunk at 21m done. RAM usage: 2888 vis-frames cached.
Chunk at 24m done. RAM usage: 3249 vis-frames cached.
Chunk at 27m done. RAM usage: 3610 vis-frames cached.
Chunk at 30m done. RAM usage: 3971 vis-frames cached.
Chunk at 33m done. RAM usage: 4332 vis-frames cached.
Chunk at 36m done. RAM usage: 4693 vis-frames cached.
Chunk at 39m done. RAM usage: 5054 vis-frames cached.
Chunk at 42m done. RAM usage: 5415 vis-frames cached.
Chunk at 45m done. RAM usage: 5776 vis-frames cached.
Chunk at 48m done. RAM usage: 6137 vis-frames cached.
Chunk at 51m done. RAM usage: 6498 vis-frames cached.
Chunk at 54m done. RAM usage: 6859 vis-frames cached.
Chunk at 57m done. RAM usage: 7220 vis-frames cached.
Chunk at 60m done. RAM usage: 7581 vis-frames cached.
Chunk at 63m done. RAM usage: 7942 vis-frames cached.
Chunk at 66m done. RAM usage: 8303 vis-frames cached.
Chunk at 69m done. RAM usage: 8664 vis-frames cached.
Chunk at 72m done. RAM usage: 9025 vis-frames cached.
Chunk at 75m done. RAM usage: 9386 vis-frames cached.
Chunk at 78m done. RAM usage: 9747 vis-frames cached.
Chunk at 81m done. RAM usage: 10108 vis-frames cached.
Chunk at 84m done. RAM usage: 10469 vis-frames cached.
Chunk at 87m done. RAM usage: 10830 vis-frames cached.
Chunk at 90m done. RAM usage: 11191 vis-frames cached.
Chunk at 93m done. RAM usage: 11552 vis-frames cached.
Chunk at 96m done. RAM usage: 11913 vis-frames cached.
Chunk at 99m done. RAM usage: 12274 vis-frames cached.
Chunk at 102m done. RAM usage: 12635 vis-frames cached.
Chunk at 105m done. RAM usage: 12996 vis-frames cached.
Chunk at 108m done. RAM usage: 13357 vis-frames cached.
Chunk at 111m done. RAM usage: 13718 vis-frames cached.
Chunk at 114m done. RAM usage: 14079 vis-frames cached.
Chunk at 117m done. RAM usage: 14440 vis-frames cached.
Chunk at 120m done. RAM usage: 14801 vis-frames cached.
Chunk at 123m done. RAM usage: 15162 vis-frames cached.
Chunk at 126m done. RAM usage: 15523 vis-frames cached.
Chunk at 129m done. RAM usage: 15884 vis-frames cached.
Chunk at 132m done. RAM usage: 16245 vis-frames cached.
Chunk at 135m done. RAM usage: 16606 vis-frames cached.
Chunk at 138m done. RAM usage: 16967 vis-frames cached.
Chunk at 141m done. RAM usage: 17328 vis-frames cached.
Chunk at 144m done. RAM usage: 17689 vis-frames cached.
Chunk at 147m done. RAM usage: 18050 vis-frames cached.
Chunk at 150m done. RAM usage: 18411 vis-frames cached.
Chunk at 153m done. RAM usage: 18772 vis-frames cached.
Chunk at 156m done. RAM usage: 19133 vis-frames cached.
Chunk at 159m done. RAM usage: 19425 vis-frames cached.
Aggregation complete. Internal RAM resolution: 2 FPS
Computed left margin: 285px (frac: 0.223)
Saved visualization to: /home/zezen/Music/Music_on_Data/Recorded/20260429T190053_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/eventogram.png
📊  Generating AI-friendly event summary files…
Saved summary events CSV to: /home/zezen/Music/Music_on_Data/Recorded/20260429T190053_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/summary_events.csv
📊  Generating detailed AI-friendly event delta JSON (with RLE compression)…
Saved AI-friendly delta JSON (compressed) to: /home/zezen/Music/Music_on_Data/Recorded/20260429T190053_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/detailed_events_delta_ai_attention_friendly.json
📊  Generating interactive Plotly dashboard…
Saved interactive dashboard to: /home/zezen/Music/Music_on_Data/Recorded/20260429T190053_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/interactive_dashboard.html
⏲  🗃️  Reminder: input file duration: 9642.106856
	Command being timed: "python /home/zezen/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py --model_type=Cnn14_DecisionLevelMax --checkpoint_path=/media/zezen/HP_P7_Data/Temp/AI_models/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth --cuda /home/zezen/Music/Music_on_Data/Recorded/20260429T190053.mp3"
	User time (seconds): 2415.31
	System time (seconds): 238.76
	Percent of CPU this job got: 227%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 19:28.18
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 6380988
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 252137
	Minor (reclaiming a frame) page faults: 60037433
	Voluntary context switches: 419798
	Involuntary context switches: 1009616
	Swaps: 0
	File system inputs: 9188032
	File system outputs: 5816
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0

📊  Inference complete. Launching Shapash Correlations Dashboard...
2026-05-01 09:45:03 above-hp2-silver numexpr.utils[115541] INFO NumExpr defaulting to 8 threads.
loading /media/zezen/HP_P7_Data/Music/Recorded/20260429T190053_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/full_event_log.csv
explaining target: Speech (avg prob: 0.6037)
analyzing 27 features for Speech
training model with 20 estimators...
compiling shapash on 220 samples...
INFO: Shap explainer type - <shap.explainers._tree.TreeExplainer object at 0x77e7dab05850>
computing global importance...

--- TOP ACOUSTIC WEIGHTS (PREDICTORS FOR SPEECH) ---
Conversation                     0.336748
Male singing                     0.235240
Inside, large room or hall       0.104867
Male speech, man speaking        0.093976
Drawer open or close             0.041088
Singing                          0.036294
Tap                              0.030876
Sliding door                     0.016962
Female speech, woman speaking    0.016337
Chuckle, chortle                 0.015770
dtype: float64
--------------------------------------------------

saving acoustic brain to: /media/zezen/HP_P7_Data/Music/Recorded/20260429T190053_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/shapash_brain_speech.pkl
initializing webapp...
--------------------------------------------------
DASHBOARD STARTING (Explaining Speech)...
Open your browser at: http://localhost:8050
--------------------------------------------------

real	3m52.587s
user	3m31.809s
sys	0m3.706s

zezen@above-hp2-silver:~/Downloads/GitHub/audioset_tagging_cnn$ open /home/zezen/Music/Music_on_Data/Recorded/20260429T190053_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/interactive_dashboard.html
zezen@above-hp2-silver:~/Downloads/GitHub/audioset_tagging_cnn$ Opening in existing browser session.
^C
zezen@above-hp2-silver:~/Downloads/GitHub/audioset_tagging_cnn$ bash audio_me.sh /home/zezen/Music/Music_on_Data/Recorded/20260429T225029.mp3
--- AudioSet Tagging CNN Inference (SED Mode) ---
Model Type: Cnn14_DecisionLevelMax
Checkpoint path: /media/zezen/HP_P7_Data/Temp/AI_models/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth
Input file and parameters: /home/zezen/Music/Music_on_Data/Recorded/20260429T225029.mp3
Add  --dynamic_eventogram   or  --static_eventogram  for a video of the graph with a moving marker.

Note: If you see errors with sox / mp3 parsing, then apt install 'sox' and 'libsox-fmt-mp3', test it via 'sox --info /home/zezen/Music/Music_on_Data/Recorded/20260429T225029.mp3 ' 
Note: Paths provided as command-line arguments should be absolute to avoid ambiguity.
Note: if no torchaudio backends, do find a patch file, usually called speechbrain.patch, and patch speechbrain itself. 
2026-05-01 11:46:12 above-hp2-silver numexpr.utils[569478] INFO NumExpr defaulting to 8 threads.
Eventogrammer, version 6.7.6. Material Changes:  * 50x RAM Optimization: Internal RAM structures (Eventogram/Spectrogram) are now max-pooled to 2 FPS. * Improved idempotency. 
Notes: The processing time ratio now is: 15 second of the orignal duration takes 1 seconds to process on a regular 300 GFLOPs, 4 core CPU.
If the file is too long, use e.g. this to split:
mkdir split_input_media && cd split_input_media && ffmpeg -i /home/zezen/Music/Music_on_Data/Recorded/20260429T225029.mp3 -c copy -f segment -segment_time 1200 output_%03d.mp4
This script is an adaptation of: https://github.com/qiuqiangkong/audioset_tagging_cnn so see there if something be amiss.
Note on the models:
* `Cnn14_DecisionLevelMax_mAP=0.385.pth` (Sound Event Detection): This model uses Decision-level pooling. It calculates classification probabilities for every small segment of time first, and only then takes the maximum probability to represent the whole clip. Resolution: Cnn14_DecisionLevelMax is specifically designed for Sound Event Detection (SED). Because it maintains the time dimension through the classifier, it can output the framewise_output used by the inference script to generate the 'Eventograms' and the CSV logs.
* The other models are good for audio tagging: use the '--audio_tagging' switch for that mode.

Note on speed: works 1.7 times faster in Prooted Debian than in Termux, see the comments why so.
Note on the out of memory crashes: close all other programs in Droid, especially the browser. Or just restart whole phone. Or do it in Recovery.
Tips: 'undefined symbol: torch_library_impl' or 'NotImplementedError':
This is often a version mismatch between torch and torchaudio, simply run:
pip install -U torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install -U torchcodec --extra-index-url https://download.pytorch.org/whl/cpu
In e.g. Prooted Debian under Termux, torchcodec has dependencies on NVIDA .so files and the script errors, so you may need to git clone and pip install it from scratch (which works without a hitch) then.
If you see 'NotImplementedError: sys.platform = android' after an update:
1. Edit: /data/data/com.termux/files/usr/lib/python3.12/site-packages/torchaudio/_internally_replaced_utils.py
2. Change 'if sys.platform == "linux":' to 'if sys.platform == "android":' - it works.
Using moviepy version: 2.1.2
Using torchaudio version: 2.11.0+cpu
Using torchcodec version: 0.11.1+cpu

Using device: cpu
Using CPU.
Copied AI analysis guide to: /home/zezen/Music/Music_on_Data/Recorded/20260429T225029_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/auditory_cognition_guide_template.md
/usr/lib/python3/dist-packages/paramiko/pkey.py:81: CryptographyDeprecationWarning: TripleDES has been moved to cryptography.hazmat.decrepit.ciphers.algorithms.TripleDES and will be removed from cryptography.hazmat.primitives.ciphers.algorithms in 48.0.0.
  "cipher": algorithms.TripleDES,
/usr/lib/python3/dist-packages/paramiko/transport.py:254: CryptographyDeprecationWarning: TripleDES has been moved to cryptography.hazmat.decrepit.ciphers.algorithms.TripleDES and will be removed from cryptography.hazmat.primitives.ciphers.algorithms in 48.0.0.
  "class": algorithms.TripleDES,
⏲  🗃️  File duration: 0:29:23
Loaded waveform shape: torch.Size([1, 30089664]), sample rate: 16000
Processed waveform shape: (60179328,)
📊  Processing in 3.0m chunks. CSV: 100 FPS | RAM/Vis: 2 FPS
Chunk at 0m done. RAM usage: 361 vis-frames cached.
Chunk at 3m done. RAM usage: 722 vis-frames cached.
Chunk at 6m done. RAM usage: 1083 vis-frames cached.
Chunk at 9m done. RAM usage: 1444 vis-frames cached.
Chunk at 12m done. RAM usage: 1805 vis-frames cached.
Chunk at 15m done. RAM usage: 2166 vis-frames cached.
Chunk at 18m done. RAM usage: 2527 vis-frames cached.
Chunk at 21m done. RAM usage: 2888 vis-frames cached.
Chunk at 24m done. RAM usage: 3249 vis-frames cached.
Chunk at 27m done. RAM usage: 3610 vis-frames cached.
Chunk at 30m done. RAM usage: 3772 vis-frames cached.
Aggregation complete. Internal RAM resolution: 2 FPS
Computed left margin: 285px (frac: 0.223)
Traceback (most recent call last):
  File "/home/zezen/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py", line 1228, in <module>
    sound_event_detection(args)
  File "/home/zezen/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py", line 639, in sound_event_detection
    axs[1].xaxis.set_ticklabels(x_labels[:len(x_ticks)], rotation=45, ha='right', fontsize=10)
  File "/home/zezen/.local/lib/python3.12/site-packages/matplotlib/axis.py", line 2106, in set_ticklabels
    raise ValueError(
ValueError: The number of FixedLocator locations (22), usually from a call to set_ticks, does not match the number of labels (21).
Command exited with non-zero status 1
	Command being timed: "python /home/zezen/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py --model_type=Cnn14_DecisionLevelMax --checkpoint_path=/media/zezen/HP_P7_Data/Temp/AI_models/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth --cuda /home/zezen/Music/Music_on_Data/Recorded/20260429T225029.mp3"
	User time (seconds): 470.54
	System time (seconds): 52.70
	Percent of CPU this job got: 225%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 3:52.09
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 2876400
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 11622
	Minor (reclaiming a frame) page faults: 13345679
	Voluntary context switches: 46204
	Involuntary context switches: 171207
	Swaps: 0
	File system inputs: 1566472
	File system outputs: 176
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 1

📊  Inference complete. Launching Shapash Correlations Dashboard...
2026-05-01 11:49:47 above-hp2-silver numexpr.utils[581060] INFO NumExpr defaulting to 8 threads.
loading /media/zezen/HP_P7_Data/Music/Recorded/20260429T225029_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/full_event_log.csv
explaining target: Speech (avg prob: 0.4217)
analyzing 55 features for Speech
training model with 20 estimators...
compiling shapash on 220 samples...
INFO: Shap explainer type - <shap.explainers._tree.TreeExplainer object at 0x787116753d70>
computing global importance...

--- TOP ACOUSTIC WEIGHTS (PREDICTORS FOR SPEECH) ---
Male speech, man speaking        0.457966
Conversation                     0.184285
Chuckle, chortle                 0.042773
Field recording                  0.033480
Music                            0.026376
Female speech, woman speaking    0.024024
Fire                             0.020269
Thunder                          0.017931
Outside, urban or manmade        0.016970
Inside, public space             0.013928
dtype: float64
--------------------------------------------------

saving acoustic brain to: /media/zezen/HP_P7_Data/Music/Recorded/20260429T225029_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/shapash_brain_speech.pkl
initializing webapp...
--------------------------------------------------
DASHBOARD STARTING (Explaining Speech)...
Open your browser at: http://localhost:8050
--------------------------------------------------

real	1m10.503s
user	0m59.739s
sys	0m2.127s

zezen@above-hp2-silver:~/Downloads/GitHub/audioset_tagging_cnn$
```

