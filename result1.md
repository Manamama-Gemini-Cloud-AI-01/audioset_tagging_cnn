End of processing 🐰 Aliases 2


ro.build.version.security_patch: 2020-02-05
getprop ro.build.display.id: RMX3085_11_A.26
u0_a278@localhost
-----------------
OS: Android REL 11 aarch64
Host: realme 8 (RMX3085)
Kernel: Linux 4.14.186+
Uptime: 13 mins
Packages: 142 (pacman), 906 (dpkg)
Shell: bash 5.3.9
DE: RMX3085_11_A.26
WM: WindowManager (SurfaceFlinger)
Terminal: Termux 0.118.3
Terminal Font: UbuntuMono Nerd Font Mono
CPU: 2 x MT6785V/CD (8) @ 2.05 GHz
GPU: ARM Mali-G76 MC4 r0p0 (4) @ 0.01 GHz [Integr]
Memory: 2.09 GiB / 5.51 GiB (38%)
Swap: 813.42 MiB / 6.00 GiB (13%)
Disk (/): 1.97 GiB / 1.97 GiB (100%) - ext4 [Read]
Disk (/storage/emulated): 96.96 GiB / 108.01 GiB e
Local IP (bt-pan): 192.168.44.1/24
Local IP (ccmni0): 100.117.248.151/8
Battery: 58% [AC Connected, Charging]
Locale: en_US.UTF-8




  We are : 📲 Realme A (main one)                 

End of processing 🗺️ : Termux .bashrc

To paste the content into Termux terminal: Ctrl+Alt+V, to retrieve the previous command in history: Ctrl+P, next command: Ctrl+N.



cd ~/Downloads/GitHub/audioset_tagging_cnn/ && bash audio_me.sh /storage/emulated/0/Movies/Jolly_Mom_Has_No_Ide.2.mp4
--- AudioSet Tagging CNN Inference (SED Mode) ---
Model Type: Cnn14_DecisionLevelMax
Checkpoint path: /storage/emulated/0/LLMs/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth
Input file and parameters: /storage/emulated/0/Movies/Jolly_Mom_Has_No_Ide.2.mp4
Add  --dynamic_eventogram   or  --static_eventogram  for a video of the graph with a moving marker.

Note: If you see errors with sox / mp3 parsing, then apt install 'sox' and 'libsox-fmt-mp3', test it via 'sox --info /storage/emulated/0/Movies/Jolly_Mom_Has_No_Ide.2.mp4 '
Note: Paths provided as command-line arguments should be absolute to avoid ambiguity.
Note: if no torchaudio backends, do find a patch file, usually called speechbrain.patch, and patch speechbrain itself.
Eventogrammer, version 6.6.5. Material Changes:
 * Removed torchcodec: as it crashes on Android due to broken CUDA library linkage. On the other hand, torchaudio nowadays imports torchcodec always, so a note about it is left in the tips.
 * Broken the 'Aggregation Bottleneck': High-res data (100 FPS) is now streamed directly to disk (CSV) during inference.
 * 50x RAM Optimization: Internal RAM structures (Eventogram/Spectrogram) are now max-pooled to 2 FPS.
 * Chunked Post-processing: Spectrogram is calculated in 3m segments to prevent OOM spikes on long files. CSV reflects samples at lower frequence: size saving.
Notes: The processing time ratio now is: 15 second of the orignal duration takes 1 seconds to process on a regular 300 GFLOPs, 4 core CPU.
If the file is too long, use e.g. this to split:
mkdir split_input_media && cd split_input_media && ffmpeg -i /storage/emulated/0/Movies/Jolly_Mom_Has_No_Ide.2.mp4 -c copy -f segment -segment_time 1200 output_%03d.mp4
This script is an adaptation of: https://github.com/qiuqiangkong/audioset_tagging_cnn so see there if something be amiss.
Note on the models:
* `Cnn14_DecisionLevelMax_mAP=0.385.pth` (Sound Event Detection): This model uses Decision-level pooling. It calculates classification probabilities for every small segment of time first, and only then takes the maximum probability to represent the whole clip. Resolution: Cnn14_DecisionLevelMax is specifically designed for Sound Event Detection (SED). Because it maintains the time dimension through the classifier, it can output the framewise_output used by the inference script to generate the 'Eventograms' and the CSV logs.
* The other models are good for audio tagging: use the '--audio_tagging' switch for that mode.

Using moviepy version: 2.1.2
Using torchaudio version: 2.11.0
Tips: 'undefined symbol: torch_library_impl' or 'NotImplementedError':
This is often a version mismatch between torch and torchaudio, simply run:
pip install -U torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install -U torchcodec --extra-index-url https://download.pytorch.org/whl/cpu
On the other hand, in e.g. Prooted Debian under Termux, torchcodec has dependencies on NVIDA .so files and errors here, so now the script does not implement 'import torchcodec' at all, just in case.
If you see 'NotImplementedError: sys.platform = android' after an update:
1. Edit: /data/data/com.termux/files/usr/lib/python3.13/site-packages/torchaudio/_internally_replaced_utils.py
2. Change 'if sys.platform == "linux":' to 'if sys.platform == "android":' - it works.

Using device: cpu
Using CPU.
Copied AI analysis guide to: /storage/emulated/0/Movies/Jolly_Mom_Has_No_Ide.2_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/auditory_cognition_guide_template.md
⏲  🗃️  File duration: 0:48:45
🮲  🗃️  Video FPS (avg): 30.000
📽  🗃️  Video resolution: 1280x720
Loaded waveform shape: torch.Size([2, 128997376]), sample rate: 44100
Processed waveform shape: (93603536,)
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
Chunk at 48m done. RAM usage: 5867 vis-frames cached.
Aggregation complete. Internal RAM resolution: 2 FPS
Computed left margin: 337px (frac: 0.263)
Saved visualization to: /storage/emulated/0/Movies/Jolly_Mom_Has_No_Ide.2_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/eventogram.png
📊  Generating AI-friendly event summary files…
Saved summary events CSV to: /storage/emulated/0/Movies/Jolly_Mom_Has_No_Ide.2_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/summary_events.csv
📊  Generating detailed AI-friendly event delta JSON (with RLE compression)…
Saved AI-friendly delta JSON (compressed) to: /storage/emulated/0/Movies/Jolly_Mom_Has_No_Ide.2_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/detailed_events_delta_ai_attention_friendly.json
📊  Generating interactive Plotly dashboard…
Saved interactive dashboard to: /storage/emulated/0/Movies/Jolly_Mom_Has_No_Ide.2_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/interactive_dashboard.html
⏲  🗃️  Reminder: input file duration: 2925.110567
        Command being timed: "python /data/data/com.termux/files/home/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py --model_type=Cnn14_DecisionLevelMax --checkpoint_path=/storage/emulated/0/LLMs/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth --cuda /storage/emulated/0/Movies/Jolly_Mom_Has_No_Ide.2.mp4"
        User time (seconds): 1389.33
        System time (seconds): 273.70
        Percent of CPU this job got: 195%
        Elapsed (wall clock) time (h:mm:ss or m:ss): 14:09.15
        Average shared text size (kbytes): 0
        Average unshared data size (kbytes): 0
        Average stack size (kbytes): 0
        Average total size (kbytes): 0
        Maximum resident set size (kbytes): 4326836
        Average resident set size (kbytes): 0
        Major (requiring I/O) page faults: 10529
        Minor (reclaiming a frame) page faults: 41446869
        Voluntary context switches: 278007
        Involuntary context switches: 2074906
        Swaps: 0
        File system inputs: 5103704
        File system outputs: 360040
        Socket messages sent: 0
        Socket messages received: 0
        Signals delivered: 0
        Page size (bytes): 4096
        Exit status: 0

📊  Inference complete. Launching Shapash Correlations Dashboard...
/data/data/com.termux/files/usr/lib/python3.13/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (7.0.1)/charset_normalizer (3.4.4) doesn't match a supported version!
  warnings.warn(
loading /storage/emulated/0/Movies/Jolly_Mom_Has_No_Ide.2_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/full_event_log.csv
explaining target: Speech (avg prob: 0.5814)
analyzing 13 features for Speech
training model with 20 estimators...
compiling shapash on 219 samples...
INFO: Shap explainer type - <shap.explainers._tree.TreeExplainer object at 0x76c92a1e80>
computing global importance...

--- TOP ACOUSTIC WEIGHTS (PREDICTORS FOR SPEECH) ---
Male speech, man speaking     0.401818
Inside, small room            0.149814
Inside, large room or hall    0.126902
Music                         0.065477
Conversation                  0.059837
Radio                         0.040379
Outside, rural or natural     0.029828
Animal                        0.029318
Outside, urban or manmade     0.024192
Narration, monologue          0.023201
dtype: float64
--------------------------------------------------

saving acoustic brain to: /storage/emulated/0/Movies/Jolly_Mom_Has_No_Ide.2_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/shapash_brain_speech.pkl
initializing webapp...
--------------------------------------------------
DASHBOARD STARTING (Explaining Speech)...
Open your browser at: http://localhost:8050
--------------------------------------------------

real    1m33.659s
user    1m27.153s
sys     0m4.291s

[1]+  Done                       updatedb  (wd: ~)
(wd now: ~/Downloads/GitHub/audioset_tagging_cnn)
~/.../GitHub/audioset_tagging_cnn $
