deb
root@localhost
--------------
OS: Debian GNU/Linux 13 (trixie) aarch64
Host: realme RMX3085
Kernel: 6.17.0-PRoot-Distro
Uptime: 2 mins
Packages: 2245 (dpkg), 1 (pkg)
Shell: bash 5.2.37
Terminal: proot
CPU: MT6785V/CD (8) @ 2.000GHz
Memory: 2480MiB / 5638MiB





You can use the internal sd card via :
ls /storage/self/primary/

Processing: 🐰 Aliases 2
version 1.5.4....

To update stuff:

pip install -U yt-dlp[default] yt-dlp-ejs
pip install mcp-neo4j-cypher mcp-server-git mcp_server_fetch -U
pip install mcp-neo4j-memory -U
pip install -U google_workspace_mcp
pip install -U docling docling-core  docling-ibm-models  --no-build-isolation -v --no-deps
pip install -U llm llm-cmd  llm-embed-onnx  llm-gemini  llm-gguf  llm-groq  llm-llama-cpp  llm-mistral  llm-mlc  llm-python  llm-sentence-transformers llama-index llama-index-readers-file
pip install -U datasets

pip install -U funasr --no-deps
pip install --upgrade scenedetect
pip install -U ctranslate2 -v --no-deps
pip install -U pyannote.audio
pip install -U crates maturin meson-python versioneer setuptools-scm
pip install --upgrade protobuf torchmetrics gast torchcodec --extra-index-url https://download.pytorch.org/whl/cpu

pip install -U whisperx --no-deps
pip install -U numba --no-deps
pip install -U pandas scipy

# Do not update via pip: torch or torchaudio ! use apt for that: 'apt reinstall python3-torchaudio python3-torch python3-numba python3-numpy'
apt reinstall python-numpy
# rm -rf '/data/data/com.termux/files/usr/lib/node_modules/@google/gemini-cli'
npm install -g @google/gemini-cli
# Or, if fails, then:
# npm install -g @google/gemini-cli --ignore-scripts --force
gemini --version
npm install -g @googleworkspace/cli --ignore-scripts
And patch it : patch /data/data/com.termux/files/usr/lib/node_modules/@googleworkspace/cli/binary.js < gws.patch, see https://github.com/googleworkspace/cli/issues/271
#npm install -g moltbot-termux
If 'apt upgrade' fails, then: 'apt -o Dpkg::Options::=--force-overwrite install {package name}'
proot-distro login debian  --shared-tmp -- npm install -g @google/gemini-cli

End of processing 🐰 Aliases 2


Processing Debian local 🎋 .bash_aliases, ver 2.2.1 ...
Environment at start of script:
Linux localhost 6.17.0-PRoot-Distro #1 SMP PREEMPT_DYNAMIC Fri, 10 Oct 2025 00:00:00 +0000 aarch64 GNU/Linux
PATH: /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/games:/usr/games:/data/data/com.termux/files/usr/bin:/system/bin:/system/xbin:/data/data/com.termux/files/home/Downloads/GitHub/depot_tools
LD_LIBRARY_PATH:
CFLAGS:
LDFLAGS:
CPPFLAGS:
PKG_CONFIG_PATH:
C_INCLUDE_PATH:
CPLUS_INCLUDE_PATH:
USE_VULKAN:
rm: cannot remove '/usr/lib/python3.13/EXTERNALLY-MANAGED': No such file or directory
Use :
pip install --force-reinstall --target  /data/data/com.termux/files/usr/lib/python3.13/site-packages
or, better:
pip install --force-reinstall --root /data/data/com.termux/files/usr
to install to Termux from Prooted Debian

Environment at the end of script, local 🎋 prooted system:
Linux localhost 6.17.0-PRoot-Distro #1 SMP PREEMPT_DYNAMIC Fri, 10 Oct 2025 00:00:00 +0000 aarch64 GNU/Linux
PATH: /usr/local/lib/python3.13/dist-packages/bin:/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/games:/usr/games:/system/bin:/system/xbin:/data/data/com.termux/files/home/.local/bin/:/root/go/bin
LD_LIBRARY_PATH: /usr/local/lib:/usr/local/lib/python3.13/dist-packages/lib:/usr/local/lib/python3.13/dist-packages/torchaudio/lib:/usr/local/lib/python3.13/dist-packages/lib:/usr/local/lib/python3.13/dist-packages/llama_cpp/lib:
CFLAGS:
LDFLAGS: -lpython3.13
CPPFLAGS:
C_INCLUDE_PATH:
CPLUS_INCLUDE_PATH: /sources/third_party/vulkan/src/include/:
USE_VULKAN:
LLVM_ROOT:
PKG_CONFIG_PATH:
bash: /root/.cargo/env: No such file or directory
root@localhost:~# cd ~/Downloads/GitHub/audioset_tagging_cnn/ && bash audio_me.sh /storage/emulated/0/Movies/Jolly_Mom_Has_No_Ide.2.mp4
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
Using torchaudio version: 2.10.0
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
Computed left margin: 338px (frac: 0.264)
Saved visualization to: /storage/emulated/0/Movies/Jolly_Mom_Has_No_Ide.2_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/eventogram.png
📊  Generating AI-friendly event summary files…
Saved summary events CSV to: /storage/emulated/0/Movies/Jolly_Mom_Has_No_Ide.2_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/summary_events.csv
📊  Generating detailed AI-friendly event delta JSON (with RLE compression)…
Saved AI-friendly delta JSON (compressed) to: /storage/emulated/0/Movies/Jolly_Mom_Has_No_Ide.2_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/detailed_events_delta_ai_attention_friendly.json
📊  Generating interactive Plotly dashboard…
Saved interactive dashboard to: /storage/emulated/0/Movies/Jolly_Mom_Has_No_Ide.2_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/interactive_dashboard.html
⏲  🗃️  Reminder: input file duration: 2925.110567
        Command being timed: "python /root/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py --model_type=Cnn14_DecisionLevelMax --checkpoint_path=/storage/emulated/0/LLMs/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth --cuda /storage/emulated/0/Movies/Jolly_Mom_Has_No_Ide.2.mp4"
        User time (seconds): 1984.91
        System time (seconds): 72.26
        Percent of CPU this job got: 411%
        Elapsed (wall clock) time (h:mm:ss or m:ss): 8:20.42
        Average shared text size (kbytes): 0
        Average unshared data size (kbytes): 0
        Average stack size (kbytes): 0
        Average total size (kbytes): 0
        Maximum resident set size (kbytes): 4405724
        Average resident set size (kbytes): 0
        Major (requiring I/O) page faults: 1010
        Minor (reclaiming a frame) page faults: 3198995
        Voluntary context switches: 398144
        Involuntary context switches: 495662
        Swaps: 0
        File system inputs: 5685320
        File system outputs: 359400
        Socket messages sent: 0
        Socket messages received: 0
        Signals delivered: 0
        Page size (bytes): 4096
        Exit status: 0

📊  Inference complete. Launching Shapash Correlations Dashboard...
loading /storage/emulated/0/Movies/Jolly_Mom_Has_No_Ide.2_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/full_event_log.csv
explaining target: Speech (avg prob: 0.5814)
analyzing 13 features for Speech
training model with 20 estimators...
compiling shapash on 219 samples...
INFO: Shap explainer type - <shap.explainers._tree.TreeExplainer object at 0x7bf81a96a0>
computing global importance...

--- TOP ACOUSTIC WEIGHTS (PREDICTORS FOR SPEECH) ---
Male speech, man speaking     0.401711
Inside, small room            0.151086
Inside, large room or hall    0.125279
Music                         0.066330
Conversation                  0.059912
Radio                         0.040232
Outside, rural or natural     0.030390
Animal                        0.029316
Narration, monologue          0.023381
Outside, urban or manmade     0.023216
dtype: float64
--------------------------------------------------

saving acoustic brain to: /storage/emulated/0/Movies/Jolly_Mom_Has_No_Ide.2_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/shapash_brain_speech.pkl
initializing webapp...
--------------------------------------------------
DASHBOARD STARTING (Explaining Speech)...
Open your browser at: http://localhost:8050
--------------------------------------------------

real    1m13.180s
user    0m50.870s
sys     0m6.284s

root@localhost:~/Downloads/GitHub/audioset_tagging_cnn#

