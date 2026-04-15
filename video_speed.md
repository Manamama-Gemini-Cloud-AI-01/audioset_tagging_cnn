Video writing speed. 

Do compare A with B


A. 
```
⏲  🗃️  File duration: 0:15:57
Loaded waveform shape: torch.Size([1, 15316800]), sample rate: 16000
Processed waveform shape: (30633600,)
📊  Processing in 3.0m chunks. CSV: 100 FPS | RAM/Vis: 2 FPS
Chunk at 0m done. RAM usage: 361 vis-frames cached.
Chunk at 3m done. RAM usage: 722 vis-frames cached.
Chunk at 6m done. RAM usage: 1083 vis-frames cached.
Chunk at 9m done. RAM usage: 1444 vis-frames cached.
Chunk at 12m done. RAM usage: 1805 vis-frames cached.
Chunk at 15m done. RAM usage: 1920 vis-frames cached.
Aggregation complete. Internal RAM resolution: 2 FPS
Computed left margin: 285px (frac: 0.223)
Saved visualization to: /home/zezen/Music/Music_on_Data/Recorded/2025-12-16 17.02.45_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/eventogram.png
📊  Generating AI-friendly event summary files…
Saved summary events CSV to: /home/zezen/Music/Music_on_Data/Recorded/2025-12-16 17.02.45_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/summary_events.csv
📊  Generating detailed AI-friendly event delta JSON (with RLE compression)…
Saved AI-friendly delta JSON (compressed) to: /home/zezen/Music/Music_on_Data/Recorded/2025-12-16 17.02.45_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/detailed_events_delta_ai_attention_friendly.json
📊  Generating interactive Plotly dashboard…
Saved interactive dashboard to: /home/zezen/Music/Music_on_Data/Recorded/2025-12-16 17.02.45_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/interactive_dashboard.html
🎞  Rendering static eventogram video …
MoviePy - Building video /home/zezen/Music/Music_on_Data/Recorded/2025-12-16 17.02.45_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/2025-12-16 17.02.45_audioset_tagging_cnn_eventogram_static.mp4.
MoviePy - Writing audio in 2025-12-16 17.02.45_audioset_tagging_cnn_eventogram_staticTEMP_MPY_wvf_snd.mp3
MoviePy - Done.                                                                                                                                                                               
MoviePy - Writing video /home/zezen/Music/Music_on_Data/Recorded/2025-12-16 17.02.45_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/2025-12-16 17.02.45_audioset_tagging_cnn_eventogram_static.mp4

frame_index:   0%|▌                                                                                                                             | 119/28719 [00:05<22:27, 21.22it/s, now=None]
frame_index:   1%|█▍                                                                                                                            | 337/28719 [00:16<22:16, 21.23it/s, now=None]
frame_index:   9%|██████████▉                                                                                                                  | 2526/28719 [01:58<21:29, 20.31it/s, now=None]
frame_index:   9%|███████████                                                                                                                  | 2536/28719 [01:59<21:58, 19.86it/s, now=None]
frame_index:   9%|███████████▌                                                                                                                 | 2643/28719 [02:04<20:39, 21.04it/s, now=None]
frame_index:  16%|███████████████████▉                                                                                                         | 4587/28719 [03:39<19:17, 20.85it/s, now=None]
```

B. 
```
echo
zezen@above-hp2-silver:~/Downloads/GitHub/audioset_tagging_cnn$ /usr/bin/time -v bash audio_me.sh "/home/zezen/Music/Music_on_Data/Recorded/2025-12-16 17.02.45.ogg" --dynamic_eventogram
--- AudioSet Tagging CNN Inference (SED Mode) ---
Model Type: Cnn14_DecisionLevelMax
Checkpoint path: /media/zezen/HP_P7_Data/Temp/AI_models/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth
Input file and parameters: /home/zezen/Music/Music_on_Data/Recorded/2025-12-16 17.02.45.ogg --dynamic_eventogram
Add  '--dynamic_eventogram'  or  '--static_eventogram'  for a video of the graph with a moving marker.

Note: If you see errors with sox / mp3 parsing → install sox + libsox-fmt-mp3, test it via 'sox --info /home/zezen/Music/Music_on_Data/Recorded/2025-12-16 17.02.45.ogg --dynamic_eventogram ' 
Note: Paths provided as command-line arguments should be absolute to avoid ambiguity.
Note: if no torchaudio backends, do find a patch file, usually called speechbrain.patch, and patch speechbrain itself. 
2026-04-15 12:49:47 above-hp2-silver numexpr.utils[248509] INFO NumExpr defaulting to 8 threads.
Eventogrammer, version 6.6.2. Material Changes: 
 * Broken the 'Aggregation Bottleneck': High-res data (100 FPS) is now streamed directly to disk (CSV) during inference.
 * 50x RAM Optimization: Internal RAM structures (Eventogram/Spectrogram) are now max-pooled to 2 FPS.
 * Chunked Post-processing: Spectrogram is calculated in 3m segments to prevent OOM spikes on long files. CSV reflects samples at lower frequence: size saving. left_frac restored. 
Notes: a file of duration of 30 mins requires 6GB RAM to process, with the processing time ratio: 1 second of orignal duration : 10 seconds to process on a regular 200 GFLOPs, 4 core CPU.
If the file is too long, use e.g. this to split:
mkdir split_input_media && cd split_input_media && ffmpeg -i /home/zezen/Music/Music_on_Data/Recorded/2025-12-16 17.02.45.ogg -c copy -f segment -segment_time 1200 output_%03d.mp4
This script is an adaptation of: https://github.com/qiuqiangkong/audioset_tagging_cnn so see there if something be amiss.
Note on the  models:
* `Cnn14_DecisionLevelMax_mAP=0.385.pth` (Sound Event Detection): This model uses Decision-level pooling. It calculates classification probabilities for every small segment of time first, and only then takes the maximum probability to represent the whole clip. Resolution: Cnn14_DecisionLevelMax is specifically designed for Sound Event Detection (SED). Because it maintains the time dimension through the classifier, it can output the framewise_output used by the inference script to generate the 'Eventograms' and the CSV logs.
* The other models are good for audio tagging: use the '--audio_tagging' switch for that mode.

Using moviepy version: 2.1.2
Using torchaudio version: 2.11.0+cpu
Using torchcodec version: 0.11.1+cpu
Tips: 'undefined symbol: torch_library_impl' or 'NotImplementedError':
This is often a version mismatch between torch and torchaudio/torchcodec, simply run:
pip install -U torch torchaudio torchcodec --extra-index-url https://download.pytorch.org/whl/cpu
If you see 'NotImplementedError: sys.platform = android' after an update:
1. Edit: /data/data/com.termux/files/usr/lib/python3.12/site-packages/torchcodec/_internally_replaced_utils.py
2. Change 'if sys.platform == "linux":' to 'if sys.platform == "android":' - it works.


Using device: cpu
Using CPU.
Copied AI analysis guide to: /home/zezen/Music/Music_on_Data/Recorded/2025-12-16 17.02.45_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/auditory_cognition_guide_template.md
/usr/lib/python3/dist-packages/paramiko/pkey.py:81: CryptographyDeprecationWarning: TripleDES has been moved to cryptography.hazmat.decrepit.ciphers.algorithms.TripleDES and will be removed from cryptography.hazmat.primitives.ciphers.algorithms in 48.0.0.
  "cipher": algorithms.TripleDES,
/usr/lib/python3/dist-packages/paramiko/transport.py:254: CryptographyDeprecationWarning: TripleDES has been moved to cryptography.hazmat.decrepit.ciphers.algorithms.TripleDES and will be removed from cryptography.hazmat.primitives.ciphers.algorithms in 48.0.0.
  "class": algorithms.TripleDES,

⏲  🗃️  File duration: 0:15:57
Loaded waveform shape: torch.Size([1, 15316800]), sample rate: 16000
Processed waveform shape: (30633600,)
📊  Processing in 3.0m chunks. CSV: 100 FPS | RAM/Vis: 2 FPS
Chunk at 0m done. RAM usage: 361 vis-frames cached.
Chunk at 3m done. RAM usage: 722 vis-frames cached.
Chunk at 6m done. RAM usage: 1083 vis-frames cached.
Chunk at 9m done. RAM usage: 1444 vis-frames cached.
Chunk at 12m done. RAM usage: 1805 vis-frames cached.
Chunk at 15m done. RAM usage: 1920 vis-frames cached.
Aggregation complete. Internal RAM resolution: 2 FPS
Computed left margin: 285px (frac: 0.223)
Saved visualization to: /home/zezen/Music/Music_on_Data/Recorded/2025-12-16 17.02.45_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/eventogram.png
📊  Generating AI-friendly event summary files…
Saved summary events CSV to: /home/zezen/Music/Music_on_Data/Recorded/2025-12-16 17.02.45_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/summary_events.csv
📊  Generating detailed AI-friendly event delta JSON (with RLE compression)…
Saved AI-friendly delta JSON (compressed) to: /home/zezen/Music/Music_on_Data/Recorded/2025-12-16 17.02.45_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/detailed_events_delta_ai_attention_friendly.json
📊  Generating interactive Plotly dashboard…
Saved interactive dashboard to: /home/zezen/Music/Music_on_Data/Recorded/2025-12-16 17.02.45_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/interactive_dashboard.html
🎞  Rendering dynamic eventogram video at 30 FPS…
MoviePy - Building video /home/zezen/Music/Music_on_Data/Recorded/2025-12-16 17.02.45_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/2025-12-16 17.02.45_audioset_tagging_cnn_eventogram_dynamic.mp4.
MoviePy - Writing audio in 2025-12-16 17.02.45_audioset_tagging_cnn_eventogram_dynamicTEMP_MPY_wvf_snd.mp3
MoviePy - Done.                                                                                                                                                                               
MoviePy - Writing video /home/zezen/Music/Music_on_Data/Recorded/2025-12-16 17.02.45_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/2025-12-16 17.02.45_audioset_tagging_cnn_eventogram_dynamic.mp4




frame_index:  33%|████████████████████████████████████████▉                                                                                    | 9392/28719 [21:08<40:07,  8.03it/s, now=None]
frame_index:  33%|████████████████████████████████████████▉                                                                                    | 9397/28719 [21:08<39:52,  8.08it/s, now=None]
frame_index:  33%|████████████████████████████████████████▉                                                                                    | 9400/28719 [21:09<39:42,  8.11it/s, now=None]
frame_index:  33%|████████████████████████████████████████▉                                                                                    | 9416/28719 [21:11<39:36,  8.12it/s, now=None] 

```



As you see, with static video the speed is much faster. Also, now the video created has 3 Hz of images (as CSV rows are  much less frequent), which causes video to be jerky, but I can live with it. 

But why such a difference in video writing speed? Maybe too many matplotlib calls per second still and smth is duplicated?  
