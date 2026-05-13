zezen@zezen-HP-Pavilion-x360-Convertible:~/Downloads/GitHub/audioset_tagging_cnn$ fastfetch
zezen@zezen-HP-Pavilion-x360-Convertible
----------------------------------------
OS: Zorin OS 16.3 x86_64
Host: HP Pavilion x360 Convertible
Kernel: Linux 5.15.0-139-generic
Uptime: 22 hours, 14 mins
Packages: 3232 (dpkg), 30 (flatpak), 14 (snap)
Shell: bash 5.0.17
Display (LG Electronics LG TV): 1920x1080 in 7", 60 Hz [External]
Display (LGD048C): 1920x1080 in 13", 60 Hz [Built-in] *
DE: GNOME 3.38.4
WM: Mutter (Wayland)
WM Theme: ZorinOrange-Dark
Theme: ZorinOrange-Dark [GTK2/3/4]
Icons: ZorinOrange-Dark [GTK2/3/4]
Font: Cantarell (11pt) [GTK2/3/4]
Cursor: Adwaita (24px)
Terminal: terminator 3.10.20
Terminal Font: Mono (10pt)
CPU: Intel(R) Core(TM) i5-5200U (4) @ 2.70 GHz
GPU: Intel HD Graphics 5500 @ 0.90 GHz [Integrated]
Memory: 2.19 GiB / 7.67 GiB (29%)
Swap: 6.00 MiB / 8.00 GiB (0%)
Disk (/): 52.57 GiB / 65.58 GiB (80%) - ext4
Disk (/mnt/HP_P7_Data): 47.08 GiB / 50.88 GiB (93%) - ext4
Local IP (bnep0): 192.168.44.148/24
Locale: en_US.UTF-8



It works on another Ubuntu: 

zezen@zezen-HP-Pavilion-x360-Convertible:~/Downloads/GitHub/audioset_tagging_cnn$ bash audio_me.sh "/home/zezen/Videos/P5/Archive/Female_male_cry_Jebra.mp4.mp3"  --dynamic_eventogram
--- AudioSet Tagging CNN Inference (SED Mode) ---
Model Type: Cnn14_DecisionLevelMax
Checkpoint path: /media/zezen/HP_P7_Data/Temp/AI_models/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth
Input file and parameters: /home/zezen/Videos/P5/Archive/Female_male_cry_Jebra.mp4.mp3 --dynamic_eventogram
Add  --dynamic_eventogram   or  --static_eventogram  for a video of the graph with a moving marker.

Note: If you see errors with sox / mp3 parsing, then apt install 'sox' and 'libsox-fmt-mp3', test it via 'sox --info /home/zezen/Videos/P5/Archive/Female_male_cry_Jebra.mp4.mp3 --dynamic_eventogram ' 
Note: Paths provided as command-line arguments should be absolute to avoid ambiguity.
Note: if no torchaudio backends, do find a patch file, usually called speechbrain.patch, and patch speechbrain itself. 
2026-05-13 21:37:22 zezen-HP-Pavilion-x360-Convertible numexpr.utils[18869] INFO NumExpr defaulting to 4 threads.
Eventogrammer, version 6.8.8
This script is an adaptation of: https://github.com/qiuqiangkong/audioset_tagging_cnn so see there if something be amiss.
    Recent material Changes:  * Constants Promoted: vis_fps, output_fps, and adaptive_lookahead are now CLI arguments. * Speed Hack: Persistent Matplotlib figures with Artist Updates. * Visual Fix: Proper window centering for scrolling eventograms. Gemini AI rationalized code in many places.

Note on the models:
* `Cnn14_DecisionLevelMax_mAP=0.385.pth` (Sound Event Detection): This model uses Decision-level pooling. It calculates classification probabilities for every small segment of time first, and only then takes the maximum probability to represent the whole clip. Resolution: Cnn14_DecisionLevelMax is specifically designed for Sound Event Detection (SED). Because it maintains the time dimension through the classifier, it can output the framewise_output used by the inference script to generate the 'Eventograms' and the CSV logs.
* The other models are good for audio tagging: use the '--audio_tagging' switch for that mode.

Note on the processing speed: The processing time ratio now is: 15 second of the orignal duration takes 1 seconds to process on a regular 300 GFLOPs, 4 core CPU, without video visualizations.
It works 1.7 times faster in Prooted Debian than in Termux, see the comments why so.
Note on the out of memory crashes: close all other programs in Droid, especially the browser. Or just restart whole phone. Or do it in Recovery.
If the file is too long, use e.g. this to split:
mkdir split_input_media && cd split_input_media && ffmpeg -i /home/zezen/Videos/P5/Archive/Female_male_cry_Jebra.mp4.mp3 -c copy -f segment -segment_time 1200 output_%03d.mp4
Tips on bugs: 'undefined symbol: torch_library_impl' or 'NotImplementedError':
This is often a version mismatch between torch and torchaudio, simply run:
pip install -U torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install -U torchcodec --extra-index-url https://download.pytorch.org/whl/cpu
In e.g. Prooted Debian under Termux, torchcodec has dependencies on NVIDA .so files and the script errors, so you may need to git clone and pip install it from scratch (which works without a hitch) then.
If you see 'NotImplementedError: sys.platform = android' after an update:
1. Edit: /data/data/com.termux/files/usr/lib/python3.10/site-packages/torchaudio/_internally_replaced_utils.py
2. Change 'if sys.platform == "linux":' to 'if sys.platform == "android":' - it works.

Using moviepy version: 2.1.1
Using torchaudio version: 2.11.0+cpu
Using torchcodec version: 0.11.1+cpu

Using device: cpu
Using CPU.
Copied AI analysis guide to: /home/zezen/Videos/P5/Archive/Female_male_cry_Jebra.mp4_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/auditory_cognition_guide_template.md
Error: Failed to parse FFprobe JSON output for /home/zezen/Videos/P5/Archive/Female_male_cry_Jebra.mp4.mp3: Expecting value: line 1 column 1 (char 0)
Warning: Fallback duration probe failed for /home/zezen/Videos/P5/Archive/Female_male_cry_Jebra.mp4.mp3: could not convert string to float: ''
Error: Could not determine duration for /home/zezen/Videos/P5/Archive/Female_male_cry_Jebra.mp4.mp3. Exiting.
Warning: Duration probe failed for /home/zezen/Videos/P5/Archive/Female_male_cry_Jebra.mp4.mp3. Attempting MP3 recovery...
Error: Failed to parse FFprobe JSON output for /tmp/Female_male_cry_Jebra.mp4_recovered.mp3: Expecting value: line 1 column 1 (char 0)
Warning: Fallback duration probe failed for /tmp/Female_male_cry_Jebra.mp4_recovered.mp3: could not convert string to float: ''
Error: Could not determine duration for /tmp/Female_male_cry_Jebra.mp4_recovered.mp3. Exiting.
Error: Could not determine duration even after recovery. Exiting.
	Command being timed: "python /home/zezen/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py --model_type=Cnn14_DecisionLevelMax --checkpoint_path=/media/zezen/HP_P7_Data/Temp/AI_models/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth --cuda /home/zezen/Videos/P5/Archive/Female_male_cry_Jebra.mp4.mp3 --dynamic_eventogram"
	User time (seconds): 10.02
	System time (seconds): 0.85
	Percent of CPU this job got: 106%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:10.21
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 1292732
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 17
	Minor (reclaiming a frame) page faults: 275988
	Voluntary context switches: 114
	Involuntary context switches: 1378
	Swaps: 0
	File system inputs: 992
	File system outputs: 896
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0

⚠️  Warning: full_event_log.csv not found at /mnt/HP_P7_Data/Videos/Archive/Female_male_cry_Jebra.mp4_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/full_event_log.csv. Skipping dashboard.




