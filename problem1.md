I wrote a couple of scripts here and in my daily work I do below: 


```
zezen@above-hp2-silver:~/Downloads/GitHub/audioset_tagging_cnn$ bash audio_me.sh "/home/zezen/Downloads/Test/20260302T082319 walk lights tram river.mp3"
--- AudioSet Tagging CNN Inference (SED Mode) ---
Model Type: Cnn14_DecisionLevelMax
Checkpoint path: /media/zezen/HP_P7_Data/Temp/AI_models/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth
Input file and parameters: /home/zezen/Downloads/Test/20260302T082319 walk lights tram river.mp3
Add  '--dynamic_eventogram'  or  '--static_eventogram'  for a video of the graph with a moving marker.

Note: If you see errors with sox / mp3 parsing → install sox + libsox-fmt-mp3, test it via 'sox --info /home/zezen/Downloads/Test/20260302T082319 walk lights tram river.mp3 ' 
Note: Paths provided as command-line arguments should be absolute to avoid ambiguity.
Eventogrammer, version 6.3.4. Recently changed:  * Fixed path bug in output directory naming. Added auto-sanitization of corrupt audio streams via FFmpeg. Integrated portable interactive Plotly dashboard (Top 50). Removed manifest generation. Added a static_eventogram argument. Reorder artifact creation logic.
Notes: a file of duration of 30 mins requires 6GB RAM to process, with the processing time ratio: 1 second of orignal duration : 10 seconds to process on a regular 200 GFLOPs, 4 core CPU.
If the file is too long, use e.g. this to split:
mkdir split_input_media && cd split_input_media && ffmpeg -i /home/zezen/Downloads/Test/20260302T082319 walk lights tram river.mp3 -c copy -f segment -segment_time 1200 output_%03d.mp4
This script is an adaptation of: https://github.com/qiuqiangkong/audioset_tagging_cnn so see there if something be amiss.
Note on models:
* `Cnn14_DecisionLevelMax_mAP=0.385.pth` (Sound Event Detection): This model uses Decision-level pooling. It calculates classification probabilities for every small segment of time first, and only then takes the maximum probability to represent the whole clip. Resolution: Cnn14_DecisionLevelMax is specifically designed for Sound Event Detection (SED). Because it maintains the time dimension through the classifier, it can output the framewise_output used by the inference script to generate the 'Eventograms' and the CSV logs.
* The other models are good for audio tagging: use the '--audio_tagging' switch for that mode.

Using moviepy version: 2.1.2
Using torchaudio version: 2.10.0+cpu

Using device: cpu
Using CPU.
Copied AI analysis guide to: /home/zezen/Downloads/Test/20260302T082319 walk lights tram river_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/auditory_cognition_guide_template.md
/usr/lib/python3/dist-packages/paramiko/pkey.py:81: CryptographyDeprecationWarning:

TripleDES has been moved to cryptography.hazmat.decrepit.ciphers.algorithms.TripleDES and will be removed from cryptography.hazmat.primitives.ciphers.algorithms in 48.0.0.

/usr/lib/python3/dist-packages/paramiko/transport.py:254: CryptographyDeprecationWarning:

TripleDES has been moved to cryptography.hazmat.decrepit.ciphers.algorithms.TripleDES and will be removed from cryptography.hazmat.primitives.ciphers.algorithms in 48.0.0.

⏲  🗃️  File duration: 0:03:07
Loaded waveform shape: torch.Size([1, 3006720]), sample rate: 16000
Processed waveform shape: (6013440,)
Processing chunk: start=0, len=1
Chunk output shape: (18001, 527)
Processing chunk: start=5760000, len=1
Chunk output shape: (793, 527)
Sound event detection result (time_steps x classes_num): (18794, 527)
Saved sound event detection visualization to: /home/zezen/Downloads/Test/20260302T082319 walk lights tram river_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/eventogram.png
Computed left margin (px): 165, axes bbox (fig-fraction): Bbox(x0=0.12890625, y0=0.07999999999999985, x1=1.0, y1=0.5043902439024389)
Saved full framewise CSV (wide matrix) to: /home/zezen/Downloads/Test/20260302T082319 walk lights tram river_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/full_event_log.csv
📊  Generating AI-friendly event summary files…
Saved summary events CSV to: /home/zezen/Downloads/Test/20260302T082319 walk lights tram river_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/summary_events.csv
📊  Generating detailed AI-friendly event delta JSON…
Saved AI-friendly delta JSON to: /home/zezen/Downloads/Test/20260302T082319 walk lights tram river_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/detailed_events_delta_ai_attention_friendly.json
📊  Generating interactive Plotly dashboard…
Saved interactive dashboard to: /home/zezen/Downloads/Test/20260302T082319 walk lights tram river_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/interactive_dashboard.html
⏲  🗃️  Reminder: input file duration: 187.92

real	0m38.026s
user	0m49.511s
sys	0m4.661s
 
nothing to commit, working tree clean
zezen@above-hp2-silver:~/Downloads/GitHub/audioset_tagging_cnn$ python scripts/
Archive/               Shapash_visualization/ Streetsounds/          
zezen@above-hp2-silver:~/Downloads/GitHub/audioset_tagging_cnn$ python scripts/Shapash_visualization/
Archive/                                  launch_correlations_dashboard.py          peek_at_shapash_brain_speech.pkl_file.py  shapash_from_GitHub/
zezen@above-hp2-silver:~/Downloads/GitHub/audioset_tagging_cnn$ python scripts/Shapash_visualization/launch_correlations_dashboard.py "/home/zezen/Downloads/Test/20260302T082319 walk lights tram river_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/full_event_log.csv"
loading /home/zezen/Downloads/Test/20260302T082319 walk lights tram river_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/full_event_log.csv
explaining target: Speech (avg prob: 0.2888)
analyzing 55 features for Speech
training model with 20 estimators...
compiling shapash on 220 samples...
INFO: Shap explainer type - <shap.explainers._tree.TreeExplainer object at 0x7dcd10017f80>
computing global importance...

--- TOP ACOUSTIC WEIGHTS (PREDICTORS FOR SPEECH) ---
Male speech, man speaking              0.609732
Hubbub, speech noise, speech babble    0.119829
Clickety-clack                         0.060698
Inside, small room                     0.033566
Thunder                                0.018051
Motorboat, speedboat                   0.017308
Rustle                                 0.016400
Boat, Water vehicle                    0.008805
Fire                                   0.008209
Gurgling                               0.007560
dtype: float64
--------------------------------------------------

saving acoustic brain to: /home/zezen/Downloads/Test/20260302T082319 walk lights tram river_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/shapash_brain_speech.pkl
initializing webapp...
--------------------------------------------------
DASHBOARD STARTING (Explaining Speech)...
Open your browser at: http://localhost:8050
--------------------------------------------------
Dash is running on http://0.0.0.0:8050/

INFO:dash.dash:Dash is running on http://0.0.0.0:8050/

 * Serving Flask app 'shapash.webapp.smart_app'
 * Debug mode: off
INFO:werkzeug:WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8050
 * Running on http://172.16.61.28:8050
INFO:werkzeug:Press CTRL+C to quit
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:16] "GET / HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:16] "GET /assets/material-icons.css?m=1772266833.9364164 HTTP/1.1" 304 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:16] "GET /_dash-component-suites/dash_daq/dash_daq.v0_6_0m1772266833.min.js HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:16] "GET /_dash-component-suites/dash_bootstrap_components/_components/dash_bootstrap_components.v1_7_1m1772266833.min.js HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:16] "GET /_dash-component-suites/dash/dash-renderer/build/dash_renderer.v4_0_0m1772266904.min.js HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:16] "GET /assets/style.css?m=1772266833.9374154 HTTP/1.1" 304 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:16] "GET /assets/jquery.js?m=1772266833.9364164 HTTP/1.1" 304 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:16] "GET /assets/main.js?m=1772266833.9364164 HTTP/1.1" 304 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:16] "GET /_dash-component-suites/dash/html/dash_html_components.v4_0_0m1772266904.min.js HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:16] "GET /_dash-component-suites/dash/dcc/dash_core_components.v4_0_0m1772266904.js HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:17] "GET /_dash-dependencies HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:17] "GET /assets/favicon.ico?m=1772266833.9354177 HTTP/1.1" 304 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:17] "GET /_dash-layout HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:17] "GET /_dash-component-suites/dash/dcc/async-dropdown.js HTTP/1.1" 304 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:17] "GET /_dash-component-suites/dash/dcc/async-graph.js HTTP/1.1" 304 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:17] "GET /_dash-component-suites/plotly/package_data/plotly.min.js HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:17] "GET /_dash-component-suites/dash/dash_table/async-highlight.js HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:17] "GET /_dash-component-suites/dash/dash_table/async-table.js HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:17] "GET /_dash-component-suites/dash/dcc/async-slider.js HTTP/1.1" 304 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:17] "GET /assets/shapash-fond-fonce.png HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:17] "GET /assets/settings.png HTTP/1.1" 304 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:17] "GET /assets/reload.png HTTP/1.1" 304 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:17] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:17] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:17] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:17] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:17] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:17] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:17] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:17] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:17] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:17] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:17] "POST /_dash-update-component HTTP/1.1" 204 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:17] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:17] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:17] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:18] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:18] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:18] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:18] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:18] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:18] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:18] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:18] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:18] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:18] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:18] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:18] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:18] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:18] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:18] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:18] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:19] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:19] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:19] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:19] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:19] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:20] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:20] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:30:20] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:35] "GET / HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "GET /_dash-component-suites/dash/dcc/async-dropdown.js HTTP/1.1" 304 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "GET /_dash-component-suites/dash/dcc/async-graph.js HTTP/1.1" 304 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "GET /_dash-component-suites/plotly/package_data/plotly.min.js HTTP/1.1" 304 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "GET /_dash-component-suites/dash/dash_table/async-highlight.js HTTP/1.1" 304 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "POST /_dash-update-component HTTP/1.1" 204 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "GET /_dash-component-suites/dash/dash_table/async-table.js HTTP/1.1" 304 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "GET /_dash-component-suites/dash/dcc/async-slider.js HTTP/1.1" 304 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "GET /assets/shapash-fond-fonce.png HTTP/1.1" 304 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "GET /assets/settings.png HTTP/1.1" 304 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:36] "GET /assets/reload.png HTTP/1.1" 304 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:37] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:37] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:37] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:37] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:37] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:37] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:38] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:38] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:38] "POST /_dash-update-component HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [02/Mar/2026 09:31:38] "POST /_dash-update-component HTTP/1.1" 200 -

```


and then I peek via web browser at this and at `file:///home/zezen/Downloads/Test/20260302T082319%20walk%20lights%20tram%20river_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/interactive_dashboard.html` 


I do that half half on Termux and on Ubuntu . 

I need to streamline /home/zezen/Downloads/GitHub/audioset_tagging_cnn/audio_me.sh somehow so that I do e.g. 

1. Copy the mp3 filename (think of Nautilus way, in short). 
2. Go to Termux or Terminator, type (soon to be aliased, I shall do so myself) audio_me and paste filename from 1 
3. Press Enter.  
4. The one resulting html file and the server be displayed via some xdg-open mechanism. 


Ask me what be unclear first. 


