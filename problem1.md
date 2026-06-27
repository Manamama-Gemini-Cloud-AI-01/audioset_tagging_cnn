Discuss first only the nature of these problems so far: 

zezen@zezen-HP-Pavilion-x360-Convertible:~/Downloads/GitHub/audioset_tagging_cnn$ git_workflow_pull_then_commit_preferred 
The git sync (git : pull, add, commit, push) script does not assume <branch> is set, e.g., main or current branch name, because if a wrong branch is gven, then the script fails
On branch master
Your branch is up to date with 'origin/master'.

Changes not staged for commit:
  (use "git add/rm <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	deleted:    audio_recordings1.sh

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	audio_recordings_of_dimowner.audiorecorder_android.sh

no changes added to commit (use "git add" and/or "git commit -a")
Already up to date.
On branch master
Your branch is up to date with 'origin/master'.

Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
	renamed:    audio_recordings1.sh -> audio_recordings_of_dimowner.audiorecorder_android.sh

[master ce5cc05] Automated commit, probably minor changes
 1 file changed, 0 insertions(+), 0 deletions(-)
 rename audio_recordings1.sh => audio_recordings_of_dimowner.audiorecorder_android.sh (100%)
Enumerating objects: 3, done.
Counting objects: 100% (3/3), done.
Delta compression using up to 4 threads
Compressing objects: 100% (2/2), done.
Writing objects: 100% (2/2), 284 bytes | 142.00 KiB/s, done.
Total 2 (delta 1), reused 0 (delta 0)
remote: Resolving deltas: 100% (1/1), completed with 1 local object.
To github.com:Manamama-Gemini-Cloud-AI-01/audioset_tagging_cnn.git
   c640aa1..ce5cc05  master -> master
On branch master
Your branch is up to date with 'origin/master'.

nothing to commit, working tree clean
zezen@zezen-HP-Pavilion-x360-Convertible:~/Downloads/GitHub/audioset_tagging_cnn$ bash audio_me.sh "/mnt/HP_P7_Data/Audio/records/2026-06-26 22.10.50.mp3"
--- AudioSet Tagging CNN Inference (SED Mode) ---
Model Type: Cnn14_DecisionLevelMax
Checkpoint path: /media/zezen/HP_P7_Data/Temp/AI_models/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth
Input file and parameters: /mnt/HP_P7_Data/Audio/records/2026-06-26 22.10.50.mp3
Add  --dynamic_eventogram   or  --static_eventogram  for a video of the graph with a moving marker.

Note: Paths provided as command-line arguments should be absolute to avoid ambiguity.
We are checking the real path of the first argument...
Traceback (most recent call last):
  File "/home/zezen/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py", line 25, in <module>
    import matplotlib
  File "/home/zezen/.local/lib/python3.12/site-packages/matplotlib/__init__.py", line 263, in <module>
    _check_versions()
  File "/home/zezen/.local/lib/python3.12/site-packages/matplotlib/__init__.py", line 257, in _check_versions
    module = importlib.import_module(modname)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/importlib/__init__.py", line 90, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zezen/.local/lib/python3.12/site-packages/kiwisolver/__init__.py", line 8, in <module>
    from ._cext import (
ModuleNotFoundError: No module named 'kiwisolver._cext'
Command exited with non-zero status 1
	Command being timed: "python /home/zezen/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py --model_type=Cnn14_DecisionLevelMax --checkpoint_path=/media/zezen/HP_P7_Data/Temp/AI_models/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth --cuda /mnt/HP_P7_Data/Audio/records/2026-06-26 22.10.50.mp3"
	User time (seconds): 0.78
	System time (seconds): 0.05
	Percent of CPU this job got: 151%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:00.55
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 53860
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 25
	Minor (reclaiming a frame) page faults: 9944
	Voluntary context switches: 83
	Involuntary context switches: 341
	Swaps: 0
	File system inputs: 7200
	File system outputs: 1000
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 1

zezen@zezen-HP-Pavilion-x360-Convertible:~/Downloads/GitHub/audioset_tagging_cnn$ bash audio_me.sh "/mnt/HP_P7_Data/Audio/records/2026-06-26 22.10.50.mp3"^C
zezen@zezen-HP-Pavilion-x360-Convertible:~/Downloads/GitHub/audioset_tagging_cnn$ pipe -U matplotlib
Special pip install, verbose etc: 
Using pip 26.1.2 from /home/zezen/.local/lib/python3.12/site-packages/pip (python 3.12)
Defaulting to user installation because normal site-packages is not writeable
Looking in indexes: https://pypi.org/simple, https://termux-user-repository.github.io/pypi/
Requirement already satisfied: matplotlib in /home/zezen/.local/lib/python3.12/site-packages (3.9.4)
Collecting matplotlib
  Obtaining dependency information for matplotlib from https://files.pythonhosted.org/packages/94/95/7f522393c88313336b20d70fc849555757b2e5febc22b83b3a3f0fd4bce9/matplotlib-3.11.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata
  Downloading matplotlib-3.11.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (80 kB)
Requirement already satisfied: contourpy>=1.0.1 in /home/zezen/.local/lib/python3.12/site-packages (from matplotlib) (1.3.2)
Requirement already satisfied: cycler>=0.10 in /home/zezen/.local/lib/python3.12/site-packages (from matplotlib) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /home/zezen/.local/lib/python3.12/site-packages (from matplotlib) (4.60.1)
Requirement already satisfied: kiwisolver>=1.3.1 in /home/zezen/.local/lib/python3.12/site-packages (from matplotlib) (1.4.9)
Requirement already satisfied: numpy>=1.25 in /home/zezen/.local/lib/python3.12/site-packages (from matplotlib) (2.4.6)
Requirement already satisfied: packaging>=20.0 in /home/zezen/.local/lib/python3.12/site-packages (from matplotlib) (25.0)
Requirement already satisfied: pillow>=9 in /home/zezen/.local/lib/python3.12/site-packages (from matplotlib) (12.2.0)
Requirement already satisfied: pyparsing>=3 in /home/zezen/.local/lib/python3.12/site-packages (from matplotlib) (3.2.5)
Requirement already satisfied: python-dateutil>=2.7 in /home/zezen/.local/lib/python3.12/site-packages (from matplotlib) (2.9.0.post0)
Requirement already satisfied: six>=1.5 in /home/zezen/.local/lib/python3.12/site-packages (from python-dateutil>=2.7->matplotlib) (1.17.0)
Downloading matplotlib-3.11.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (10.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10.0/10.0 MB 61.4 kB/s  0:03:31
Installing collected packages: matplotlib
  Attempting uninstall: matplotlib
    Found existing installation: matplotlib 3.9.4
    Uninstalling matplotlib-3.9.4:
      Removing file or directory /home/zezen/.local/lib/python3.12/site-packages/__pycache__/pylab.cpython-312.pyc
      Removing file or directory /home/zezen/.local/lib/python3.12/site-packages/matplotlib-3.9.4.dist-info/
      Removing file or directory /home/zezen/.local/lib/python3.12/site-packages/matplotlib/
      Removing file or directory /home/zezen/.local/lib/python3.12/site-packages/mpl_toolkits/axes_grid1/
      Removing file or directory /home/zezen/.local/lib/python3.12/site-packages/mpl_toolkits/axisartist/
      Removing file or directory /home/zezen/.local/lib/python3.12/site-packages/mpl_toolkits/mplot3d/
      Removing file or directory /home/zezen/.local/lib/python3.12/site-packages/pylab.py
      Successfully uninstalled matplotlib-3.9.4
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
open-interpreter 0.4.3 requires selenium<5.0.0,>=4.24.0, which is not installed.
music21 8.3.0 requires webcolors>=1.5, which is not installed.
vidstab 1.7.4 requires progress, which is not installed.
pyannote-audio 4.0.5 requires torchcodec>=0.7.0, but you have torchcodec 0.0.0.dev0 which is incompatible.
open-interpreter 0.4.3 requires anthropic<0.38.0,>=0.37.1, but you have anthropic 0.112.0 which is incompatible.
open-interpreter 0.4.3 requires psutil<6.0.0,>=5.9.6, but you have psutil 7.1.3 which is incompatible.
open-interpreter 0.4.3 requires starlette<0.38.0,>=0.37.2, but you have starlette 0.48.0 which is incompatible.
open-interpreter 0.4.3 requires tiktoken<0.8.0,>=0.7.0, but you have tiktoken 0.13.0 which is incompatible.
open-interpreter 0.4.3 requires typer<0.13.0,>=0.12.5, but you have typer 0.21.2 which is incompatible.
flair 0.15.1 requires transformers[sentencepiece]<5.0.0,>=4.25.0, but you have transformers 5.12.1 which is incompatible.
shapash 2.8.1 requires dash<3.0.0,>=2.3.1, but you have dash 4.0.0 which is incompatible.
shapash 2.8.1 requires pandas<3.0.0,>=2.2.2, but you have pandas 3.0.3 which is incompatible.
shapash 2.8.1 requires plotly<6.0.0,>=5.0.0, but you have plotly 6.5.2 which is incompatible.
shapash 2.8.1 requires scikit-learn<1.6.0,>=1.4.2, but you have scikit-learn 1.9.0 which is incompatible.
bliss 2.3.0 requires matplotlib==3.9.*, but you have matplotlib 3.11.0 which is incompatible.
bliss 2.3.0 requires numpy<2,>=1.21.1, but you have numpy 2.4.6 which is incompatible.
bliss 2.3.0 requires pandas<2.4, but you have pandas 3.0.3 which is incompatible.
bliss 2.3.0 requires pillow<12.1, but you have pillow 12.2.0 which is incompatible.
bliss 2.3.0 requires redis<5.0.1,>=4, but you have redis 8.0.1 which is incompatible.
bliss 2.3.0 requires requests<2.33, but you have requests 2.34.2 which is incompatible.
imat 3.2.2 requires matplotlib~=3.7.1, but you have matplotlib 3.11.0 which is incompatible.
imat 3.2.2 requires numpy~=1.23.5, but you have numpy 2.4.6 which is incompatible.
imat 3.2.2 requires pandas~=2.0.2, but you have pandas 3.0.3 which is incompatible.
imat 3.2.2 requires requests~=2.31.0, but you have requests 2.34.2 which is incompatible.
imat 3.2.2 requires scipy~=1.10.1, but you have scipy 1.16.3 which is incompatible.
imat 3.2.2 requires tokenizers~=0.13.3, but you have tokenizers 0.22.1 which is incompatible.
imat 3.2.2 requires tqdm~=4.65.0, but you have tqdm 4.67.1 which is incompatible.
gradio 3.41.2 requires aiofiles<24.0,>=22.0, but you have aiofiles 25.1.0 which is incompatible.
gradio 3.41.2 requires numpy~=1.0, but you have numpy 2.4.6 which is incompatible.
gradio 3.41.2 requires pandas<3.0,>=1.0, but you have pandas 3.0.3 which is incompatible.
gradio 3.41.2 requires pillow<11.0,>=8.0, but you have pillow 12.2.0 which is incompatible.
gradio 3.41.2 requires websockets<12.0,>=10.0, but you have websockets 15.0.1 which is incompatible.
tts 0.22.0 requires pandas<2.0,>=1.4, but you have pandas 3.0.3 which is incompatible.
whisperx 3.8.6 requires huggingface-hub<1.0.0, but you have huggingface-hub 1.21.0 which is incompatible.
whisperx 3.8.6 requires torch~=2.8.0, but you have torch 2.12.1+cpu which is incompatible.
whisperx 3.8.6 requires torchaudio~=2.8.0, but you have torchaudio 2.11.0+cpu which is incompatible.
whisperx 3.8.6 requires torchcodec<0.8.0,>=0.6.0; (sys_platform == "linux" and platform_machine == "x86_64") or sys_platform == "darwin" or sys_platform == "win32", but you have torchcodec 0.0.0.dev0 which is incompatible.
whisperx 3.8.6 requires torchvision~=0.23.0, but you have torchvision 0.27.1+cpu which is incompatible.
Successfully installed matplotlib-3.11.0
	Command being timed: "pip install --no-build-isolation -v --retries 10 --resume-retries 50 --timeout 60 --no-binary :all --extra-index-url https://termux-user-repository.github.io/pypi/ -U matplotlib"
	User time (seconds): 11.53
	System time (seconds): 2.21
	Percent of CPU this job got: 6%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 3:47.70
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 121180
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 3
	Minor (reclaiming a frame) page faults: 45393
	Voluntary context switches: 30505
	Involuntary context switches: 27057
	Swaps: 0
	File system inputs: 267344
	File system outputs: 133728
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0
zezen@zezen-HP-Pavilion-x360-Convertible:~/Downloads/GitHub/audioset_tagging_cnn$ pipe -U matplotlib
Special pip install, verbose etc: 
Using pip 26.1.2 from /home/zezen/.local/lib/python3.12/site-packages/pip (python 3.12)
Defaulting to user installation because normal site-packages is not writeable
^CERROR: Operation cancelled by user
Command exited with non-zero status 1
	Command being timed: "pip install --no-build-isolation -v --retries 10 --resume-retries 50 --timeout 60 --no-binary :all --extra-index-url https://termux-user-repository.github.io/pypi/ -U matplotlib"
	User time (seconds): 1.09
	System time (seconds): 0.12
	Percent of CPU this job got: 98%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:01.23
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 44116
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 0
	Minor (reclaiming a frame) page faults: 14706
	Voluntary context switches: 16
	Involuntary context switches: 276
	Swaps: 0
	File system inputs: 0
	File system outputs: 8
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 1
zezen@zezen-HP-Pavilion-x360-Convertible:~/Downloads/GitHub/audioset_tagging_cnn$ bash audio_me.sh "/mnt/HP_P7_Data/Audio/records/2026-06-26 22.10.50.mp3"
--- AudioSet Tagging CNN Inference (SED Mode) ---
Model Type: Cnn14_DecisionLevelMax
Checkpoint path: /media/zezen/HP_P7_Data/Temp/AI_models/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth
Input file and parameters: /mnt/HP_P7_Data/Audio/records/2026-06-26 22.10.50.mp3
Add  --dynamic_eventogram   or  --static_eventogram  for a video of the graph with a moving marker.

Note: Paths provided as command-line arguments should be absolute to avoid ambiguity.
We are checking the real path of the first argument...
Traceback (most recent call last):
  File "/home/zezen/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py", line 25, in <module>
    import matplotlib
  File "/home/zezen/.local/lib/python3.12/site-packages/matplotlib/__init__.py", line 265, in <module>
    _check_versions()
  File "/home/zezen/.local/lib/python3.12/site-packages/matplotlib/__init__.py", line 259, in _check_versions
    module = importlib.import_module(modname)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/importlib/__init__.py", line 90, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zezen/.local/lib/python3.12/site-packages/kiwisolver/__init__.py", line 8, in <module>
    from ._cext import (
ModuleNotFoundError: No module named 'kiwisolver._cext'
Command exited with non-zero status 1
	Command being timed: "python /home/zezen/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py --model_type=Cnn14_DecisionLevelMax --checkpoint_path=/media/zezen/HP_P7_Data/Temp/AI_models/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth --cuda /mnt/HP_P7_Data/Audio/records/2026-06-26 22.10.50.mp3"
	User time (seconds): 0.75
	System time (seconds): 0.08
	Percent of CPU this job got: 138%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:00.60
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 47660
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 0
	Minor (reclaiming a frame) page faults: 8084
	Voluntary context switches: 5
	Involuntary context switches: 265
	Swaps: 0
	File system inputs: 0
	File system outputs: 0
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 1

zezen@zezen-HP-Pavilion-x360-Convertible:~/Downloads/GitHub/audioset_tagging_cnn$ pipe -U kiwisolver
Special pip install, verbose etc: 
Using pip 26.1.2 from /home/zezen/.local/lib/python3.12/site-packages/pip (python 3.12)
Defaulting to user installation because normal site-packages is not writeable
Looking in indexes: https://pypi.org/simple, https://termux-user-repository.github.io/pypi/
Requirement already satisfied: kiwisolver in /home/zezen/.local/lib/python3.12/site-packages (1.4.9)
Collecting kiwisolver
  Obtaining dependency information for kiwisolver from https://files.pythonhosted.org/packages/c4/13/680c54afe3e65767bed7ec1a15571e1a2f1257128733851ade24abcefbcc/kiwisolver-1.5.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata
  Downloading kiwisolver-1.5.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (5.1 kB)
Downloading kiwisolver-1.5.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (1.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.5/1.5 MB 42.8 kB/s  0:00:31
Installing collected packages: kiwisolver
  Attempting uninstall: kiwisolver
    Found existing installation: kiwisolver 1.4.9
    Uninstalling kiwisolver-1.4.9:
      Removing file or directory /home/zezen/.local/lib/python3.12/site-packages/kiwisolver-1.4.9.dist-info/
      Removing file or directory /home/zezen/.local/lib/python3.12/site-packages/kiwisolver/
      Successfully uninstalled kiwisolver-1.4.9
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
music21 8.3.0 requires webcolors>=1.5, which is not installed.
vidstab 1.7.4 requires progress, which is not installed.
flair 0.15.1 requires transformers[sentencepiece]<5.0.0,>=4.25.0, but you have transformers 5.12.1 which is incompatible.
shapash 2.8.1 requires dash<3.0.0,>=2.3.1, but you have dash 4.0.0 which is incompatible.
shapash 2.8.1 requires pandas<3.0.0,>=2.2.2, but you have pandas 3.0.3 which is incompatible.
shapash 2.8.1 requires plotly<6.0.0,>=5.0.0, but you have plotly 6.5.2 which is incompatible.
shapash 2.8.1 requires scikit-learn<1.6.0,>=1.4.2, but you have scikit-learn 1.9.0 which is incompatible.
bliss 2.3.0 requires matplotlib==3.9.*, but you have matplotlib 3.11.0 which is incompatible.
bliss 2.3.0 requires numpy<2,>=1.21.1, but you have numpy 2.4.6 which is incompatible.
bliss 2.3.0 requires pandas<2.4, but you have pandas 3.0.3 which is incompatible.
bliss 2.3.0 requires pillow<12.1, but you have pillow 12.2.0 which is incompatible.
bliss 2.3.0 requires redis<5.0.1,>=4, but you have redis 8.0.1 which is incompatible.
bliss 2.3.0 requires requests<2.33, but you have requests 2.34.2 which is incompatible.
imat 3.2.2 requires matplotlib~=3.7.1, but you have matplotlib 3.11.0 which is incompatible.
imat 3.2.2 requires numpy~=1.23.5, but you have numpy 2.4.6 which is incompatible.
imat 3.2.2 requires pandas~=2.0.2, but you have pandas 3.0.3 which is incompatible.
imat 3.2.2 requires requests~=2.31.0, but you have requests 2.34.2 which is incompatible.
imat 3.2.2 requires scipy~=1.10.1, but you have scipy 1.16.3 which is incompatible.
imat 3.2.2 requires tokenizers~=0.13.3, but you have tokenizers 0.22.1 which is incompatible.
imat 3.2.2 requires tqdm~=4.65.0, but you have tqdm 4.67.1 which is incompatible.
gradio 3.41.2 requires aiofiles<24.0,>=22.0, but you have aiofiles 25.1.0 which is incompatible.
gradio 3.41.2 requires numpy~=1.0, but you have numpy 2.4.6 which is incompatible.
gradio 3.41.2 requires pandas<3.0,>=1.0, but you have pandas 3.0.3 which is incompatible.
gradio 3.41.2 requires pillow<11.0,>=8.0, but you have pillow 12.2.0 which is incompatible.
gradio 3.41.2 requires websockets<12.0,>=10.0, but you have websockets 15.0.1 which is incompatible.
tts 0.22.0 requires pandas<2.0,>=1.4, but you have pandas 3.0.3 which is incompatible.
Successfully installed kiwisolver-1.5.0
	Command being timed: "pip install --no-build-isolation -v --retries 10 --resume-retries 50 --timeout 60 --no-binary :all --extra-index-url https://termux-user-repository.github.io/pypi/ -U kiwisolver"
	User time (seconds): 4.62
	System time (seconds): 0.32
	Percent of CPU this job got: 13%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:37.29
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 66660
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 0
	Minor (reclaiming a frame) page faults: 21995
	Voluntary context switches: 630
	Involuntary context switches: 1143
	Swaps: 0
	File system inputs: 592
	File system outputs: 20608
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0
zezen@zezen-HP-Pavilion-x360-Convertible:~/Downloads/GitHub/audioset_tagging_cnn$ bash audio_me.sh "/mnt/HP_P7_Data/Audio/records/2026-06-26 22.10.50.mp3"
--- AudioSet Tagging CNN Inference (SED Mode) ---
Model Type: Cnn14_DecisionLevelMax
Checkpoint path: /media/zezen/HP_P7_Data/Temp/AI_models/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth
Input file and parameters: /mnt/HP_P7_Data/Audio/records/2026-06-26 22.10.50.mp3
Add  --dynamic_eventogram   or  --static_eventogram  for a video of the graph with a moving marker.

Note: Paths provided as command-line arguments should be absolute to avoid ambiguity.
We are checking the real path of the first argument...
2026-06-27 21:06:00 zezen-HP-Pavilion-x360-Convertible matplotlib.font_manager[1413311] INFO Failed to extract font properties from /usr/share/fonts/truetype/noto/NotoColorEmoji.ttf: Non-scalable fonts are not supported
2026-06-27 21:06:01 zezen-HP-Pavilion-x360-Convertible matplotlib.font_manager[1413311] WARNING Matplotlib is building the font cache; this may take a moment.
2026-06-27 21:06:05 zezen-HP-Pavilion-x360-Convertible matplotlib.font_manager[1413311] INFO generated new fontManager
Stay tuned...
Traceback (most recent call last):
  File "/home/zezen/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py", line 60, in <module>
    import h5py
  File "/home/zezen/.local/lib/python3.12/site-packages/h5py/__init__.py", line 25, in <module>
    from . import _errors
ImportError: cannot import name '_errors' from partially initialized module 'h5py' (most likely due to a circular import) (/home/zezen/.local/lib/python3.12/site-packages/h5py/__init__.py)
Command exited with non-zero status 1
	Command being timed: "python /home/zezen/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py --model_type=Cnn14_DecisionLevelMax --checkpoint_path=/media/zezen/HP_P7_Data/Temp/AI_models/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth --cuda /mnt/HP_P7_Data/Audio/records/2026-06-26 22.10.50.mp3"
	User time (seconds): 9.86
	System time (seconds): 0.96
	Percent of CPU this job got: 84%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:12.73
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 264360
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 2
	Minor (reclaiming a frame) page faults: 56402
	Voluntary context switches: 10844
	Involuntary context switches: 2268
	Swaps: 0
	File system inputs: 590416
	File system outputs: 3528
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 1

zezen@zezen-HP-Pavilion-x360-Convertible:~/Downloads/GitHub/audioset_tagging_cnn$ 


