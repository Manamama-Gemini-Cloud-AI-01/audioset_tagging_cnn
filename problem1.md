Works, but swap file increases at each pass and then OOM.  Teach me about the architecture first: why the old chunks are stored in memory. Now, if a file is longer than about 10 mins, this happens:

~/.../GitHub/audioset_tagging_cnn $ bash audio_me.sh "/storage/emulated/0/Music/Recordings/2026-04-12 11.35.00.ogg"



Using device: cpu

Using CPU.

Copied AI analysis guide to: /storage/emulated/0/Music/Recordings/2026-04-12 11.35.00_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/auditory_cognition_guide_template.md

⏲ 🗃️ File duration: 0:34:50

Loaded waveform shape: torch.Size([1, 33451200]), sample rate: 16000

Processed waveform shape: (66902400,)

Processing chunk: start=0, len=1

Chunk output shape: (18001, 527)

Processing chunk: start=5760000, len=1

Chunk output shape: (18001, 527)

Processing chunk: start=11520000, len=1

Chunk output shape: (18001, 527)

Processing chunk: start=17280000, len=1

Chunk output shape: (18001, 527)

Processing chunk: start=23040000, len=1

Chunk output shape: (18001, 527)

Processing chunk: start=28800000, len=1

Chunk output shape: (18001, 527)

Processing chunk: start=34560000, len=1

Chunk output shape: (18001, 527)

Processing chunk: start=40320000, len=1

Chunk output shape: (18001, 527)

Processing chunk: start=46080000, len=1

Chunk output shape: (18001, 527)

Processing chunk: start=51840000, len=1

Chunk output shape: (18001, 527)

Processing chunk: start=57600000, len=1
Using CPU.

Copied AI analysis guide to: /storage/emulated/0/Music/Recordings/2026-04-12 11.35.00_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/auditory_cognition_guide_template.md

⏲ 🗃️ File duration: 0:34:50

Loaded waveform shape: torch.Size([1, 33451200]), sample rate: 16000

Processed waveform shape: (66902400,)

Processing chunk: start=0, len=1

Chunk output shape: (18001, 527)

Processing chunk: start=5760000, len=1

Chunk output shape: (18001, 527)

Processing chunk: start=11520000, len=1

Chunk output shape: (18001, 527)

Processing chunk: start=17280000, len=1

Chunk output shape: (18001, 527)

Processing chunk: start=23040000, len=1

Chunk output shape: (18001, 527)

Processing chunk: start=28800000, len=1

Chunk output shape: (18001, 527)

Processing chunk: start=34560000, len=1

Chunk output shape: (18001, 527)

Processing chunk: start=40320000, len=1

Chunk output shape: (18001, 527)

Processing chunk: start=46080000, len=1

Chunk output shape: (18001, 527)

Processing chunk: start=51840000, len=1

Chunk output shape: (18001, 527)

Processing chunk: start=57600000, len=1

[OOM crash, at around 80 percent swap used, Termux is gone]



Ref: 

```
~/.../GitHub/audioset_tagging_cnn $ mediainfo "/storage/emulated/0/Music/Recordings/2026-04-12 11.35.00.ogg"
General
Complete name                            : /storage/emulated/0/Music/Recordings/2026-04-12 11.35.00.ogg
Format                                   : Ogg
File size                                : 9.52 MiB
Duration                                 : 34 min 50 s
Overall bit rate mode                    : Variable
Overall bit rate                         : 38.2 kb/s
Writing application                      : android-audio-recorder

Audio
ID                                       : 1775986501 (0x69DB6745)
Format                                   : Vorbis
Format settings, Floor                   : 1
Duration                                 : 34 min 50 s
Bit rate mode                            : Variable
Bit rate                                 : 56.0 kb/s
Channel(s)                               : 1 channel
Sampling rate                            : 16.0 kHz
Compression mode                         : Lossy
Stream size                              : 14.0 MiB
Writing library                          : libVorbis (Now 100% fewer shells) (20180316 (Now 100% fewer shells))

~/.../GitHub/audioset_tagging_cnn $ free
               total        used        free      shared  buff/cache   available
Mem:         5774008     3232524      177496      105736     2363988     2072204
Swap:        6291452     1723968     4567484
~/.../GitHub/audioset_tagging_cnn $ fastfetch
u0_a278@localhost

-----------------

OS: Android REL 11 aarch64
Host: realme 8 (RMX3085)
Kernel: Linux 4.14.186+
Uptime: 1 day, 22 hours, 9 mins
Packages: 142 (pacman), 905 (dpkg)
Shell: bash 5.3.9
DE: RMX3085_11_A.26
WM: WindowManager (SurfaceFlinger)
Terminal: /dev/pts/29 10.3p1
CPU: 2 x MT6785V/CD (8) @ 2.05 GHz
GPU: ARM Mali-G76 MC4 r0p0 (4) @ 0.01 GHz [Integrated]
Memory: 3.51 GiB / 5.51 GiB (64%)
Swap: 1.64 GiB / 6.00 GiB (27%)
Disk (/): 1.97 GiB / 1.97 GiB (100%) - ext4 [Read-only]
Disk (/storage/emulated): 92.86 GiB / 108.01 GiB (86%) - fuse
Local IP (bt-pan): 192.168.44.1/24
Local IP (ccmni0): 100.112.105.100/8
Local IP (rndis0): 10.110.81.235/24
Battery: 97% [USB Connected, Charging]
Locale: en_US.UTF-8

~/.../GitHub/audioset_tagging_cnn $ 
```


