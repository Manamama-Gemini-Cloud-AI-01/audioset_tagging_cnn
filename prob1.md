What is the difference between the two .pth model files? :


zezen@above-hp2-silver:~/Downloads/GitHub/audioset_tagging_cnn$ ll -rs
total 639624
     4 drwxrwxr-x  3 zezen zezen      4096 Oct  8 19:30  utils/
     4 drwxrwxr-x  2 zezen zezen      4096 Nov  2 14:43  Source/
     4 drwxrwxr-x  4 zezen zezen      4096 Feb 17 09:45  scripts/
     4 drwxrwxr-x  2 zezen zezen      4096 Oct  8 19:28  resources/
     4 -rw-rw-r--  1 zezen zezen        95 Oct  8 19:28  requirements.txt
    12 -rw-rw-r--  1 zezen zezen     12248 Oct  8 19:28  README.md
     4 drwxrwxr-x  3 zezen zezen      4096 Feb 22 23:33  pytorch/
     4 -rw-rw-r--  1 zezen zezen      1136 Feb 22 14:22  prob2.md
     4 drwxrwxr-x  2 zezen zezen      4096 Oct 25 10:30  outputs/
     4 drwxrwxr-x  3 zezen zezen      4096 Oct  8 19:28  metadata/
     4 drwxrwxr-x  2 zezen zezen      4096 Oct 13 00:47  logs/
     4 -rw-rw-r--  1 zezen zezen      1081 Oct  8 19:28  LICENSE.MIT
     4 -rw-rw-r--  1 zezen zezen       177 Oct 19 17:00  .gitignore
     4 drwxrwxr-x  8 zezen zezen      4096 Feb 24 09:16  .git/
     8 -rw-rw-r--  1 zezen zezen      5887 Oct 26 21:59  GEMINI.md
     4 drwxrwxr-x  3 zezen zezen      4096 Nov 13 15:36  docs/
319768 -rw-rw-r--  1 zezen zezen 327428481 Oct  8 19:34 'Cnn14_mAP=0.431.pth'
319764 -rw-rw-r--  1 zezen zezen 327428481 Oct  8 20:41 'Cnn14_DecisionLevelMax_mAP=0.385.pth'
     4 -rwxrwxr-x  1 zezen zezen       321 Feb 18 15:45  audio_me.sh*
     4 drwxrwxr-x  2 zezen zezen      4096 Nov 13 15:35  archive/
     4 drwxrwxr-x 43 zezen zezen      4096 Feb 23 17:06  ../
     4 drwxrwxr-x 13 zezen zezen      4096 Feb 23 00:29  ./
zezen@above-hp2-silver:~/Downloads/GitHub/audioset_tagging_cnn$ diff Cnn14_mAP\=0.431.pth Cnn14_DecisionLevelMax_mAP\=0.385.pth 
Binary files Cnn14_mAP=0.431.pth and Cnn14_DecisionLevelMax_mAP=0.385.pth differ
zezen@above-hp2-silver:~/Downloads/GitHub/audioset_tagging_cnn$ 




