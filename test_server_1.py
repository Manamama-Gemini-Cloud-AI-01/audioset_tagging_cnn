import joblib
import os
import sys

pkl1 = "/home/zezen/Videos/Videos_on_P7/Something_very_weird_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/shapash_brain_speech.pkl"

if not os.path.exists(pkl1):
    print(f"Error: {pkl1} not found")
    sys.exit(1)

print("--- Launching Server 1 on Port 8050 ---")
try:
    xpl = joblib.load(pkl1)
    xpl.init_app()
    app = xpl.smartapp.app
    # app.run is a blocking call in the main thread
    app.run(host="0.0.0.0", port=8050, debug=False)
except Exception as e:
    print(f"Server 1 Error: {e}")
