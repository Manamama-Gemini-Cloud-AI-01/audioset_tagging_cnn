import joblib
import os
import sys

pkl2 = "/home/zezen/Videos/Videos_on_P7/Claude_Mythos_-_High_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/shapash_brain_speech.pkl"

if not os.path.exists(pkl2):
    print(f"Error: {pkl2} not found")
    sys.exit(1)

print("--- Launching Server 2 on Port 8050 ---")
try:
    xpl = joblib.load(pkl2)
    xpl.init_app()
    app = xpl.smartapp.app
    # app.run is a blocking call in the main thread
    app.run(host="0.0.0.0", port=8050, debug=False)
except Exception as e:
    print(f"Server 2 Error: {e}")
