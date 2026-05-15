import time
from shapash import SmartExplainer
import os
import sys
import joblib

# Paths to the pkl files
pkl1 = "/home/zezen/Videos/Videos_on_P7/Something_very_weird_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/shapash_brain_speech.pkl"
pkl2 = "/home/zezen/Videos/Videos_on_P7/Claude_Mythos_-_High_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/shapash_brain_speech.pkl"

if not os.path.exists(pkl1) or not os.path.exists(pkl2):
    print("Error: PKL files not found.")
    sys.exit(1)

print("--- Launching App 1 on port 8050 ---")
try:
    # Load the whole object directly
    xpl1 = joblib.load(pkl1)
    app1 = xpl1.run_app(port=8050)
    print("App 1 is running.")
    time.sleep(3)
except Exception as e:
    print(f"Error launching App 1: {e}")
    sys.exit(1)

print("\n--- Attempting to launch App 2 on the SAME port 8050 ---")
try:
    xpl2 = joblib.load(pkl2)
    app2 = xpl2.run_app(port=8050)
    print("App 2 launched successfully. (This means Shapash handles the port or App 1 is dying)")
    time.sleep(3)
except Exception as e:
    print(f"\nSUCCESS: Port overlap caught: {e}")

print("\nCleaning up...")
if 'app1' in locals():
    print("Killing App 1")
    app1.kill()
if 'app2' in locals():
    print("Killing App 2")
    try:
        app2.kill()
    except:
        pass

print("Test complete.")
