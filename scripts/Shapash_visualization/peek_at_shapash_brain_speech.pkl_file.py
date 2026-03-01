import joblib
import pandas as pd

# 1. Load the brain
brain_path = "/home/zezen/Videos/Videos_on_P7/Watch_me/SRT/1_Bugs_Life_original_/Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/shapash_brain_speech.pkl"
print(f"loading brain from: {brain_path}")
xpl = joblib.load(brain_path)

# 2. Check if importance is already computed
if xpl.features_imp is None:
    print("importance data not found in brain, computing now...")
    # This uses the pre-calculated SHAP values stored inside the .pkl
    xpl.compute_features_import()

# 3. Peak at the top 10 acoustic weights
print("\n--- TOP ACOUSTIC WEIGHTS LEARNED BY THIS BRAIN ---")
if isinstance(xpl.features_imp, pd.Series):
    print(xpl.features_imp.sort_values(ascending=False).head(10))
else:
    # In some cases it might be a list (multi-class)
    print(xpl.features_imp)

# 4. Check what else is inside
print("\n--- BRAIN INVENTORY ---")
print(f"Samples stored: {len(xpl.x_init)}")
print(f"Features tracked: {len(xpl.columns_dict)}")
print(f"Model type: {type(xpl.model)}")
