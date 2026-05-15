import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from shapash import SmartExplainer
import os

# Create dummy data
X = pd.DataFrame(np.random.rand(100, 5), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
# 3 targets (sounds)
Y = pd.DataFrame(np.random.rand(100, 3), columns=['Sound1', 'Sound2', 'Sound3'])

# Train ONE multi-target model
model = RandomForestRegressor(n_estimators=10)
model.fit(X, Y)

# TRICK: Monkeypatch to look like a classifier
model.classes_ = Y.columns.tolist()
model.predict_proba = model.predict

# Initialize Shapash
try:
    xpl = SmartExplainer(model=model)
    # Important: list of contributions
    import shap
    explainer = shap.TreeExplainer(model)
    list_contrib = explainer.shap_values(X)
    
    # Check shape of list_contrib
    print(f"Number of contribution matrices: {len(list_contrib)}")
    
    xpl.compile(x=X, contributions=list_contrib, y_target=Y)
    print("Compilation successful!")
    print("Case detected:", xpl._case)
    print("Classes detected:", xpl._classes)
    
    # Check if we can get importance for a specific "class" (target)
    xpl.compute_features_import()
    print("Importance computed.")
    
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Compilation failed:", e)
