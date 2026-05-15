import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from shapash import SmartExplainer
import os
import shap

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

# Prepare y_pred as the top class (required for classification dashboard)
# Even though these are independent probabilities, we pick one to satisfy the schema
top_class = Y.idxmax(axis=1)
y_pred = pd.DataFrame(top_class, columns=['pred'], index=X.index)

# Initialize Shapash
try:
    xpl = SmartExplainer(model=model)
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    list_contrib_raw = explainer.shap_values(X)
    
    # SHAP for multi-output regression returns (outputs, samples, features)
    # We need to convert it to a list of DataFrames
    list_contrib = [pd.DataFrame(c, columns=X.columns, index=X.index) for c in list_contrib_raw]
    
    print(f"Number of contribution matrices: {len(list_contrib)}")
    
    # Pass y_pred explicitly to avoid the predict() call in compile
    xpl.compile(x=X, contributions=list_contrib, y_pred=y_pred, y_target=y_pred) # use same for target for now
    
    print("Compilation successful!")
    print("Case detected:", xpl._case)
    print("Classes detected:", xpl._classes)
    
    xpl.compute_features_import()
    print("Importance computed.")
    
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Compilation failed:", e)
