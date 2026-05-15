import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from shapash import SmartExplainer
import os

# Create dummy multi-target data
X = pd.DataFrame(np.random.rand(100, 5), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
Y = pd.DataFrame(np.random.rand(100, 3), columns=['t1', 't2', 't3'])

# Train ONE multi-target model
model = RandomForestRegressor(n_estimators=10)
model.fit(X, Y)

# Initialize Shapash
try:
    xpl = SmartExplainer(model=model)
    xpl.compile(x=X, y_target=Y)
    print("Compilation successful for multi-output regressor")
    print("Features importance type:", type(xpl.features_imp))
except Exception as e:
    print("Compilation failed:", e)
