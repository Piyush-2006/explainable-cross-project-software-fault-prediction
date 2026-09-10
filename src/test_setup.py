print("HELLO - PROJECT TEST")

import pandas
import numpy
import sklearn

print("Project environment is working!")
print("Pandas:", pandas.__version__)
print("NumPy:", numpy.__version__)
print("Scikit-learn:", sklearn.__version__)

try:
    import xgboost
    print("XGBoost:", xgboost.__version__)
except ImportError:
    print("XGBoost: NOT INSTALLED")

try:
    import shap
    print("SHAP:", shap.__version__)
except ImportError:
    print("SHAP: NOT INSTALLED")