
import joblib
import pandas as pd

def test_model_prediction():
    model = joblib.load("model.joblib")
    sample = pd.DataFrame([{"age":50,"chol":240,"thalach":155}])
    pred = model.predict(sample)
    assert pred[0] in [0,1]
