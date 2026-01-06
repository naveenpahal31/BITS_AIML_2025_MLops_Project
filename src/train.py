
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

df = pd.DataFrame({
    "age":[52,45,60,35,50],
    "chol":[250,230,270,180,240],
    "thalach":[160,150,140,170,155],
    "target":[1,0,1,0,1]
})

X = df.drop("target", axis=1)
y = df["target"]

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

pipeline.fit(X, y)
joblib.dump(pipeline, "model.joblib")
print("Model trained & saved")
