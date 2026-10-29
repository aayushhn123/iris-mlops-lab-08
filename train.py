import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
acc = accuracy_score(y_te, model.predict(X_te))
joblib.dump(model, "model.joblib")
print(f"Test accuracy: {acc:.4f}")
