from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_and_save_model():
    os.makedirs('app', exist_ok=True)  # Create app directory for the model
    iris = load_iris()
    print(iris)
    X, y = iris.data, iris.target
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, 'app/iris_model.pkl')
    print("Model trained and saved to app/iris_model.pkl")

if __name__ == "__main__":
    train_and_save_model()