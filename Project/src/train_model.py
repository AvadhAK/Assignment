import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def train_model():
    # Load the dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split\
        (X, y, test_size=0.2, random_state=42)

    # Train a Decision Tree classifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Save the model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved as model.pkl")


if __name__ == "__main__":
    train_model()
