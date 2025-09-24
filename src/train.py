import joblib
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_logistic_regression(X_train, y_train, X_test, y_test, models_dir):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression accuracy: {acc:.2f}")
    model_path = models_dir / "log_reg_model.pkl"
    joblib.dump(model, model_path)
    return model_path, acc

def train_custom_nn(X_train, y_train, X_test, y_test, models_dir, input_dim):
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    print(f"Custom Neural Network accuracy: {acc:.2f}")
    model_path = models_dir / "custom_nn_model.h5"
    model.save(model_path)
    return model_path, acc

def train_models(features_path: Path, models_dir: Path, input_dim: int):
    X_train, y_train, X_test, y_test = joblib.load(features_path)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Training Logistic Regression
    log_reg_path, log_reg_acc = train_logistic_regression(X_train, y_train, X_test, y_test, models_dir)
    
    # Training Custom Neural Network
    nn_path, nn_acc = train_custom_nn(X_train, y_train, X_test, y_test, models_dir, input_dim)
    
    return {
        "logistic_regression": {"model_path": log_reg_path, "accuracy": log_reg_acc},
        "custom_neural_network": {"model_path": nn_path, "accuracy": nn_acc}
    }

if __name__ == "__main__":
    features_file = Path("data/features/tfidf_data.pkl")
    models_dir = Path("models")
    input_dimension = 5000  
    
    results = train_models(features_file, models_dir, input_dimension)
    print("Training complete. Model accuracies:")
    print(results) explain every line of custom neural network in detail 
