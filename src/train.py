import joblib
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

def train_custom_nn(features_path: Path, models_dir: Path, input_dim: int):
    
    X_train, y_train, X_test, y_test = joblib.load(features_path)
    

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
    print(f"Training complete. Accuracy = {acc:.2f}")
    
    
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "custom_nn_model.h5"
    model.save(model_path)
    
    return model_path, acc

if __name__ == "__main__":
    features_file = Path("data/features/tfidf_data.pkl")
    models_dir = Path("models")
    
    
    input_dimension = 5000  
    
    train_custom_nn(features_file, models_dir, input_dimension)
