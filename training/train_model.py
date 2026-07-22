"""
LogiEdge Model Training (Task D1)
Trains a classification model:
  - 2-hidden-layer MLP with 32 and 16 units, ReLU activation
  - Input: 6-value normalised feature vector
  - Output: 3 classes (Normal, Warning, Critical)
  - Validation accuracy must exceed 88%

Saves:
  - Keras SavedModel (M1 — FP32 Baseline)
  - TFLite FP32 model
  - Training history plot

Usage:
    python train_model.py
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
DATASET_PATH = os.path.join(MODELS_DIR, "dataset.npz")

# Class labels
CLASS_NAMES = ["Normal", "Warning", "Critical"]


def build_model(input_dim=6, num_classes=3):
    """Build a 2-hidden-layer MLP as specified in the assignment."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu', name='hidden1'),
        tf.keras.layers.Dense(16, activation='relu', name='hidden2'),
        tf.keras.layers.Dense(num_classes, activation='softmax', name='output')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Set seeds for reproducibility
    SEED = 0
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Load dataset
    print("Loading dataset...")
    data = np.load(DATASET_PATH)
    X = data["X_normalised"]
    y = data["y"]
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    # Train/validation split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")

    # Build model
    model = build_model(input_dim=X.shape[1])
    model.summary()

    # Compute class weights to boost Critical (Class 2) recall
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    # Further boost Class 2 (Critical) for safety-critical recall
    weights[2] *= 3.0
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"\nClass weights: {class_weight}")

    # Train
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=16,
        class_weight=class_weight,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=15,
                restore_best_weights=True, min_delta=0.001
            )
        ]
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation Accuracy: {val_acc * 100:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")

    if val_acc < 0.88:
        print("\n[WARNING] Validation accuracy is below 88%!")
        print("The assignment requires >88%. Consider adjusting features or architecture.")
    else:
        print(f"\n[OK] Validation accuracy {val_acc*100:.2f}% exceeds the 88% threshold.")

    # Detailed classification report
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=CLASS_NAMES))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_val, y_pred)
    print(cm)

    # Check Class 2 (Critical) recall — must exceed 95%
    if cm.shape[0] > 2:
        critical_recall = cm[2, 2] / cm[2].sum() if cm[2].sum() > 0 else 0
        print(f"\nClass 2 (Critical) Recall: {critical_recall * 100:.2f}%")
        if critical_recall < 0.95:
            print("[WARNING] Critical recall below 95% — required for deployment recommendation")
        else:
            print("[OK] Critical recall exceeds 95%")

    # Save Keras model (M1 — FP32 Baseline)
    keras_model_path = os.path.join(MODELS_DIR, "logibridge_fp32.keras")
    model.save(keras_model_path)
    print(f"\nSaved Keras model to {keras_model_path}")

    # Export SavedModel for TFLite conversion
    saved_model_path = os.path.join(MODELS_DIR, "logibridge_fp32_saved")
    model.export(saved_model_path)
    print(f"Exported SavedModel to {saved_model_path}")

    # Convert to TFLite FP32 (M1 baseline)
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    tflite_model = converter.convert()

    tflite_fp32_path = os.path.join(MODELS_DIR, "model_fp32.tflite")
    with open(tflite_fp32_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved TFLite FP32 model to {tflite_fp32_path}")
    print(f"  File size: {os.path.getsize(tflite_fp32_path) / 1024:.2f} KB")

    # Also copy to inference directory
    inference_model_path = os.path.join(SCRIPT_DIR, "..", "inference", "model.tflite")
    with open(inference_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"Copied model to {inference_model_path}")

    # Save validation data for later benchmarking
    np.savez(
        os.path.join(MODELS_DIR, "val_data.npz"),
        X_val=X_val, y_val=y_val
    )
    print("Saved validation data for benchmarking.")

    # Save training history
    np.save(
        os.path.join(MODELS_DIR, "training_history.npy"),
        {
            "accuracy": history.history["accuracy"],
            "val_accuracy": history.history["val_accuracy"],
            "loss": history.history["loss"],
            "val_loss": history.history["val_loss"]
        }
    )
    print("Saved training history.")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  M1 (FP32 Baseline) saved to: {tflite_fp32_path}")
    print(f"  Next: run convert_ptq.py for M2 and prune_quantise.py for M3")
    print("=" * 60)


if __name__ == "__main__":
    main()
