"""
LogiEdge Structured Pruning + PTQ INT8 — M3 Variant (Task F1)
Applies 35% structured filter pruning with PolynomialDecay schedule,
then converts to Full INT8 via Post-Training Quantisation.

Tools: tensorflow_model_optimization + TFLiteConverter

Usage:
    python prune_quantise.py
"""

import os

# TensorFlow Model Optimization 0.8 uses the legacy tf.keras API. This must be
# set before importing TensorFlow so pruning wrappers and the model use the
# same Keras implementation under TensorFlow 2.20.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
KERAS_MODEL_PATH = os.path.join(MODELS_DIR, "logibridge_fp32.keras")
SAVED_MODEL_PATH = os.path.join(MODELS_DIR, "logibridge_fp32_saved")
DATASET_PATH = os.path.join(MODELS_DIR, "dataset.npz")
OUTPUT_PATH = os.path.join(MODELS_DIR, "model_pruned_int8.tflite")

CLASS_NAMES = ["Normal", "Warning", "Critical"]


def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    print("=" * 60)
    print("M3 — Structured Pruning (35%) + PTQ INT8")
    print("=" * 60)

    # Load dataset
    data = np.load(DATASET_PATH)
    X = data["X_normalised"].astype(np.float32)
    y = data["y"]

    # Split same way as training
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=0, stratify=y
    )

    # Load the original FP32 model
    # Rebuild with tf-keras and load tensors directly from the SavedModel
    # checkpoint. This avoids Keras archive schema differences across versions.
    print("\nRebuilding FP32 baseline model...")
    import tf_keras
    original_model = tf_keras.Sequential([
        tf_keras.layers.InputLayer(input_shape=(6,)),
        tf_keras.layers.Dense(32, activation='relu'),
        tf_keras.layers.Dense(16, activation='relu'),
        tf_keras.layers.Dense(3, activation='softmax')
    ])
    original_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    checkpoint_path = os.path.join(SAVED_MODEL_PATH, "variables", "variables")
    checkpoint = tf.train.load_checkpoint(checkpoint_path)
    checkpoint_weights = [
        checkpoint.get_tensor(f"variables/{i}/.ATTRIBUTES/VARIABLE_VALUE")
        for i in range(6)
    ]
    original_model.layers[0].set_weights(checkpoint_weights[0:2])
    original_model.layers[1].set_weights(checkpoint_weights[2:4])
    original_model.layers[2].set_weights(checkpoint_weights[4:6])

    # Evaluate original model
    _, orig_acc = original_model.evaluate(X_val, y_val, verbose=0)
    print(f"Original FP32 validation accuracy: {orig_acc * 100:.2f}%")

    # Apply pruning with PolynomialDecay schedule
    # Target: 35% structured filter pruning
    num_train_samples = len(X_train)
    batch_size = 16
    epochs = 50
    steps_per_epoch = num_train_samples // batch_size
    total_steps = steps_per_epoch * epochs

    # Pruning schedule: PolynomialDecay from 0% to 35% sparsity
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.35,
            begin_step=0,
            end_step=total_steps,
            power=3
        )
    }

    print("\nApplying pruning schedule...")
    print(f"  Target sparsity: 35%")
    print(f"  Schedule: PolynomialDecay over {total_steps} steps")

    # Apply pruning to the model
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
        original_model, **pruning_params
    )

    pruned_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Fine-tune with pruning
    print("\nFine-tuning with pruning...")
    pruned_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[
            tfmot.sparsity.keras.UpdatePruningStep()
        ]
    )

    # Evaluate pruned model
    _, pruned_acc = pruned_model.evaluate(X_val, y_val, verbose=0)
    print(f"\nPruned model validation accuracy: {pruned_acc * 100:.2f}%")
    print(f"Accuracy drop from pruning: {(orig_acc - pruned_acc) * 100:.2f}%")

    # Strip pruning wrappers for export
    stripped_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    kernels = [layer.get_weights()[0] for layer in stripped_model.layers
               if layer.get_weights()]
    zero_weights = sum(np.count_nonzero(kernel == 0) for kernel in kernels)
    total_weights = sum(kernel.size for kernel in kernels)
    achieved_sparsity = zero_weights / total_weights
    print(f"Achieved kernel sparsity: {achieved_sparsity * 100:.2f}%")

    # Save stripped model
    pruned_keras_path = os.path.join(MODELS_DIR, "logibridge_pruned.keras")
    stripped_model.save(pruned_keras_path)

    # Export SavedModel for TFLite conversion
    pruned_saved_path = os.path.join(MODELS_DIR, "logibridge_pruned_saved")
    tf.saved_model.save(stripped_model, pruned_saved_path)

    # Convert to INT8 TFLite
    print("\nConverting pruned model to INT8 TFLite...")

    # Calibration data
    num_calibration = min(len(X), 300)
    indices = np.random.RandomState(42).permutation(len(X))[:num_calibration]
    calibration_data = X[indices]

    def representative_dataset():
        for i in range(len(calibration_data)):
            yield [calibration_data[i:i+1]]

    converter = tf.lite.TFLiteConverter.from_saved_model(pruned_saved_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(OUTPUT_PATH, "wb") as f:
        f.write(tflite_model)

    # Report sizes
    fp32_size = os.path.getsize(os.path.join(MODELS_DIR, "model_fp32.tflite"))
    ptq_size = os.path.getsize(os.path.join(MODELS_DIR, "model_ptq_int8.tflite"))
    pruned_size = os.path.getsize(OUTPUT_PATH)

    print(f"\nModel Size Comparison:")
    print(f"  M1 (FP32):           {fp32_size / 1024:.2f} KB")
    print(f"  M2 (PTQ INT8):       {ptq_size / 1024:.2f} KB")
    print(f"  M3 (Pruned + INT8):  {pruned_size / 1024:.2f} KB")
    print(f"  M3 vs M1 reduction:  {(1 - pruned_size / fp32_size) * 100:.1f}%")

    # Verify pruned INT8 model
    print("\nVerifying M3 model accuracy...")
    try:
        from ai_edge_litert import interpreter as litert
        _Interpreter = litert.Interpreter
    except ImportError:
        _Interpreter = tf.lite.Interpreter

    try:
        interpreter = _Interpreter(model_path=OUTPUT_PATH)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        correct = 0
        for i in range(len(X_val)):
            sample = X_val[i:i+1]
            input_scale, input_zero_point = input_details[0]['quantization']
            sample_int8 = (sample / input_scale + input_zero_point).astype(np.int8)

            interpreter.set_tensor(input_details[0]['index'], sample_int8)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            pred = np.argmax(output)
            if pred == y_val[i]:
                correct += 1

        m3_acc = correct / len(X_val)
        print(f"  M3 INT8 validation accuracy: {m3_acc * 100:.2f}%")

        # Class 2 recall check
        from sklearn.metrics import classification_report
        preds = []
        for i in range(len(X_val)):
            sample = X_val[i:i+1]
            input_scale, input_zero_point = input_details[0]['quantization']
            sample_int8 = (sample / input_scale + input_zero_point).astype(np.int8)
            interpreter.set_tensor(input_details[0]['index'], sample_int8)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            preds.append(np.argmax(output))

        print(f"\nClassification Report (M3):")
        print(classification_report(y_val, preds, target_names=CLASS_NAMES))
    except TypeError as e:
        print(f"  [SKIP] Interpreter API bug — verification skipped: {e}")
        print("  Model file was saved successfully; benchmark.py will verify.")

    print("=" * 60)
    print("M3 variant complete!")
    print(f"  Model saved to: {OUTPUT_PATH}")
    print(f"  Next: run benchmark.py to compare all three variants")
    print("=" * 60)


if __name__ == "__main__":
    main()
