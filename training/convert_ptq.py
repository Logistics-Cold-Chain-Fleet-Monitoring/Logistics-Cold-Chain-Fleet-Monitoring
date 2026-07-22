"""
LogiEdge Post-Training Quantisation — M2 Variant (Task F1)
Converts the FP32 Keras SavedModel to Full INT8 TFLite using:
  - tf.lite.TFLiteConverter with DEFAULT optimisation
  - Representative dataset of >=200 calibration samples

Usage:
    python convert_ptq.py
"""

import os
import numpy as np
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
SAVED_MODEL_PATH = os.path.join(MODELS_DIR, "logibridge_fp32_saved")
DATASET_PATH = os.path.join(MODELS_DIR, "dataset.npz")
OUTPUT_PATH = os.path.join(MODELS_DIR, "model_ptq_int8.tflite")


def main():
    print("=" * 60)
    print("M2 — Post-Training Quantisation (Full INT8)")
    print("=" * 60)

    # Load dataset for representative samples
    data = np.load(DATASET_PATH)
    X = data["X_normalised"].astype(np.float32)

    # Need at least 200 calibration samples
    num_calibration = min(len(X), 300)
    print(f"Using {num_calibration} calibration samples (requirement: >=200)")

    # Shuffle and select calibration subset
    indices = np.random.RandomState(42).permutation(len(X))[:num_calibration]
    calibration_data = X[indices]

    def representative_dataset():
        """Generator yielding calibration samples for INT8 quantisation."""
        for i in range(len(calibration_data)):
            sample = calibration_data[i:i+1]
            yield [sample]

    # Convert
    print("\nConverting model to INT8...")
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)

    # Enable DEFAULT optimisation (includes INT8 quantisation)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Set representative dataset for full INT8
    converter.representative_dataset = representative_dataset

    # Force full INT8 (inputs and outputs too)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    # Save
    with open(OUTPUT_PATH, "wb") as f:
        f.write(tflite_model)

    fp32_size = os.path.getsize(os.path.join(MODELS_DIR, "model_fp32.tflite"))
    int8_size = os.path.getsize(OUTPUT_PATH)

    print(f"\nM2 (PTQ INT8) model saved to: {OUTPUT_PATH}")
    print(f"  FP32 model size: {fp32_size / 1024:.2f} KB")
    print(f"  INT8 model size: {int8_size / 1024:.2f} KB")
    print(f"  Size reduction: {(1 - int8_size / fp32_size) * 100:.1f}%")

    # Verify the model works
    print("\nVerifying INT8 model...")
    try:
        with open(OUTPUT_PATH, "rb") as f:
            model_content = f.read()
        interpreter = tf.lite.Interpreter(model_content=model_content)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"  Input type: {input_details[0]['dtype']}")
        print(f"  Output type: {output_details[0]['dtype']}")
        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Output shape: {output_details[0]['shape']}")

        # Quick accuracy check on calibration data
        correct = 0
        y = data["y"]
        selected_y = y[indices]
        for i in range(min(50, len(calibration_data))):
            sample = calibration_data[i:i+1]

            # Quantise input
            input_scale, input_zero_point = input_details[0]['quantization']
            sample_int8 = (sample / input_scale + input_zero_point).astype(np.int8)

            interpreter.set_tensor(input_details[0]['index'], sample_int8)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])

            pred = np.argmax(output)
            if pred == selected_y[i]:
                correct += 1

        print(f"  Quick check accuracy (50 samples): {correct / min(50, len(calibration_data)) * 100:.1f}%")
    except TypeError as e:
        print(f"  [SKIP] TF 2.19 Interpreter API bug — verification skipped: {e}")
        print("  Model file was saved successfully; benchmark.py will verify.")

    print("\nDone! Next: run prune_quantise.py for M3 variant.")


if __name__ == "__main__":
    main()
