from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import shutil
import onnx
import onnx2tf

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
INPUT_MODEL = "best.pt"
IMG_SIZE = 640
EXPORT_ROOT = Path("exports")

# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    model_path = Path(INPUT_MODEL)
    if not model_path.exists():
        print(f"ERROR: {model_path} not found.")
        return

    # ---------- RUN NAME ----------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"export_{timestamp}"
    
    # Setup directories
    EXPORT_ROOT.mkdir(exist_ok=True)
    archive_dir = EXPORT_ROOT / run_name
    archive_dir.mkdir(parents=True, exist_ok=True)
    latest_dir = EXPORT_ROOT / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.mkdir()

    # ---------- STEP 1: EXPORT TO ONNX ----------
    print(f"\nSTEP 1: Exporting {model_path} to ONNX...")
    model = YOLO(model_path)
    
    # Export to ONNX first (opset 12 is safest for TFLite)
    onnx_file = model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        opset=12,
        simplify=True
    )
    
    onnx_path = Path(onnx_file)
    print(f"ONNX export successful: {onnx_path}")

    # ---------- STEP 2: CONVERT ONNX TO TFLITE (DIRECT) ----------
    print(f"\nSTEP 2: Converting ONNX to TFLite via library...")
    
    tflite_out_dir = model_path.parent / "tflite_out"
    
    try:
        # Minimal arguments to avoid version conflicts
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(tflite_out_dir),
            batch_size=1,                     # FIXES THE RANK ERROR
            output_integer_quantized_tflite=False
        )
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return

    # Find the resulting file
    generated_tflite = list(tflite_out_dir.glob("*.tflite"))[0]

    # ---------- GENERATE LABELS ----------
    names = model.names
    sorted_names = [names[i] for i in sorted(names.keys())]
    labels_content = "\n".join(sorted_names)

    labels_file = latest_dir / "labels.txt"
    labels_file.write_text(labels_content, encoding="utf-8")
    shutil.copy(labels_file, archive_dir / "labels.txt")

    # ---------- ORGANIZE FILES ----------
    print(f"\nOrganizing files...")

    # Copy to Archive
    shutil.copy(model_path, archive_dir / "best.pt")
    shutil.copy(generated_tflite, archive_dir / "model.tflite")

    # Copy to Latest
    shutil.copy(model_path, latest_dir / "best.pt")
    shutil.copy(generated_tflite, latest_dir / "model.tflite")

    # ---------- CLEANUP ----------
    # Clean up the intermediate folder
    try:
        shutil.rmtree(tflite_out_dir)
        # onnx_path.unlink() # Delete ONNX file if you want
    except Exception as e:
        print(f"Warning: Cleanup failed ({e})")

    # ---------- DONE ----------
    print("\nEXPORT COMPLETE")
    print("Files ready for Limelight:")
    print(f"1. {latest_dir / 'model.tflite'}")
    print(f"2. {latest_dir / 'labels.txt'}")

if __name__ == "__main__":
    main()