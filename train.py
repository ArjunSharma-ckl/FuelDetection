from ultralytics import YOLO
import torch
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
import shutil

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATA_YAML = "PH2026FuelDetection.v1i.yolov8/data.yaml"
BASE_MODEL = "yolov8s.pt"   # BEST balance for Limelight
IMG_SIZE = 640
BATCH = 32
EPOCHS = 60              # early stopping decides
PATIENCE = 45
WORKERS = 8
DEVICE = 0

EXPORT_ROOT = Path("exports")

# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    # ---------- GPU CHECK ----------
    assert torch.cuda.is_available(), "CUDA not available"
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", torch.cuda.get_device_properties(0).total_memory // 1024**3, "GB")

    # ---------- RUN NAME ----------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"robot_{timestamp}"

    # ---------- LOAD MODEL ----------
    model = YOLO(BASE_MODEL)

    # ---------- TRAIN (IMPROVED CONFIG) ----------
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,

        optimizer="AdamW",
        lr0=0.00035,
        cos_lr=True,
        patience=PATIENCE,
        cache="disk",

        # ---- FRC-OPTIMIZED AUGMENTATIONS ----
        hsv_h=0.01,
        hsv_s=0.45,
        hsv_v=0.35,

        degrees=2.0,
        translate=0.04,
        scale=0.25,

        fliplr=0.5,
        mosaic=0.2,
        close_mosaic=20,
        mixup=0.0,

        name=run_name,
        exist_ok=False
    )

    # ---------- LOCATE BEST WEIGHTS ----------
    run_dir = Path(results.save_dir)
    weights_dir = run_dir / "weights"
    best_pt = weights_dir / "best.pt"

    if not best_pt.exists():
        print("❌ best.pt not found — training failed")
        return

    # ---------- EXPORT FP16 TFLITE (SAFE PATH) ----------
    print("\nExporting FP16 TFLite (NO INT8)...")
    export_model = YOLO(best_pt)

    export_model.export(
        format="tflite",
        imgsz=IMG_SIZE,
        half=True   # FP16, fast, stable, Limelight-safe
    )

    # Find produced TFLite
    tflite_files = list(weights_dir.rglob("*.tflite"))
    if not tflite_files:
        print("❌ TFLite export failed")
        return

    best_tflite = tflite_files[0]

    # ---------- LIMELIGHT EXPORT STRUCTURE ----------
    EXPORT_ROOT.mkdir(exist_ok=True)

    archive_dir = EXPORT_ROOT / run_name
    archive_dir.mkdir(parents=True, exist_ok=True)

    latest_dir = EXPORT_ROOT / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.mkdir()

    # Copy outputs
    shutil.copy(best_pt, archive_dir / "best.pt")
    shutil.copy(best_tflite, archive_dir / "model.tflite")

    shutil.copy(best_pt, latest_dir / "best.pt")
    shutil.copy(best_tflite, latest_dir / "model.tflite")

    # Write labels file for Limelight
    labels_txt = latest_dir / "labels.txt"
    labels_txt.write_text("Robot\n")

    shutil.copy(labels_txt, archive_dir / "labels.txt")

    # ---------- DONE ----------
    print("\n✅ TRAINING COMPLETE")
    print("Run folder:", run_dir)
    print("Best model:", best_pt)
    print("TFLite model:", best_tflite)
    print("\n📤 Upload to Limelight from:")
    print(latest_dir)


# -------------------------------------------------
# WINDOWS MULTIPROCESSING GUARD
# -------------------------------------------------
if __name__ == "__main__":
    mp.freeze_support()
    main()
