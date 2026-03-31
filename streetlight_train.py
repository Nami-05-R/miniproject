import os
import zipfile

from ultralytics import YOLO


DATASET_ZIP = "Street Light.v1i.yolov8.zip"
EXTRACT_ROOT = "streetlight_dataset"
DEFAULT_OUTPUT_NAME = "streetlight_det"


def ensure_extracted_dataset(zip_path, extract_root):
    train_images = os.path.join(extract_root, "train", "images")
    valid_images = os.path.join(extract_root, "valid", "images")
    data_yaml = os.path.join(extract_root, "data.yaml")

    if os.path.isdir(train_images) and os.path.isdir(valid_images) and os.path.isfile(data_yaml):
        return os.path.abspath(data_yaml)

    if not os.path.isfile(zip_path):
        raise FileNotFoundError(
            "Streetlight dataset not found. "
            "Place the detection export zip in the project folder as "
            f"'{DATASET_ZIP}' or extract it into '{EXTRACT_ROOT}'."
        )

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_root)

    if not os.path.isfile(data_yaml):
        raise FileNotFoundError("Extraction completed but data.yaml was not found in the dataset folder.")

    return os.path.abspath(data_yaml)


def main():
    dataset_yaml = ensure_extracted_dataset(DATASET_ZIP, EXTRACT_ROOT)

    model = YOLO("yolov8n.pt")
    model.train(
        data=dataset_yaml,
        epochs=30,
        imgsz=640,
        batch=8,
        project="runs/detect",
        name=DEFAULT_OUTPUT_NAME,
    )

    print("Streetlight detector training completed.")
    print("Expected weights: runs/detect/streetlight_det/weights/best.pt")


if __name__ == "__main__":
    main()
