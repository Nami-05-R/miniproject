import sys

from PIL import Image
from ultralytics import YOLO

from road_safety_core import (
    analyze_streetlights,
    detect_potholes,
    get_available_model_path,
    get_available_streetlight_model_path,
)


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "real_test.jpg"
    model_path = get_available_model_path()

    if not model_path:
        raise FileNotFoundError("No trained YOLO pothole model was found.")

    model = YOLO(model_path)
    streetlight_model_path = get_available_streetlight_model_path()
    streetlight_model = YOLO(streetlight_model_path) if streetlight_model_path else None
    image = Image.open(image_path).convert("RGB")

    pothole_result = detect_potholes(model, image, conf_threshold=0.25, iou_threshold=0.50)
    streetlight_result = analyze_streetlights(image, streetlight_model=streetlight_model)

    print(f"Image: {image_path}")
    print(f"Model: {model_path}")
    print(
        "Streetlight engine: "
        + (
            streetlight_model_path
            if streetlight_model_path
            else "heuristic fallback"
        )
    )
    print(f"Total potholes: {pothole_result['total_potholes']}")
    print(f"Road condition: {pothole_result['road_condition']}")
    print(f"Average confidence: {pothole_result['avg_confidence']:.2f}")
    print(f"Day phase: {streetlight_result['day_phase']}")
    print(f"Streetlight status: {streetlight_result['streetlight_status']}")
    if "model_confidence" in streetlight_result:
        print(f"Streetlight model confidence: {streetlight_result['model_confidence']:.3f}")
    print("-" * 40)

    if not pothole_result["detections"]:
        print("No potholes were detected.")
    else:
        for index, detection in enumerate(pothole_result["detections"], start=1):
            print(
                f"{index}. Severity={detection['severity']}, "
                f"Confidence={detection['confidence']:.2f}, "
                f"Risk={detection['risk']}, Box={detection['box']}"
            )


if __name__ == "__main__":
    main()
