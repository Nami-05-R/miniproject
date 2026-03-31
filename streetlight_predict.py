import sys

from PIL import Image
from ultralytics import YOLO

from road_safety_core import (
    analyze_streetlights,
    get_available_streetlight_model_path,
)


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "real_test.jpg"
    model_path = get_available_streetlight_model_path()

    if not model_path:
        raise FileNotFoundError(
            "No trained streetlight model was found. "
            "Train one with streetlight_train.py or place weights at "
            "runs/detect/streetlight_det/weights/best.pt."
        )

    streetlight_model = YOLO(model_path)
    image = Image.open(image_path).convert("RGB")
    result = analyze_streetlights(image, streetlight_model=streetlight_model)

    print(f"Image: {image_path}")
    print(f"Streetlight model: {model_path}")
    print(f"Day phase: {result['day_phase']}")
    print(f"Streetlight status: {result['streetlight_status']}")
    print(f"Analysis source: {result.get('analysis_source', 'heuristic')}")
    if "model_confidence" in result:
        print(f"Model confidence: {result['model_confidence']:.3f}")


if __name__ == "__main__":
    main()
