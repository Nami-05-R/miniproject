import io
import os
import smtplib
import sqlite3
import tempfile
from datetime import datetime
from email.message import EmailMessage

import cv2
import folium
import numpy as np
import pandas as pd
from PIL.ExifTags import GPSTAGS, TAGS
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _get_default_db_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "road_safety_monitoring.sqlite3")


DB_PATH = _get_default_db_path()
REPAIR_STATUSES = ["Open", "Assigned", "In Progress", "Resolved"]
MODEL_CANDIDATES = (
    os.path.join("models", "pothole_best.pt"),
    os.path.join("runs", "detect", "train2", "weights", "best.pt"),
    os.path.join("runs", "detect", "train", "weights", "best.pt"),
    "pothole_model_final.pt",
)
STREETLIGHT_MODEL_CANDIDATES = (
    os.path.join("models", "streetlight_best.pt"),
    os.path.join("runs", "detect", "runs", "detect", "streetlight_det", "weights", "best.pt"),
    os.path.join("runs", "detect", "streetlight_det", "weights", "best.pt"),
    os.path.join("runs", "detect", "train", "weights", "best.pt"),
    os.path.join("runs", "classify", "streetlight_cls", "weights", "best.pt"),
    os.path.join("runs", "classify", "train", "weights", "best.pt"),
    "streetlight_classifier.pt",
)


def get_available_model_path():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def get_available_streetlight_model_path():
    for path in STREETLIGHT_MODEL_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def normalize_streetlight_label(label):
    label_text = str(label).strip().upper()
    if "DIM" in label_text:
        return "DIM"
    if "OFF" in label_text:
        return "OFF"
    if "ON" in label_text:
        return "ON"
    return label_text


def score_from_streetlight_label(label):
    return {"OFF": 1, "DIM": 2, "ON": 3}.get(str(label).upper(), 2)


def classify_streetlight_with_model(streetlight_model, image):
    if streetlight_model is None:
        return None

    results = streetlight_model.predict(source=np.array(image.convert("RGB")), verbose=False)
    if not results:
        return None

    result = results[0]
    probs = getattr(result, "probs", None)
    if probs is None:
        return None

    names = getattr(result, "names", {}) or {}
    top1_index = int(getattr(probs, "top1", -1))
    if top1_index < 0:
        return None

    raw_label = names.get(top1_index, str(top1_index))
    label = normalize_streetlight_label(raw_label)
    confidence = float(getattr(probs, "top1conf", 0.0))
    return {
        "streetlight_status": label,
        "streetlight_score": score_from_streetlight_label(label),
        "model_confidence": round(confidence, 3),
        "analysis_source": "trained_classifier",
    }


def infer_status_from_region(region_rgb):
    if region_rgb.size == 0:
        return "OFF", 1, 0.0

    hsv = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2GRAY)
    neutral_mask = cv2.inRange(hsv, np.array([0, 0, 190]), np.array([180, 120, 255]))
    warm_mask = cv2.inRange(hsv, np.array([5, 80, 160]), np.array([40, 255, 255]))
    bright_mask = cv2.bitwise_or(neutral_mask, warm_mask)
    bright_ratio = float(np.count_nonzero(bright_mask)) / max(bright_mask.size, 1)
    peak_intensity = int(gray.max())
    mean_intensity = float(gray.mean())

    if bright_ratio >= 0.03 or peak_intensity >= 245 or mean_intensity >= 150:
        return "ON", 3, bright_ratio
    if bright_ratio >= 0.008 or peak_intensity >= 215 or mean_intensity >= 95:
        return "DIM", 2, bright_ratio
    return "OFF", 1, bright_ratio


def detect_streetlights_with_model(streetlight_model, image):
    if streetlight_model is None:
        return None

    if getattr(streetlight_model, "task", None) != "detect":
        return None

    rgb_image = image.convert("RGB")
    rgb_array = np.array(rgb_image)
    results = streetlight_model.predict(source=rgb_array, verbose=False)
    if not results:
        return None

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return {
            "streetlight_status": "OFF",
            "streetlight_score": 1,
            "analysis_source": "trained_detector",
            "detected_regions": 0,
            "detector_confidence": 0.0,
            "annotated_image": rgb_array.copy(),
        }

    annotated = rgb_array.copy()
    region_statuses = []
    region_scores = []
    detector_confidences = []

    for box in boxes:
        confidence = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, rgb_array.shape[1])
        y2 = min(y2, rgb_array.shape[0])
        region = rgb_array[y1:y2, x1:x2]
        status, score, bright_ratio = infer_status_from_region(region)
        region_statuses.append(status)
        region_scores.append(score)
        detector_confidences.append(confidence)

        color = {
            "ON": (255, 215, 0),
            "DIM": (255, 140, 0),
            "OFF": (220, 60, 60),
        }.get(status, (255, 255, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{status} {confidence:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    if "ON" in region_statuses:
        overall_status = "ON"
    elif "DIM" in region_statuses:
        overall_status = "DIM"
    else:
        overall_status = "OFF"

    return {
        "streetlight_status": overall_status,
        "streetlight_score": score_from_streetlight_label(overall_status),
        "analysis_source": "trained_detector",
        "detected_regions": len(region_statuses),
        "detector_confidence": round(float(np.mean(detector_confidences)), 3),
        "annotated_image": annotated,
    }


def init_db(db_path=DB_PATH):
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA journal_mode=MEMORY")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                image_name TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                gps_source TEXT,
                day_phase TEXT,
                processing_module TEXT,
                priority_score INTEGER,
                priority_label TEXT,
                repair_status TEXT,
                streetlight_status TEXT,
                streetlight_score INTEGER,
                ambient_brightness REAL,
                road_health_score INTEGER,
                pothole_count INTEGER,
                small_count INTEGER,
                medium_count INTEGER,
                large_count INTEGER,
                avg_confidence REAL,
                risk_score INTEGER,
                road_condition TEXT,
                notes TEXT
            )
            """
        )
        existing_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(inspections)").fetchall()
        }
        if "processing_module" not in existing_columns:
            connection.execute(
                "ALTER TABLE inspections ADD COLUMN processing_module TEXT"
            )
        if "priority_score" not in existing_columns:
            connection.execute("ALTER TABLE inspections ADD COLUMN priority_score INTEGER")
        if "priority_label" not in existing_columns:
            connection.execute("ALTER TABLE inspections ADD COLUMN priority_label TEXT")
        if "repair_status" not in existing_columns:
            connection.execute("ALTER TABLE inspections ADD COLUMN repair_status TEXT")
        if "road_health_score" not in existing_columns:
            connection.execute("ALTER TABLE inspections ADD COLUMN road_health_score INTEGER")


def clear_history(db_path=DB_PATH):
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA journal_mode=MEMORY")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("DELETE FROM inspections")


def insert_inspection(record, db_path=DB_PATH):
    columns = [
        "created_at",
        "image_name",
        "latitude",
        "longitude",
        "gps_source",
        "day_phase",
        "processing_module",
        "priority_score",
        "priority_label",
        "repair_status",
        "streetlight_status",
        "streetlight_score",
        "ambient_brightness",
        "road_health_score",
        "pothole_count",
        "small_count",
        "medium_count",
        "large_count",
        "avg_confidence",
        "risk_score",
        "road_condition",
        "notes",
    ]
    values = [record.get(column) for column in columns]
    placeholders = ", ".join(["?"] * len(columns))

    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA journal_mode=MEMORY")
        connection.execute("PRAGMA temp_store=MEMORY")
        cursor = connection.execute(
            f"""
            INSERT INTO inspections ({", ".join(columns)})
            VALUES ({placeholders})
            """,
            values,
        )
        return cursor.lastrowid


def fetch_history_dataframe(db_path=DB_PATH):
    init_db(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA journal_mode=MEMORY")
        connection.execute("PRAGMA temp_store=MEMORY")
        dataframe = pd.read_sql_query(
            "SELECT * FROM inspections ORDER BY datetime(created_at) DESC, id DESC",
            connection,
        )
    return dataframe


def update_repair_status(record_id, repair_status, db_path=DB_PATH):
    if repair_status not in REPAIR_STATUSES:
        raise ValueError(f"Unsupported repair status: {repair_status}")

    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA journal_mode=MEMORY")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute(
            "UPDATE inspections SET repair_status = ? WHERE id = ?",
            (repair_status, int(record_id)),
        )


def extract_gps_from_image(image):
    try:
        exif_data = image._getexif()
        if not exif_data:
            return None

        gps_info = {}
        for key, value in exif_data.items():
            if TAGS.get(key) != "GPSInfo":
                continue
            for gps_key, gps_value in value.items():
                gps_tag = GPSTAGS.get(gps_key)
                gps_info[gps_tag] = gps_value

        if "GPSLatitude" not in gps_info or "GPSLongitude" not in gps_info:
            return None

        def to_float(rational):
            if isinstance(rational, tuple):
                return rational[0] / rational[1]
            return float(rational)

        def convert_to_degrees(values):
            degrees = to_float(values[0])
            minutes = to_float(values[1])
            seconds = to_float(values[2])
            return degrees + (minutes / 60.0) + (seconds / 3600.0)

        latitude = convert_to_degrees(gps_info["GPSLatitude"])
        longitude = convert_to_degrees(gps_info["GPSLongitude"])

        if gps_info.get("GPSLatitudeRef") == "S":
            latitude *= -1
        if gps_info.get("GPSLongitudeRef") == "W":
            longitude *= -1

        return latitude, longitude
    except Exception:
        return None


def classify_pothole(area_ratio):
    if area_ratio < 0.05:
        return {
            "severity": "Small",
            "risk": "Low Risk",
            "severity_score": 1,
            "color": (0, 200, 0),
        }
    if area_ratio < 0.15:
        return {
            "severity": "Medium",
            "risk": "Moderate Risk",
            "severity_score": 2,
            "color": (0, 165, 255),
        }
    return {
        "severity": "Large",
        "risk": "High Risk",
        "severity_score": 3,
        "color": (0, 0, 255),
    }


def get_road_condition(risk_score):
    if risk_score == 0:
        return "No potholes detected"
    if risk_score < 5:
        return "Good"
    if risk_score < 10:
        return "Moderate Damage"
    return "Severe Damage"


def calculate_priority(risk_score, day_phase, streetlight_status, pothole_count):
    score = int(risk_score)
    if str(day_phase) == "Night":
        score += 2
    if str(streetlight_status) == "OFF":
        score += 2
    elif str(streetlight_status) == "DIM":
        score += 1
    if int(pothole_count) >= 3:
        score += 2

    if score >= 11:
        label = "Critical"
    elif score >= 8:
        label = "High"
    elif score >= 4:
        label = "Medium"
    else:
        label = "Low"
    return score, label


def calculate_road_health_score(risk_score, day_phase, streetlight_status):
    score = 100 - int(risk_score) * 7
    if str(day_phase) == "Night":
        score -= 5
    if str(streetlight_status) == "OFF":
        score -= 10
    elif str(streetlight_status) == "DIM":
        score -= 5
    return max(score, 0)


def detect_potholes(model, image, conf_threshold=0.5, iou_threshold=0.5):
    rgb_image = image.convert("RGB")
    rgb_array = np.array(rgb_image)
    annotated_bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    image_height, image_width = annotated_bgr.shape[:2]
    image_area = max(image_height * image_width, 1)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_path = temp_file.name
        rgb_image.save(temp_path, format="JPEG")
        results = model.predict(
            source=temp_path,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    detections = []
    severity_counts = {"Small": 0, "Medium": 0, "Large": 0}

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width = max(float(x2 - x1), 0.0)
            height = max(float(y2 - y1), 0.0)
            area_ratio = (width * height) / image_area
            confidence = float(box.conf[0])

            details = classify_pothole(area_ratio)
            severity_counts[details["severity"]] += 1
            detections.append(
                {
                    "severity": details["severity"],
                    "risk": details["risk"],
                    "severity_score": details["severity_score"],
                    "confidence": confidence,
                    "area_ratio": round(area_ratio, 4),
                    "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                }
            )

            color = details["color"]
            cv2.rectangle(
                annotated_bgr,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                2,
            )
            label = f"{details['severity']} ({confidence:.2f})"
            cv2.putText(
                annotated_bgr,
                label,
                (int(x1), max(int(y1) - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    risk_score = sum(item["severity_score"] for item in detections)
    average_confidence = (
        sum(item["confidence"] for item in detections) / len(detections)
        if detections
        else 0.0
    )

    return {
        "annotated_image": cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
        "detections": detections,
        "counts": severity_counts,
        "total_potholes": len(detections),
        "avg_confidence": round(average_confidence, 3),
        "risk_score": risk_score,
        "road_condition": get_road_condition(risk_score),
    }


def detect_day_phase(image):
    rgb_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    ambient_brightness = float(gray.mean())
    day_phase = "Day" if ambient_brightness >= 110 else "Night"
    return day_phase, ambient_brightness


def analyze_streetlights(image, streetlight_model=None):
    rgb_image = image.convert("RGB")
    rgb_array = np.array(rgb_image)
    day_phase, ambient_brightness = detect_day_phase(rgb_image)

    image_height = rgb_array.shape[0]
    upper_region = rgb_array[: max(int(image_height * 0.6), 1), :, :]
    hsv = cv2.cvtColor(upper_region, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(upper_region, cv2.COLOR_RGB2GRAY)

    neutral_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 190]),
        np.array([180, 120, 255]),
    )
    warm_mask = cv2.inRange(
        hsv,
        np.array([5, 80, 160]),
        np.array([40, 255, 255]),
    )
    bright_mask = cv2.bitwise_or(neutral_mask, warm_mask)
    bright_mask = cv2.medianBlur(bright_mask, 5)
    bright_mask = cv2.dilate(bright_mask, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(
        bright_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 12 <= area <= 4000:
            valid_contours.append(contour)

    hotspot_count = len(valid_contours)
    bright_ratio = float(np.count_nonzero(bright_mask)) / max(bright_mask.size, 1)
    peak_intensity = int(gray.max())

    if hotspot_count >= 3 or bright_ratio >= 0.01 or peak_intensity >= 245:
        status = "ON"
        score = 3
    elif hotspot_count >= 1 or bright_ratio >= 0.0025 or peak_intensity >= 215:
        status = "DIM"
        score = 2
    else:
        status = "OFF"
        score = 1

    annotated = rgb_array.copy()
    for contour in valid_contours:
        x, y, width, height = cv2.boundingRect(contour)
        cv2.rectangle(
            annotated,
            (x, y),
            (x + width, y + height),
            (255, 215, 0),
            2,
        )

    banner_text = f"{day_phase} | Streetlight: {status}"
    cv2.rectangle(
        annotated,
        (0, 0),
        (annotated.shape[1], 40),
        (25, 25, 25),
        -1,
    )
    cv2.putText(
        annotated,
        banner_text,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    result_payload = {
        "annotated_image": annotated,
        "day_phase": day_phase,
        "ambient_brightness": round(ambient_brightness, 2),
        "streetlight_status": status,
        "streetlight_score": score,
        "hotspot_count": hotspot_count,
        "bright_ratio": round(bright_ratio, 4),
        "analysis_source": "heuristic",
    }

    model_result = None
    if streetlight_model is not None:
        if getattr(streetlight_model, "task", None) == "classify":
            model_result = classify_streetlight_with_model(streetlight_model, rgb_image)
        elif getattr(streetlight_model, "task", None) == "detect":
            model_result = detect_streetlights_with_model(streetlight_model, rgb_image)

    if model_result:
        result_payload.update(model_result)
        annotated = model_result.get("annotated_image", annotated)
        banner_text = (
            f"{day_phase} | Streetlight: {result_payload['streetlight_status']} | "
            f"{result_payload['analysis_source'].replace('_', ' ').title()}"
        )
        cv2.rectangle(
            annotated,
            (0, 0),
            (annotated.shape[1], 40),
            (15, 55, 90),
            -1,
        )
        cv2.putText(
            annotated,
            banner_text,
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )
        result_payload["annotated_image"] = annotated

    return result_payload


def process_day_module(model, image, conf_threshold=0.5, iou_threshold=0.5, streetlight_model=None):
    pothole_result = detect_potholes(
        model,
        image,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
    streetlight_result = analyze_streetlights(image, streetlight_model=streetlight_model)
    streetlight_result["day_phase"] = "Day"
    return {
        "module_name": "Day Module",
        "pothole_result": pothole_result,
        "streetlight_result": streetlight_result,
    }


def process_night_module(model, image, conf_threshold=0.5, iou_threshold=0.5, streetlight_model=None):
    pothole_result = detect_potholes(
        model,
        image,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
    streetlight_result = analyze_streetlights(image, streetlight_model=streetlight_model)
    streetlight_result["day_phase"] = "Night"
    return {
        "module_name": "Night Module",
        "pothole_result": pothole_result,
        "streetlight_result": streetlight_result,
    }


def process_inspection_image(model, image, conf_threshold=0.5, iou_threshold=0.5, streetlight_model=None):
    detected_phase, _ = detect_day_phase(image)
    if detected_phase == "Night":
        return process_night_module(
            model,
            image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            streetlight_model=streetlight_model,
        )
    return process_day_module(
        model,
        image,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        streetlight_model=streetlight_model,
    )


def build_issue_map(records):
    if isinstance(records, pd.DataFrame):
        dataframe = records.copy()
    else:
        dataframe = pd.DataFrame(records)

    if dataframe.empty:
        return None

    dataframe = dataframe.dropna(subset=["latitude", "longitude"])
    if dataframe.empty:
        return None

    issue_map = folium.Map(
        location=[
            float(dataframe["latitude"].mean()),
            float(dataframe["longitude"].mean()),
        ],
        zoom_start=13,
        tiles="CartoDB positron",
    )

    for _, row in dataframe.iterrows():
        risk_score = int(row.get("risk_score", 0) or 0)
        streetlight_status = str(row.get("streetlight_status", "OFF"))
        day_phase = str(row.get("day_phase", "Unknown"))

        if risk_score >= 8 or (day_phase == "Night" and streetlight_status == "OFF"):
            marker_color = "red"
        elif risk_score >= 4 or streetlight_status == "DIM":
            marker_color = "orange"
        else:
            marker_color = "green"

        popup = (
            f"Image: {row.get('image_name', 'Unknown')}<br>"
            f"Potholes: {int(row.get('pothole_count', 0) or 0)}<br>"
            f"Risk Score: {risk_score}<br>"
            f"Priority: {row.get('priority_label', 'Unknown')}<br>"
            f"Repair: {row.get('repair_status', 'Open')}<br>"
            f"Road Health: {row.get('road_health_score', 'NA')}<br>"
            f"Road: {row.get('road_condition', 'Unknown')}<br>"
            f"Streetlight: {streetlight_status}<br>"
            f"Phase: {day_phase}"
        )
        priority = str(row.get("priority_label", "Low"))
        radius = {"Low": 7, "Medium": 9, "High": 11, "Critical": 13}.get(priority, 7)
        folium.CircleMarker(
            location=[float(row["latitude"]), float(row["longitude"])],
            popup=popup,
            radius=radius,
            color=marker_color,
            fill=True,
            fill_opacity=0.75,
        ).add_to(issue_map)

    return issue_map


def create_summary(records):
    dataframe = pd.DataFrame(records)
    if dataframe.empty:
        return {
            "total_inspections": 0,
            "total_potholes": 0,
            "avg_risk_score": 0.0,
            "streetlight_off_count": 0,
            "night_inspections": 0,
            "critical_locations": 0,
            "open_issues": 0,
        }

    return {
        "total_inspections": int(len(dataframe)),
        "total_potholes": int(dataframe["pothole_count"].fillna(0).sum()),
        "avg_risk_score": round(float(dataframe["risk_score"].fillna(0).mean()), 2),
        "streetlight_off_count": int(
            (dataframe["streetlight_status"].fillna("") == "OFF").sum()
        ),
        "night_inspections": int(
            (dataframe["day_phase"].fillna("") == "Night").sum()
        ),
        "critical_locations": int(
            (dataframe.get("priority_label", pd.Series(dtype=str)).fillna("") == "Critical").sum()
        ),
        "open_issues": int(
            (dataframe.get("repair_status", pd.Series(dtype=str)).fillna("Open") != "Resolved").sum()
        ),
    }


def build_hotspot_dataframe(records):
    dataframe = pd.DataFrame(records).copy()
    if dataframe.empty:
        return dataframe

    dataframe = dataframe.dropna(subset=["latitude", "longitude"])
    if dataframe.empty:
        return dataframe

    dataframe["lat_bucket"] = dataframe["latitude"].astype(float).round(3)
    dataframe["lon_bucket"] = dataframe["longitude"].astype(float).round(3)
    dataframe["repair_status"] = dataframe.get("repair_status", "Open")
    dataframe["priority_label"] = dataframe.get("priority_label", "Low")
    priority_order = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
    dataframe["priority_rank"] = dataframe["priority_label"].map(priority_order).fillna(1)

    grouped = (
        dataframe.groupby(["lat_bucket", "lon_bucket"], as_index=False)
        .agg(
            inspections=("id", "count") if "id" in dataframe.columns else ("image_name", "count"),
            total_potholes=("pothole_count", "sum"),
            avg_risk_score=("risk_score", "mean"),
            highest_priority_rank=("priority_rank", "max"),
            open_items=("repair_status", lambda s: int((s != "Resolved").sum())),
            dominant_phase=("day_phase", lambda s: s.mode().iat[0] if not s.mode().empty else "Unknown"),
            lighting_issue_count=("streetlight_status", lambda s: int(s.isin(["OFF", "DIM"]).sum())),
        )
    )
    grouped["avg_risk_score"] = grouped["avg_risk_score"].round(2)
    reverse_priority_order = {value: key for key, value in priority_order.items()}
    grouped["highest_priority"] = grouped["highest_priority_rank"].map(reverse_priority_order)
    grouped["hotspot_name"] = grouped.apply(
        lambda row: f"HS-{row['lat_bucket']:.3f}-{row['lon_bucket']:.3f}",
        axis=1,
    )
    grouped["chronic_hotspot"] = grouped.apply(
        lambda row: "Yes" if int(row["inspections"]) >= 3 or int(row["open_items"]) >= 2 else "No",
        axis=1,
    )
    grouped = grouped.sort_values(
        by=["open_items", "avg_risk_score", "total_potholes"],
        ascending=[False, False, False],
    )
    grouped = grouped.drop(columns=["highest_priority_rank"])
    return grouped


def generate_pdf_report(records, title):
    summary = create_summary(records)
    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(title, styles["Title"]))
    elements.append(
        Paragraph(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["Normal"],
        )
    )
    elements.append(Spacer(1, 12))

    summary_rows = [
        ["Metric", "Value"],
        ["Total inspections", str(summary["total_inspections"])],
        ["Total potholes", str(summary["total_potholes"])],
        ["Average risk score", str(summary["avg_risk_score"])],
        ["Streetlight OFF count", str(summary["streetlight_off_count"])],
        ["Night inspections", str(summary["night_inspections"])],
    ]
    summary_table = Table(summary_rows, colWidths=[220, 180])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.beige]),
            ]
        )
    )
    elements.append(summary_table)
    elements.append(Spacer(1, 16))

    detail_rows = [["Image", "Priority", "Repair", "Potholes", "Risk", "Location"]]
    for record in records:
        latitude = record.get("latitude")
        longitude = record.get("longitude")
        if latitude is None or longitude is None:
            location_text = "Not available"
        else:
            location_text = f"{float(latitude):.5f}, {float(longitude):.5f}"

        detail_rows.append(
            [
                str(record.get("image_name", "")),
                str(record.get("priority_label", "Low")),
                str(record.get("repair_status", "Open")),
                str(record.get("pothole_count", 0)),
                str(record.get("risk_score", 0)),
                location_text,
            ]
        )

    detail_table = Table(
        detail_rows,
        colWidths=[120, 65, 70, 55, 50, 130],
        repeatRows=1,
    )
    detail_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f6f3e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
            ]
        )
    )
    elements.append(detail_table)

    document.build(elements)
    return buffer.getvalue()


def should_trigger_external_alert(record):
    if str(record.get("priority_label", "")) == "Critical":
        return True
    return (
        str(record.get("day_phase", "")) == "Night"
        and str(record.get("streetlight_status", "")) in {"OFF", "DIM"}
        and int(record.get("risk_score", 0) or 0) >= 6
    )


def build_external_alert_message(record):
    return (
        f"Integrated Road Safety Alert\n"
        f"Time: {record.get('created_at', 'Unknown')}\n"
        f"Source: {record.get('image_name', 'Unknown')}\n"
        f"Priority: {record.get('priority_label', 'Unknown')}\n"
        f"Repair Status: {record.get('repair_status', 'Open')}\n"
        f"Road Health Score: {record.get('road_health_score', 'NA')}\n"
        f"Risk Score: {record.get('risk_score', 0)}\n"
        f"Road Condition: {record.get('road_condition', 'Unknown')}\n"
        f"Day Phase: {record.get('day_phase', 'Unknown')}\n"
        f"Streetlight: {record.get('streetlight_status', 'Unknown')}\n"
        f"Location: {record.get('latitude', 'NA')}, {record.get('longitude', 'NA')}\n"
        f"GPS Source: {record.get('gps_source', 'Unknown')}\n"
        f"Action: Field inspection and repair recommendation generated automatically."
    )


def send_email_alert(
    smtp_host,
    smtp_port,
    sender_email,
    sender_password,
    recipient_email,
    subject,
    body,
    use_tls=True,
):
    message = EmailMessage()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = subject
    message.set_content(body)

    with smtplib.SMTP(smtp_host, int(smtp_port), timeout=20) as server:
        if use_tls:
            server.starttls()
        if sender_email and sender_password:
            server.login(sender_email, sender_password)
        server.send_message(message)
