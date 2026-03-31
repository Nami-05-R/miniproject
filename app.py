

import os
import re
import tempfile
from datetime import datetime

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium
from ultralytics import YOLO

from road_safety_core import (
    analyze_streetlights,
    build_issue_map,
    build_hotspot_dataframe,
    build_external_alert_message,
    calculate_priority,
    calculate_road_health_score,
    clear_history,
    create_summary,
    extract_gps_from_image,
    fetch_history_dataframe,
    generate_pdf_report,
    get_available_model_path,
    get_available_streetlight_model_path,
    init_db,
    insert_inspection,
    process_inspection_image,
    REPAIR_STATUSES,
    send_email_alert,
    should_trigger_external_alert,
    update_repair_status,
)


st.set_page_config(page_title="Integrated Road Safety Monitoring", layout="wide")
st.title("Integrated Road Safety Monitoring System")
st.caption(
    "Unified dashboard for pothole detection, streetlight status analysis, geo-tagging, map visualization, and reporting."
)
st.markdown(
    """
    <style>
    .module-banner {
        border-radius: 16px;
        padding: 18px 22px;
        margin: 10px 0 18px 0;
        color: white;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.12);
    }
    .module-day {
        background: linear-gradient(135deg, #157f5c, #39a96b);
    }
    .module-night {
        background: linear-gradient(135deg, #17324d, #375a7f);
    }
    .module-title {
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .module-subtitle {
        font-size: 0.95rem;
        margin-top: 6px;
        opacity: 0.95;
    }
    .insight-card {
        background: #f7f9fc;
        border: 1px solid #d9e2ec;
        border-radius: 14px;
        padding: 14px 16px;
        margin: 8px 0;
    }
    .hero-shell {
        background: linear-gradient(135deg, #0f172a, #16324f 52%, #1f6f78);
        color: white;
        padding: 24px 28px;
        border-radius: 22px;
        margin: 10px 0 22px 0;
        box-shadow: 0 14px 38px rgba(15, 23, 42, 0.20);
    }
    .hero-title {
        font-size: 1.55rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .hero-copy {
        font-size: 0.98rem;
        line-height: 1.5;
        opacity: 0.96;
    }
    .badge-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin: 8px 0 14px 0;
    }
    .status-badge {
        display: inline-block;
        padding: 8px 12px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
    }
    .priority-low { background: #d1fae5; color: #065f46; }
    .priority-medium { background: #fef3c7; color: #92400e; }
    .priority-high { background: #fed7aa; color: #9a3412; }
    .priority-critical { background: #fecaca; color: #991b1b; }
    .repair-open { background: #dbeafe; color: #1d4ed8; }
    .repair-assigned { background: #ede9fe; color: #6d28d9; }
    .repair-in-progress { background: #cffafe; color: #155e75; }
    .repair-resolved { background: #dcfce7; color: #166534; }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin: 16px 0 6px 0;
    }
    .feature-tile {
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 14px 16px;
    }
    .brand-line {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 10px;
    }
    .brand-logo {
        width: 54px;
        height: 54px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #22c55e, #0ea5e9);
        color: white;
        font-size: 1rem;
        font-weight: 800;
        letter-spacing: 0.08em;
    }
    .feature-title {
        font-weight: 700;
        margin-bottom: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero-shell">
        <div class="brand-line">
            <div class="brand-logo">IRSM</div>
            <div class="hero-title">Road Safety Command Center</div>
        </div>
        <div class="hero-copy">
            This platform turns road images into actionable field intelligence by combining pothole severity, lighting condition, GPS evidence,
            repair workflow, and hotspot monitoring. It is designed like a civic operations dashboard rather than a plain detection demo.
        </div>
        <div class="feature-grid">
            <div class="feature-tile"><div class="feature-title">Day and Night Routing</div><div>Images are automatically sent to the right analysis module.</div></div>
            <div class="feature-tile"><div class="feature-title">GPS Evidence</div><div>Each inspection can be mapped and revisited as a field location.</div></div>
            <div class="feature-tile"><div class="feature-title">Priority and Workflow</div><div>Issues become maintenance records, not just detections.</div></div>
            <div class="feature-tile"><div class="feature-title">Hotspot Intelligence</div><div>Nearby detections are grouped into operational road segments.</div></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

init_db()


@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)


@st.cache_resource
def load_streetlight_model(model_path):
    return YOLO(model_path)


def show_overview_metrics(history_df):
    if history_df.empty:
        st.info("No inspections have been stored yet. Run a new inspection to populate the dashboard.")
        return

    summary = create_summary(history_df.to_dict("records"))
    metric_columns = st.columns(6)
    metric_columns[0].metric("Stored inspections", summary["total_inspections"])
    metric_columns[1].metric("Detected potholes", summary["total_potholes"])
    metric_columns[2].metric("Average risk score", summary["avg_risk_score"])
    metric_columns[3].metric("Streetlight OFF", summary["streetlight_off_count"])
    metric_columns[4].metric("Night inspections", summary["night_inspections"])
    metric_columns[5].metric("Open issues", summary["open_issues"])


def render_priority_badges(priority_label, repair_status):
    priority_class = f"priority-{str(priority_label).lower()}"
    repair_class = f"repair-{str(repair_status).lower().replace(' ', '-')}"
    st.markdown(
        f"""
        <div class="badge-row">
            <span class="status-badge {priority_class}">Priority: {priority_label}</span>
            <span class="status-badge {repair_class}">Repair: {repair_status}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_module_banner(module_name, day_phase, streetlight_status, risk_score):
    is_night = module_name == "Night Module"
    css_class = "module-night" if is_night else "module-day"
    st.markdown(
        f"""
        <div class="module-banner {css_class}">
            <div class="module-title">{module_name} Active</div>
            <div class="module-subtitle">
                Phase: {day_phase} | Streetlight: {streetlight_status} | Risk score: {risk_score}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_real_world_insights(record):
    insights = []
    pothole_count = int(record["pothole_count"])
    risk_score = int(record["risk_score"])
    streetlight_status = record["streetlight_status"]
    day_phase = record["day_phase"]
    lat = record["latitude"]
    lon = record["longitude"]

    if risk_score >= 8:
        priority = "High priority maintenance location. This road segment should be escalated for quick repair."
    elif risk_score >= 4:
        priority = "Medium priority maintenance location. This segment should be inspected by a field team soon."
    else:
        priority = "Lower priority location. Keep this point in routine monitoring unless repeated detections increase."
    insights.append(priority)

    if day_phase == "Night" and streetlight_status in {"OFF", "DIM"}:
        insights.append(
            "Night-time visibility risk detected. Poor lighting combined with potholes can increase accident probability for two-wheelers and cars."
        )
    elif day_phase == "Night":
        insights.append(
            "Night module confirms lighting support is present, which reduces some visibility risk even if potholes exist."
        )
    else:
        insights.append(
            "Day module is active, so road-surface damage is the main safety signal while lighting risk is less critical in this inspection."
        )

    if pothole_count == 0:
        insights.append(
            "No potholes were detected in this image, so this location can still be logged as a safe or monitored reference point."
        )
    else:
        insights.append(
            f"GPS-tagged detection makes this useful for municipal teams because the issue can be mapped and revisited at coordinates {lat:.6f}, {lon:.6f}."
        )

    insights.append(
        "In a real deployment, repeated uploads from the same area could be used to rank roads by recurring damage, lighting failure, and repair urgency."
    )
    return insights


def parse_coordinates_from_filename(file_name):
    patterns = [
        r"(-?\d{1,2}\.\d+)[,_ ]+(-?\d{1,3}\.\d+)",
        r"lat(-?\d{1,2}\.\d+).*lon(-?\d{1,3}\.\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, file_name, re.IGNORECASE)
        if not match:
            continue
        latitude = float(match.group(1))
        longitude = float(match.group(2))
        if -90 <= latitude <= 90 and -180 <= longitude <= 180:
            return latitude, longitude
    return None


def build_alert_messages(record):
    alerts = []
    if record["priority_label"] == "Critical":
        alerts.append("Critical alert: dispatch field maintenance and supervisor review immediately.")
    if record["day_phase"] == "Night" and record["streetlight_status"] in {"OFF", "DIM"}:
        alerts.append("Visibility alert: lighting weakness plus road damage raises night-time accident risk.")
    if record["road_health_score"] <= 50:
        alerts.append("Road health alert: this location has dropped into a poor-condition band.")
    return alerts


def get_energy_analysis(record):
    if record["day_phase"] == "Day" and record["streetlight_status"] == "ON":
        return "Possible streetlight energy waste detected during daytime."
    if record["day_phase"] == "Night" and record["streetlight_status"] == "OFF":
        return "Lighting service gap detected at night."
    if record["day_phase"] == "Night" and record["streetlight_status"] == "DIM":
        return "Streetlight output appears weak, suggesting reduced lighting efficiency."
    return "No obvious smart-lighting anomaly detected for this inspection."


def extract_video_frames(video_file, sample_every_seconds=2, max_frames=6):
    suffix = os.path.splitext(video_file.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
        temp_video.write(video_file.getbuffer())
        temp_path = temp_video.name

    frames = []
    capture = cv2.VideoCapture(temp_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 1.0
    frame_step = max(int(fps * sample_every_seconds), 1)
    frame_index = 0

    try:
        while capture.isOpened() and len(frames) < max_frames:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % frame_step == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb_frame)
                timestamp_seconds = round(frame_index / fps, 1)
                frames.append((timestamp_seconds, pil_frame))
            frame_index += 1
    finally:
        capture.release()
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return frames


def interpolate_route_coordinates(start_lat, start_lon, end_lat, end_lon, index, total_points):
    if total_points <= 1:
        return float(start_lat), float(start_lon)
    ratio = index / max(total_points - 1, 1)
    latitude = float(start_lat) + (float(end_lat) - float(start_lat)) * ratio
    longitude = float(start_lon) + (float(end_lon) - float(start_lon)) * ratio
    return latitude, longitude


def compact_text(value):
    return " ".join(str(value or "").split()).strip()


def build_contact_note(reporter_name, reporter_contact, landmark, citizen_note):
    note_parts = ["Citizen report"]
    if compact_text(reporter_name):
        note_parts.append(f"Reporter: {compact_text(reporter_name)}")
    if compact_text(reporter_contact):
        note_parts.append(f"Contact: {compact_text(reporter_contact)}")
    if compact_text(landmark):
        note_parts.append(f"Landmark: {compact_text(landmark)}")
    if compact_text(citizen_note):
        note_parts.append(f"Citizen note: {compact_text(citizen_note)}")
    return " | ".join(note_parts)


def run_image_inspection(
    image_name,
    rgb_image,
    latitude,
    longitude,
    gps_source,
    note_text,
    save_to_database,
    enable_email_alerts,
    recipient_email,
    smtp_host,
    smtp_port,
    sender_email,
    sender_password,
):
    module_result = process_inspection_image(
        model,
        rgb_image,
        conf_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        streetlight_model=streetlight_model,
    )
    pothole_result = module_result["pothole_result"]
    streetlight_result = module_result["streetlight_result"]
    priority_score, priority_label = calculate_priority(
        pothole_result["risk_score"],
        streetlight_result["day_phase"],
        streetlight_result["streetlight_status"],
        pothole_result["total_potholes"],
    )
    road_health_score = calculate_road_health_score(
        pothole_result["risk_score"],
        streetlight_result["day_phase"],
        streetlight_result["streetlight_status"],
    )

    record = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_name": image_name,
        "latitude": float(latitude),
        "longitude": float(longitude),
        "gps_source": gps_source,
        "day_phase": streetlight_result["day_phase"],
        "streetlight_status": streetlight_result["streetlight_status"],
        "streetlight_score": int(streetlight_result["streetlight_score"]),
        "streetlight_analysis_source": streetlight_result.get("analysis_source", "heuristic"),
        "ambient_brightness": float(streetlight_result["ambient_brightness"]),
        "priority_score": int(priority_score),
        "priority_label": priority_label,
        "road_health_score": int(road_health_score),
        "repair_status": "Open",
        "pothole_count": int(pothole_result["total_potholes"]),
        "small_count": int(pothole_result["counts"]["Small"]),
        "medium_count": int(pothole_result["counts"]["Medium"]),
        "large_count": int(pothole_result["counts"]["Large"]),
        "avg_confidence": float(pothole_result["avg_confidence"]),
        "risk_score": int(pothole_result["risk_score"]),
        "road_condition": pothole_result["road_condition"],
        "processing_module": module_result["module_name"],
        "notes": compact_text(note_text),
    }

    if enable_email_alerts and recipient_email and should_trigger_external_alert(record):
        try:
            send_email_alert(
                smtp_host=smtp_host,
                smtp_port=smtp_port,
                sender_email=sender_email,
                sender_password=sender_password,
                recipient_email=recipient_email,
                subject=f"Road Safety Alert: {record['priority_label']} at {record['image_name']}",
                body=build_external_alert_message(record),
                use_tls=True,
            )
            record["notes"] = f"{record['notes']} | Email alert sent".strip(" |")
        except Exception as exc:
            record["notes"] = f"{record['notes']} | Email alert failed: {exc}".strip(" |")

    inserted_id = None
    if save_to_database:
        inserted_id = insert_inspection(record)
        if inserted_id is not None:
            record["id"] = int(inserted_id)

    result_bundle = {
        "record": record,
        "detections": pothole_result["detections"],
        "pothole_image": pothole_result["annotated_image"],
        "streetlight_image": streetlight_result["annotated_image"],
    }
    return result_bundle, inserted_id


def show_session_results(session_results):
    if not session_results:
        return

    session_records = [result["record"] for result in session_results]
    session_df = pd.DataFrame(session_records)
    summary = create_summary(session_records)

    st.subheader("Current Inspection Session")
    summary_columns = st.columns(5)
    summary_columns[0].metric("Images processed", summary["total_inspections"])
    summary_columns[1].metric("Potholes found", summary["total_potholes"])
    summary_columns[2].metric("Average risk score", summary["avg_risk_score"])
    summary_columns[3].metric("Streetlight OFF", summary["streetlight_off_count"])
    summary_columns[4].metric("Night images", summary["night_inspections"])

    severity_df = pd.DataFrame(
        {
            "Severity": ["Small", "Medium", "Large"],
            "Count": [
                int(session_df["small_count"].sum()),
                int(session_df["medium_count"].sum()),
                int(session_df["large_count"].sum()),
            ],
        }
    )
    streetlight_df = (
        session_df["streetlight_status"]
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Count")
    )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        severity_chart = px.bar(
            severity_df,
            x="Severity",
            y="Count",
            color="Severity",
            color_discrete_map={
                "Small": "green",
                "Medium": "orange",
                "Large": "red",
            },
            title="Pothole Severity Distribution",
        )
        st.plotly_chart(severity_chart, width="stretch")
    with chart_col2:
        streetlight_chart = px.pie(
            streetlight_df,
            names="Status",
            values="Count",
            title="Streetlight Status Distribution",
        )
        st.plotly_chart(streetlight_chart, width="stretch")

    for result in session_results:
        record = result["record"]
        with st.expander(f"Inspection: {record['image_name']}", expanded=True):
            render_module_banner(
                record["processing_module"],
                record["day_phase"],
                record["streetlight_status"],
                record["risk_score"],
            )
            image_col1, image_col2 = st.columns(2)
            with image_col1:
                st.image(
                    result["pothole_image"],
                    caption=f"Pothole detection - {record['image_name']}",
                    width="stretch",
                )
            with image_col2:
                st.image(
                    result["streetlight_image"],
                    caption=f"Streetlight analysis - {record['image_name']}",
                    width="stretch",
                )

            render_priority_badges(record["priority_label"], record["repair_status"])

            detail_columns = st.columns(6)
            detail_columns[0].metric("Potholes", int(record["pothole_count"]))
            detail_columns[1].metric("Risk score", int(record["risk_score"]))
            detail_columns[2].metric("Road health", int(record["road_health_score"]))
            detail_columns[3].metric("Streetlight", record["streetlight_status"])
            detail_columns[4].metric("Day phase", record["day_phase"])
            detail_columns[5].metric("Avg confidence", f"{record['avg_confidence']:.2f}")
            st.write(f"Processing module: {record['processing_module']}")
            st.write(f"Road condition: {record['road_condition']}")
            st.write(
                "Streetlight engine: "
                f"{str(record.get('streetlight_analysis_source', 'heuristic')).replace('_', ' ').title()}"
            )

            location_label = (
                f"{record['latitude']:.6f}, {record['longitude']:.6f}"
                if pd.notna(record["latitude"]) and pd.notna(record["longitude"])
                else "Location not available"
            )
            st.write(f"GPS source: {record['gps_source']}")
            st.write(f"Location: {location_label}")
            st.write(f"Smart lighting insight: {get_energy_analysis(record)}")
            if record["notes"]:
                st.write(f"Notes: {record['notes']}")

            st.markdown("#### Operational Insights")
            for insight in build_real_world_insights(record):
                st.markdown(
                    f'<div class="insight-card">{insight}</div>',
                    unsafe_allow_html=True,
                )
            for alert in build_alert_messages(record):
                st.error(alert)

            detections_df = pd.DataFrame(result["detections"])
            if detections_df.empty:
                st.info("No potholes were detected in this image.")
            else:
                st.dataframe(detections_df, width="stretch", hide_index=True)

    st.subheader("Session Map")
    session_map = build_issue_map(session_df)
    if session_map is not None:
        st_folium(session_map, width=1100, height=500)
    else:
        st.info("Map is unavailable because the session records do not contain valid coordinates.")

    csv_data = session_df.to_csv(index=False).encode("utf-8")
    pdf_data = generate_pdf_report(
        session_records,
        "Integrated Road Safety Monitoring Session Report",
    )
    download_col1, download_col2 = st.columns(2)
    with download_col1:
        st.download_button(
            label="Download session CSV",
            data=csv_data,
            file_name="road_safety_session_report.csv",
            mime="text/csv",
        )
    with download_col2:
        st.download_button(
            label="Download session PDF",
            data=pdf_data,
            file_name="road_safety_session_report.pdf",
            mime="application/pdf",
        )

    hotspot_df = build_hotspot_dataframe(session_records)
    if not hotspot_df.empty:
        st.subheader("Session Hotspots")
        st.dataframe(hotspot_df, width="stretch", hide_index=True)


model_path = get_available_model_path()
if not model_path:
    st.error("No trained YOLO pothole model was found in the project folder.")
    st.stop()

model = load_model(model_path)
streetlight_model_path = get_available_streetlight_model_path()
streetlight_model = load_streetlight_model(streetlight_model_path) if streetlight_model_path else None

with st.sidebar:
    st.header("Inspection Settings")
    st.write(f"Model loaded: `{model_path}`")
    if streetlight_model_path:
        st.success(f"Streetlight model loaded: `{streetlight_model_path}`")
    else:
        st.info("Streetlight analysis is using the heuristic fallback. Add a trained streetlight model to upgrade this module.")
    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
    )
    iou_threshold = st.slider(
        "IOU threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
    )
    manual_latitude = st.number_input(
        "Fallback latitude",
        value=10.850500,
        format="%.6f",
    )
    manual_longitude = st.number_input(
        "Fallback longitude",
        value=76.271100,
        format="%.6f",
    )
    survey_mode = st.checkbox("Enable route survey mode for videos", value=False)
    survey_end_latitude = st.number_input(
        "Route end latitude",
        value=10.851500,
        format="%.6f",
        disabled=not survey_mode,
    )
    survey_end_longitude = st.number_input(
        "Route end longitude",
        value=76.275100,
        format="%.6f",
        disabled=not survey_mode,
    )
    video_sample_seconds = st.slider(
        "Video frame interval (seconds)",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
    )
    max_video_frames = st.slider(
        "Maximum frames per video",
        min_value=2,
        max_value=12,
        value=6,
        step=1,
    )
    save_to_database = st.checkbox("Save inspection results to SQLite", value=True)
    enable_email_alerts = st.checkbox("Enable external email alerts", value=False)
    recipient_email = st.text_input("Alert recipient email", placeholder="authority@example.com")
    smtp_host = st.text_input("SMTP host", value="smtp.gmail.com")
    smtp_port = st.number_input("SMTP port", min_value=1, max_value=65535, value=587)
    sender_email = st.text_input("Sender email", placeholder="your_email@example.com")
    sender_password = st.text_input("Sender app password", type="password")
    if streetlight_model_path:
        st.caption(
            "Streetlight analysis is currently powered by a trained streetlight model with heuristic overlays for contextual risk interpretation."
        )
    else:
        st.caption(
            "Streetlight analysis is currently heuristic and uses image brightness patterns with day/night estimation."
        )
    st.caption(
        "Automatic GPS extraction works for images that contain EXIF GPS metadata. Videos can auto-detect coordinates from filename patterns like `10.8505_76.2711.mp4`; otherwise the app uses manual coordinates."
    )
    st.caption(
        "Route survey mode interpolates sampled video frames between the start and end coordinates, which makes road-stretch inspection more realistic."
    )

history_df = fetch_history_dataframe()
show_overview_metrics(history_df)

tabs = st.tabs(["New Inspection", "Citizen Portal", "Inspection History", "Project Summary"])

with tabs[0]:
    st.markdown(
        """
        ### Smart Intake
        Upload multiple road images, capture a live photo from a phone, upload multiple videos, or combine them. Images can use automatic EXIF GPS when available. Videos are sampled into inspection frames and can follow a route survey path.
        """
    )
    with st.form("inspection_form"):
        captured_photo = st.camera_input("Capture live road image")
        uploaded_files = st.file_uploader(
            "Upload road images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        uploaded_videos = st.file_uploader(
            "Upload road videos",
            type=["mp4", "mov", "avi", "mkv"],
            accept_multiple_files=True,
        )
        session_notes = st.text_input(
            "Inspection note",
            placeholder="Optional note stored with each image in this run",
        )
        submitted = st.form_submit_button("Run inspection")

    if submitted:
        if not uploaded_files and not uploaded_videos and not captured_photo:
            st.warning("Please upload at least one image or video, or capture a live image before running the inspection.")
        else:
            session_results = []
            image_files = list(uploaded_files or [])
            if captured_photo is not None:
                captured_photo.name = f"live_capture_{datetime.now().strftime('%H%M%S')}.jpg"
                image_files.append(captured_photo)
            video_files = uploaded_videos or []

            for uploaded_file in image_files:
                original_image = Image.open(uploaded_file)
                gps_coords = extract_gps_from_image(original_image)
                rgb_image = original_image.convert("RGB")

                if gps_coords:
                    latitude, longitude = gps_coords
                    gps_source = "Image EXIF"
                else:
                    latitude, longitude = manual_latitude, manual_longitude
                    gps_source = "Manual input"

                result_bundle, _ = run_image_inspection(
                    image_name=uploaded_file.name,
                    rgb_image=rgb_image,
                    latitude=latitude,
                    longitude=longitude,
                    gps_source=gps_source,
                    note_text=session_notes.strip(),
                    save_to_database=save_to_database,
                    enable_email_alerts=enable_email_alerts,
                    recipient_email=recipient_email,
                    smtp_host=smtp_host,
                    smtp_port=smtp_port,
                    sender_email=sender_email,
                    sender_password=sender_password,
                )
                session_results.append(result_bundle)

            for uploaded_video in video_files:
                sampled_frames = extract_video_frames(
                    uploaded_video,
                    sample_every_seconds=video_sample_seconds,
                    max_frames=max_video_frames,
                )
                if not sampled_frames:
                    continue

                video_coords = parse_coordinates_from_filename(uploaded_video.name)
                if video_coords:
                    video_latitude, video_longitude = video_coords
                    video_gps_source = "Video filename auto-detect"
                else:
                    video_latitude, video_longitude = manual_latitude, manual_longitude
                    video_gps_source = "Manual input (video)"

                for frame_index, (timestamp_seconds, rgb_image) in enumerate(sampled_frames):
                    frame_latitude = float(video_latitude)
                    frame_longitude = float(video_longitude)
                    frame_gps_source = video_gps_source
                    if survey_mode:
                        frame_latitude, frame_longitude = interpolate_route_coordinates(
                            video_latitude,
                            video_longitude,
                            survey_end_latitude,
                            survey_end_longitude,
                            frame_index,
                            len(sampled_frames),
                        )
                        frame_gps_source = "Route survey interpolation"

                    result_bundle, _ = run_image_inspection(
                        image_name=f"{uploaded_video.name} @ {timestamp_seconds}s",
                        rgb_image=rgb_image,
                        latitude=frame_latitude,
                        longitude=frame_longitude,
                        gps_source=frame_gps_source,
                        note_text=f"{session_notes.strip()} | Source video frame".strip(" |"),
                        save_to_database=save_to_database,
                        enable_email_alerts=enable_email_alerts,
                        recipient_email=recipient_email,
                        smtp_host=smtp_host,
                        smtp_port=smtp_port,
                        sender_email=sender_email,
                        sender_password=sender_password,
                    )
                    session_results.append(result_bundle)

            st.session_state["session_results"] = session_results

    show_session_results(st.session_state.get("session_results", []))

with tabs[1]:
    st.subheader("Citizen Quick Report")
    st.markdown(
        """
        This mode is designed for public reporting when the app is hosted online. A citizen can capture or upload one road image,
        add a landmark, and submit a GPS-tagged safety report that immediately enters the same priority, map, and workflow pipeline.
        """
    )
    with st.form("citizen_report_form"):
        citizen_name = st.text_input("Reporter name", placeholder="Optional")
        citizen_contact = st.text_input("Phone or email", placeholder="Optional")
        citizen_landmark = st.text_input("Nearest landmark or area", placeholder="Bus stand, school gate, junction, etc.")
        citizen_issue_type = st.selectbox(
            "Issue focus",
            ["Road damage", "Poor lighting", "Road damage and poor lighting", "General hazard"],
        )
        citizen_photo = st.camera_input("Capture road issue photo")
        citizen_upload = st.file_uploader(
            "Or upload a road image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            key="citizen_upload",
        )
        citizen_latitude = st.number_input(
            "Citizen latitude",
            value=float(manual_latitude),
            format="%.6f",
            key="citizen_latitude",
        )
        citizen_longitude = st.number_input(
            "Citizen longitude",
            value=float(manual_longitude),
            format="%.6f",
            key="citizen_longitude",
        )
        citizen_note = st.text_area(
            "Short issue description",
            placeholder="Example: Deep pothole near the left lane. Streetlight is weak after 7 PM.",
        )
        citizen_submit = st.form_submit_button("Submit citizen report")

    if citizen_submit:
        chosen_input = citizen_photo if citizen_photo is not None else citizen_upload
        if chosen_input is None:
            st.warning("Please capture or upload one road image before submitting a citizen report.")
        else:
            if citizen_photo is not None:
                chosen_input.name = f"citizen_capture_{datetime.now().strftime('%H%M%S')}.jpg"
            original_image = Image.open(chosen_input)
            gps_coords = extract_gps_from_image(original_image)
            rgb_image = original_image.convert("RGB")
            if gps_coords:
                citizen_report_latitude, citizen_report_longitude = gps_coords
                citizen_gps_source = "Image EXIF"
            else:
                citizen_report_latitude, citizen_report_longitude = citizen_latitude, citizen_longitude
                citizen_gps_source = "Citizen input"

            citizen_record_note = build_contact_note(
                reporter_name=citizen_name,
                reporter_contact=citizen_contact,
                landmark=f"{citizen_landmark} | Focus: {citizen_issue_type}",
                citizen_note=citizen_note,
            )
            citizen_result, inserted_id = run_image_inspection(
                image_name=f"Citizen report - {chosen_input.name}",
                rgb_image=rgb_image,
                latitude=citizen_report_latitude,
                longitude=citizen_report_longitude,
                gps_source=citizen_gps_source,
                note_text=citizen_record_note,
                save_to_database=True,
                enable_email_alerts=enable_email_alerts,
                recipient_email=recipient_email,
                smtp_host=smtp_host,
                smtp_port=smtp_port,
                sender_email=sender_email,
                sender_password=sender_password,
            )
            st.session_state["citizen_results"] = [citizen_result]
            st.success(
                f"Citizen report saved successfully{f' as record {inserted_id}' if inserted_id else ''}."
            )

    show_session_results(st.session_state.get("citizen_results", []))

with tabs[2]:
    refreshed_history_df = fetch_history_dataframe()
    st.subheader("Stored Inspection History")
    if refreshed_history_df.empty:
        st.info("The SQLite database is currently empty.")
    else:
        history_summary = create_summary(refreshed_history_df.to_dict("records"))
        hist_metrics = st.columns(4)
        hist_metrics[0].metric("Critical locations", history_summary["critical_locations"])
        hist_metrics[1].metric("Open issues", history_summary["open_issues"])
        hist_metrics[2].metric(
            "Resolved issues",
            max(int(len(refreshed_history_df)) - int(history_summary["open_issues"]), 0),
        )
        hist_metrics[3].metric(
            "Priority avg",
            round(float(refreshed_history_df["priority_score"].fillna(0).mean()), 2),
        )

        st.dataframe(refreshed_history_df, width="stretch", hide_index=True)

        history_chart_col1, history_chart_col2 = st.columns(2)
        with history_chart_col1:
            history_severity_df = pd.DataFrame(
                {
                    "Severity": ["Small", "Medium", "Large"],
                    "Count": [
                        int(refreshed_history_df["small_count"].sum()),
                        int(refreshed_history_df["medium_count"].sum()),
                        int(refreshed_history_df["large_count"].sum()),
                    ],
                }
            )
            severity_chart = px.bar(
                history_severity_df,
                x="Severity",
                y="Count",
                color="Severity",
                color_discrete_map={
                    "Small": "green",
                    "Medium": "orange",
                    "Large": "red",
                },
                title="Historic Pothole Severity Counts",
            )
            st.plotly_chart(severity_chart, width="stretch")
        with history_chart_col2:
            history_streetlight_df = (
                refreshed_history_df["streetlight_status"]
                .value_counts()
                .rename_axis("Status")
                .reset_index(name="Count")
            )
            streetlight_chart = px.pie(
                history_streetlight_df,
                names="Status",
                values="Count",
                title="Historic Streetlight Status",
            )
            st.plotly_chart(streetlight_chart, width="stretch")

        st.subheader("Unified Issue Map")
        history_map = build_issue_map(refreshed_history_df)
        if history_map is not None:
            st_folium(history_map, width=1100, height=500)

        hotspot_df = build_hotspot_dataframe(refreshed_history_df.to_dict("records"))
        if not hotspot_df.empty:
            st.subheader("Hotspot Intelligence")
            st.dataframe(hotspot_df, width="stretch", hide_index=True)

        if "created_at" in refreshed_history_df.columns:
            trend_df = refreshed_history_df.copy()
            trend_df["created_at"] = pd.to_datetime(trend_df["created_at"], errors="coerce")
            trend_df["inspection_day"] = trend_df["created_at"].dt.date
            trend_df = (
                trend_df.groupby("inspection_day", as_index=False)
                .agg(
                    inspections=("id", "count"),
                    avg_risk=("risk_score", "mean"),
                )
            )
            st.subheader("Time-Based Tracking")
            trend_chart = px.line(
                trend_df,
                x="inspection_day",
                y=["inspections", "avg_risk"],
                markers=True,
                title="Inspection Activity and Risk Trend",
            )
            st.plotly_chart(trend_chart, width="stretch")

        active_alerts = []
        for _, row in refreshed_history_df.iterrows():
            if row.get("priority_label") == "Critical":
                active_alerts.append(
                    f"Critical alert at record {int(row['id'])}: {row['image_name']}"
                )
            if row.get("day_phase") == "Night" and row.get("streetlight_status") in {"OFF", "DIM"}:
                active_alerts.append(
                    f"Lighting alert at record {int(row['id'])}: {row['streetlight_status']} during night inspection."
                )
        if active_alerts:
            st.subheader("Automatic Alert Center")
            for alert in active_alerts[:8]:
                st.warning(alert)

        st.subheader("Repair Workflow")
        workflow_col1, workflow_col2 = st.columns([2, 1])
        with workflow_col1:
            editable_df = refreshed_history_df[
                [
                    "id",
                    "image_name",
                    "priority_label",
                    "repair_status",
                    "risk_score",
                    "day_phase",
                    "streetlight_status",
                ]
            ].copy()
            st.dataframe(editable_df, width="stretch", hide_index=True)
        with workflow_col2:
            record_ids = refreshed_history_df["id"].astype(int).tolist()
            selected_record_id = st.selectbox("Select issue ID", record_ids)
            selected_status = st.selectbox("Update repair status", REPAIR_STATUSES)
            if st.button("Apply repair update"):
                update_repair_status(selected_record_id, selected_status)
                st.success(
                    f"Issue {selected_record_id} updated to '{selected_status}'."
                )
                st.rerun()

        st.subheader("Before and After Repair Verification")
        compare_df = refreshed_history_df[
            [
                "id",
                "image_name",
                "priority_label",
                "repair_status",
                "risk_score",
                "road_health_score",
                "created_at",
                "latitude",
                "longitude",
            ]
        ].copy()
        compare_options = compare_df.apply(
            lambda row: f"{int(row['id'])} | {row['image_name']} | {row['created_at']}",
            axis=1,
        ).tolist()
        compare_col1, compare_col2 = st.columns(2)
        with compare_col1:
            before_choice = st.selectbox("Before record", compare_options, key="before_record")
        with compare_col2:
            after_choice = st.selectbox("After record", compare_options, key="after_record")

        if before_choice and after_choice:
            before_id = int(before_choice.split("|")[0].strip())
            after_id = int(after_choice.split("|")[0].strip())
            before_row = compare_df.loc[compare_df["id"] == before_id].iloc[0]
            after_row = compare_df.loc[compare_df["id"] == after_id].iloc[0]

            before_risk = pd.to_numeric(before_row["risk_score"], errors="coerce")
            after_risk = pd.to_numeric(after_row["risk_score"], errors="coerce")
            if pd.isna(before_risk):
                before_risk = 0
            if pd.isna(after_risk):
                after_risk = 0
            before_health = pd.to_numeric(before_row["road_health_score"], errors="coerce")
            after_health = pd.to_numeric(after_row["road_health_score"], errors="coerce")
            if pd.isna(before_health):
                before_health = calculate_road_health_score(
                    before_risk,
                    refreshed_history_df.loc[refreshed_history_df["id"] == before_id, "day_phase"].iloc[0],
                    refreshed_history_df.loc[refreshed_history_df["id"] == before_id, "streetlight_status"].iloc[0],
                )
            if pd.isna(after_health):
                after_health = calculate_road_health_score(
                    after_risk,
                    refreshed_history_df.loc[refreshed_history_df["id"] == after_id, "day_phase"].iloc[0],
                    refreshed_history_df.loc[refreshed_history_df["id"] == after_id, "streetlight_status"].iloc[0],
                )

            delta_risk = after_risk - before_risk
            delta_health = float(after_health) - float(before_health)
            verify_cols = st.columns(4)
            verify_cols[0].metric("Before risk", int(before_risk))
            verify_cols[1].metric("After risk", int(after_risk), delta=int(delta_risk))
            verify_cols[2].metric("Before health", int(before_health))
            verify_cols[3].metric("After health", int(after_health), delta=int(delta_health))

            if delta_risk < 0 or delta_health > 0:
                st.success("Verification suggests road condition improved after intervention.")
            elif delta_risk > 0 or delta_health < 0:
                st.warning("Verification suggests road condition worsened or has not improved.")
            else:
                st.info("Verification shows no major change between the selected records.")

        history_csv = refreshed_history_df.to_csv(index=False).encode("utf-8")
        history_pdf = generate_pdf_report(
            refreshed_history_df.to_dict("records"),
            "Integrated Road Safety Monitoring History Report",
        )

        history_download_col1, history_download_col2 = st.columns(2)
        with history_download_col1:
            st.download_button(
                label="Download full history CSV",
                data=history_csv,
                file_name="road_safety_history.csv",
                mime="text/csv",
            )
        with history_download_col2:
            st.download_button(
                label="Download full history PDF",
                data=history_pdf,
                file_name="road_safety_history.pdf",
                mime="application/pdf",
            )

    if st.button("Clear database history"):
        clear_history()
        st.session_state["session_results"] = []
        st.rerun()

with tabs[3]:
    st.subheader("Project Coverage")
    st.markdown(
        """
        This version now covers the mini project overview end-to-end:

        1. Automated pothole detection using YOLOv8
        2. Explicit Day Module and Night Module routing
        3. Streetlight ON/OFF/DIM analysis using a trained streetlight detector plus illumination logic
        4. Geo-tag capture from image EXIF or manual coordinates
        5. Live camera capture and mobile-friendly field intake
        6. Route survey mode for video inspection
        7. Unified road safety map with multiple stored markers
        8. SQLite-based data logging and downloadable PDF or CSV reports
        9. Before-and-after repair verification and recurring hotspot tracking
        10. Citizen reporting workflow for public photo submissions
        """
    )

    st.subheader("System Workflow")
    st.code(
        "Upload image -> GPS extraction -> Day/Night detection -> "
        "Route to Day Module or Night Module -> "
        "Pothole detection + Streetlight analysis -> SQLite storage -> "
        "Map visualization -> Report generation"
    )

    st.subheader("Real-World Usefulness")
    st.markdown(
        """
        1. Municipal teams can map potholes and poor-lighting zones with GPS-tagged evidence instead of relying only on citizen complaints.
        2. Night inspections highlight roads where potholes and weak streetlights create a higher accident risk, especially for two-wheelers.
        3. Repeated inspections can build a maintenance priority list so repair budgets go first to the worst road segments.
        4. A city dashboard could use this same pipeline to compare wards, track recurring damage, and monitor whether repairs actually reduced risk.
        5. Emergency response and transport departments could use the map output to identify unsafe stretches near schools, junctions, bus routes, and hospitals.
        6. Once deployed online, citizens could submit issue photos remotely, turning the system into a public road-safety reporting platform.
        """
    )

    st.subheader("Product Ideas")
    st.markdown(
        """
        1. Hotspot clustering now groups repeated nearby detections into actionable road segments instead of isolated image records.
        2. Repair workflow now supports `Open`, `Assigned`, `In Progress`, and `Resolved` tracking.
        3. Priority scoring now elevates night-time potholes with poor lighting into stronger alerts for road authorities.
        4. A next upgrade could add before-and-after comparison photos so authorities can verify whether a repair really fixed the problem.
        5. A next upgrade could add alert rules that flag dangerous school zones, junctions, and bus corridors automatically.
        6. Public deployment can expose the Citizen Portal so reports come from field workers and ordinary road users, not only desktop operators.
        """
    )

    st.subheader("Deployment Story")
    st.markdown(
        """
        1. Local mode supports classroom demo, laptop usage, and phone access on the same Wi-Fi network.
        2. Hosted mode can publish the same app to a public URL so citizens can upload road photos from anywhere.
        3. The Citizen Portal feeds the same analytics, map, hotspot, alert, and repair workflow used by road authorities.
        4. This makes the project easy to explain as both a prototype dashboard and a scalable civic reporting platform.
        """
    )

    st.subheader("Why This Feels Like a Real Product")
    st.markdown(
        """
        1. It does not stop at object detection. It turns inspection images into operational records with status, priority, and hotspot grouping.
        2. GPS is treated as decision-support data, which means authorities can assign teams, revisit sites, and verify repairs.
        3. Day and Night modules make the system easy to explain architecturally and also make the safety logic feel realistic.
        4. Time-based tracking is built in through stored inspection timestamps, which supports trend monitoring.
        5. Before-and-after verification lets the same dashboard support repair validation, not only issue discovery.
        6. Automatic alerts and smart-lighting energy insights make the system feel more operational and less academic.
        7. The interface now behaves more like a city operations console than a notebook demo, which makes it stronger for review and presentation.
        """
    )

    st.subheader("Available Project Assets")
    asset_rows = []
    for asset_path in [
        model_path,
        "predict.py",
        "train.py",
        "data.yaml",
        os.path.join("runs", "detect", "train2", "results.png"),
    ]:
        asset_rows.append({"Path": asset_path, "Exists": os.path.exists(asset_path)})
    st.dataframe(pd.DataFrame(asset_rows), hide_index=True, width="stretch")
