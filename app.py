from datetime import datetime
import io
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

st.set_page_config(
    page_title="WildGuard AI",
    page_icon="🌿",
    layout="wide",
)

FOREST_CLASSES = {"person"}
ROAD_ANIMAL_CLASSES = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}
ALL_TARGET_CLASSES = FOREST_CLASSES | ROAD_ANIMAL_CLASSES


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
            .hero-banner {
                background: linear-gradient(135deg, #0f291e 0%, #1b4332 55%, #2d6a4f 100%);
                padding: 1.5rem 1.75rem;
                border-radius: 14px;
                color: #f8fafc;
                margin-bottom: 1.25rem;
                border: 1px solid rgba(255, 255, 255, 0.12);
            }
            .hero-badge {
                display: inline-block;
                background: rgba(255, 255, 255, 0.14);
                color: #d8f3dc;
                font-size: 0.78rem;
                font-weight: 600;
                padding: 0.25rem 0.65rem;
                border-radius: 999px;
                margin-bottom: 0.5rem;
                letter-spacing: 0.03em;
            }
            .hero-title {
                margin: 0;
                font-size: 2rem;
                font-weight: 700;
                color: #ffffff;
            }
            .hero-subtitle {
                margin: 0.35rem 0 0 0;
                font-size: 1rem;
                color: #e2e8f0;
            }
            .advisory-box {
                background-color: #fffbeb;
                border-left: 4px solid #d97706;
                padding: 0.85rem 1rem;
                border-radius: 8px;
                color: #451a03;
                margin: 0.5rem 0 1rem 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading YOLOv8n model...")
def load_model() -> YOLO:
    return YOLO("yolov8n.pt")


def compute_proximity(box_area: float, frame_area: float) -> str:
    if frame_area <= 0:
        return "Low"
    ratio = box_area / frame_area
    if ratio >= 0.12:
        return "High"
    if ratio >= 0.03:
        return "Medium"
    return "Low"


def get_road_advisory(detected_animals: List[str]) -> str:
    unique_animals = sorted(set(detected_animals))
    joined = ", ".join(unique_animals)
    if any(animal in {"elephant", "bear", "cow", "horse"} for animal in unique_animals):
        return (
            f"Large/high-impact wildlife detected ({joined}). Reduce speed immediately, "
            "maintain a safe distance, avoid honking loudly, and do not block escape paths."
        )
    return (
        f"Wildlife spotted near roadway ({joined}). Slow down, stay alert for sudden "
        "movement across lanes, and drive cautiously."
    )


def process_image(
    pil_image: Image.Image,
    model: YOLO,
    mode: str,
    confidence: float,
) -> Tuple[np.ndarray, List[Dict[str, object]], List[str]]:
    rgb_image = np.array(pil_image.convert("RGB"))
    annotated = rgb_image.copy()
    height, width, _ = annotated.shape
    frame_area = float(width * height)

    results = model.predict(source=rgb_image, conf=confidence, verbose=False)
    detections: List[Dict[str, object]] = []
    road_animals: List[str] = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            label = model.names.get(cls_id, str(cls_id)).lower()
            conf_score = float(box.conf[0].item())

            if label not in ALL_TARGET_CLASSES:
                continue

            is_person = label in FOREST_CLASSES
            if mode == "Forest Protection Only" and not is_person:
                continue
            if mode == "Wildlife Road Safety Only" and is_person:
                continue

            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            box_area = float(max(0, x2 - x1) * max(0, y2 - y1))
            proximity = compute_proximity(box_area, frame_area)

            if is_person:
                category = "Forest Protection"
                alert_status = "Forest Alert"
                color = (220, 38, 38)
            else:
                category = "Wildlife Road Safety"
                alert_status = "Road Warning"
                color = (234, 88, 12)
                road_animals.append(label)

            text = f"{label} ({conf_score:.0%}) [{proximity}]"
            thickness = max(2, int(round(min(width, height) / 250)))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            font_scale = max(0.5, min(width, height) / 1100.0)
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness - 1)
            )
            label_top = max(0, y1 - th - 10)
            label_bottom = min(height, label_top + th + 10)
            label_right = min(width, x1 + tw + 10)

            cv2.rectangle(annotated, (x1, label_top), (label_right, label_bottom), color, -1)
            cv2.putText(
                annotated,
                text,
                (x1 + 5, max(12, label_bottom - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                max(1, thickness - 1),
                cv2.LINE_AA,
            )

            detections.append(
                {
                    "Timestamp": timestamp,
                    "Category": category,
                    "Object Label": label,
                    "Confidence": f"{conf_score:.1%}",
                    "Proximity": proximity,
                    "Alert Status": alert_status,
                }
            )

    return annotated, detections, road_animals


def encode_png_bytes(rgb_array: np.ndarray) -> bytes:
    pil_out = Image.fromarray(rgb_array)
    buffer = io.BytesIO()
    pil_out.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> None:
    inject_custom_css()

    if "detection_log" not in st.session_state:
        st.session_state.detection_log = []
    if "last_processed_key" not in st.session_state:
        st.session_state.last_processed_key = None

    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-badge">MINOR PROJECT DEMO • YOLOV8 + OPENCV + STREAMLIT</div>
            <h1 class="hero-title">🌿 WildGuard AI</h1>
            <p class="hero-subtitle">Forest Protection &amp; Wildlife Road Safety Demo</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Detection Controls")
        mode = st.selectbox(
            "Monitoring Mode",
            options=[
                "All Alerts",
                "Forest Protection Only",
                "Wildlife Road Safety Only",
            ],
            index=0,
        )
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.25,
            max_value=0.85,
            value=0.45,
            step=0.05,
        )

        st.divider()
        st.subheader("Model & Monitored Classes")
        st.markdown("**Model:** `YOLOv8n` (`yolov8n.pt`)")
        st.markdown("**Forest Protection (`person`):** Red box → *Forest Alert*")
        st.markdown(
            "**Wildlife Road Safety:** Orange box → *Road Warning*  \n"
            "*Classes:* `bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, "
            "`elephant`, `bear`, `zebra`, `giraffe`"
        )

    uploaded_file = st.file_uploader(
        "Upload an image for analysis",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("Upload a **JPG**, **JPEG**, or **PNG** image above to run WildGuard AI detection.")
    else:
        pil_image = Image.open(uploaded_file)
        model = load_model()

        annotated_img, detections, road_animals = process_image(
            pil_image=pil_image,
            model=model,
            mode=mode,
            confidence=confidence,
        )

        process_key = f"{uploaded_file.name}_{uploaded_file.size}_{mode}_{confidence:.2f}"
        if st.session_state.last_processed_key != process_key:
            if detections:
                st.session_state.detection_log.extend(detections)
            st.session_state.last_processed_key = process_key

        forest_count = sum(1 for d in detections if d["Alert Status"] == "Forest Alert")
        road_count = sum(1 for d in detections if d["Alert Status"] == "Road Warning")
        total_count = len(detections)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Detections", total_count)
        m2.metric("Forest Alerts", forest_count)
        m3.metric("Road Warnings", road_count)

        if forest_count > 0:
            st.error(
                f"🚨 **Forest Alert:** Detected {forest_count} person(s) in the monitored zone. "
                "Verify authorization and log potential intrusion."
            )

        if road_count > 0:
            st.warning(f"⚠️ **Road Warning:** Detected {road_count} wildlife object(s) near roadway.")
            st.markdown(
                f'<div class="advisory-box"><strong>Road Safety Advisory:</strong> '
                f"{get_road_advisory(road_animals)}</div>",
                unsafe_allow_html=True,
            )

        col_orig, col_res = st.columns(2)
        with col_orig:
            st.subheader("Original Image")
            st.image(pil_image, use_container_width=True)
        with col_res:
            st.subheader("Annotated Result")
            st.image(annotated_img, use_container_width=True)

        png_bytes = encode_png_bytes(annotated_img)
        st.download_button(
            label="Download Annotated Image (PNG)",
            data=png_bytes,
            file_name=f"wildguard_{uploaded_file.name.rsplit('.', 1)[0]}.png",
            mime="image/png",
        )

    st.divider()
    log_header_col, log_btn_col = st.columns([4, 1])
    with log_header_col:
        st.subheader("Session Detection Log")
    with log_btn_col:
        if st.button("Clear Log", use_container_width=True):
            st.session_state.detection_log = []
            st.session_state.last_processed_key = None
            st.rerun()

    if st.session_state.detection_log:
        st.dataframe(
            st.session_state.detection_log,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No detections logged yet in this session.")


if __name__ == "__main__":
    main()