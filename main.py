import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import tempfile
import os
import socket
import logging
from pathlib import Path
import av
from streamlit.components.v1 import html as components_html
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# --- Configuration ---
MODEL_PATHS = {
    "M1 (Model 1)": "./RoadDetectionModel/RoadModel_yolov8m.pt_rounds120_b9/weights/best.pt",
    "M2 (Model 2)": "./YOLOv8_Small_2nd_Model.pt",
}
MODEL_PREFIX = {
    "M1 (Model 1)": "M1",
    "M2 (Model 2)": "M2",
}
DEFAULT_CONF = {"M1 (Model 1)": 0.35, "M2 (Model 2)": 0.40}
LIVE_FEED_TARGET_WIDTH = 480
LIVE_FEED_IMGSZ = 320
# Fixed 320px OpenVINO build of Model 1 — measured ~30% faster than PyTorch on this CPU
# (folder name must keep the `_openvino_model` suffix — ultralytics detects the format by it)
LIVE_M1_OV_PATH = "openvino_models/M1_fixed320_openvino_model"

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

CSS = """
<style>
.stApp { background: #f4f6fb; }
.block-container { max-width: 780px; padding: 1.2rem 1rem 3rem; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { visibility: hidden; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

.hero {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 45%, #c0392b 100%);
    border-radius: 22px; padding: 2rem 1.2rem 1.6rem;
    text-align: center; color: #fff; margin-bottom: 1.4rem;
}
.hero h1 { font-size: 1.9rem; margin: .4rem 0 .2rem; font-weight: 800; letter-spacing: .3px; }
.hero p  { margin: 0; opacity: .92; font-size: 1rem; }
.hero-badges { margin-top: .9rem; font-size: .9rem; opacity: .85; }

.stButton > button {
    width: 100%; height: 62px; font-size: 1.15rem; font-weight: 700;
    border-radius: 16px; border: none;
    box-shadow: 0 2px 10px rgba(15, 32, 39, .14);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #e74c3c, #c0392b); color: #fff;
}
.stButton > button[kind="secondary"] {
    background: #ffffff; color: #203a43; border: 1.5px solid #d5deee;
}

[data-testid="stFileUploaderDropzone"] {
    border-radius: 16px; border: 2px dashed #9db2d8; background: #ffffff;
}

.chip {
    display: inline-block; padding: .35rem .85rem; margin: .25rem .3rem .25rem 0;
    border-radius: 999px; background: #e8eeff; color: #203a43;
    font-weight: 600; border: 1px solid #c9d6ff; font-size: .9rem;
}
.result-img {
    border-radius: 18px; overflow: hidden;
    box-shadow: 0 4px 18px rgba(15, 32, 39, .12);
}
.page-footer { text-align: center; opacity: .65; font-size: .85rem; margin-top: 2rem; }
</style>
"""


@st.cache_resource
def load_yolo_model(path: str):
    try:
        model = YOLO(path)
        logger.info(f"Successfully loaded model from {path}")
        return model, model.names
    except Exception as e:
        logger.error(f"Failed to load model at {path}", exc_info=e)
        return None, {}


def make_annotators(color: sv.Color):
    box_annotator = sv.BoxAnnotator(thickness=2, color=color)
    label_annotator = sv.LabelAnnotator(
        text_thickness=1,
        text_scale=0.45,
        color=sv.Color.WHITE,
        text_color=sv.Color.BLACK,
        text_padding=3,
    )
    return box_annotator, label_annotator


def process_frame(
    frame: np.ndarray, models: dict, thresholds: dict, imgsz: int | None = None
) -> tuple[np.ndarray, dict]:
    annotated_frame = frame.copy()
    counts: dict[str, int] = {}
    for model_name, (model, names_map, box_ann, label_ann) in models.items():
        try:
            predict_kwargs = {"conf": thresholds[model_name], "verbose": False}
            if imgsz is not None:
                predict_kwargs["imgsz"] = imgsz
            results = model.predict(frame, **predict_kwargs)[0]
            detections = sv.Detections.from_ultralytics(results)
            labels = [
                f"{MODEL_PREFIX[model_name]}:{names_map.get(int(cls_id), str(cls_id))} {conf:.2f}"
                for cls_id, conf in zip(detections.class_id, detections.confidence)
            ]
            annotated_frame = box_ann.annotate(annotated_frame, detections)
            annotated_frame = label_ann.annotate(annotated_frame, detections, labels=labels)
            for cls_id in detections.class_id:
                cls = names_map.get(int(cls_id), str(cls_id))
                key = f"{MODEL_PREFIX[model_name]} · {cls}"
                counts[key] = counts.get(key, 0) + 1
        except Exception as e:
            logger.error(f"Error during prediction/annotation for {model_name}: {e}")
            cv2.putText(
                annotated_frame,
                f"Error processing {model_name}",
                (10, 30 + list(models.keys()).index(model_name) * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
    return annotated_frame, counts


def show_counts(counts: dict):
    st.subheader("🔎 Detections")
    if not counts:
        st.info(
            "No road anomalies detected. Try lowering the confidence sliders "
            "in ⚙️ Advanced Settings, or take a clearer photo."
        )
        return
    chips = "".join(
        f'<span class="chip">{name}: {count}</span>' for name, count in sorted(counts.items())
    )
    st.markdown(f'<div>{chips}</div>', unsafe_allow_html=True)


def decode_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def force_native_camera_capture():
    """Makes the phone open the rear camera directly when the uploader is tapped
    (browsers only allow live webcam over HTTPS; the native camera app works over HTTP)."""
    components_html(
        """
        <script>
        (function () {
            const apply = () => {
                try {
                    window.parent.document
                        .querySelectorAll('input[type=file]')
                        .forEach((el) => el.setAttribute('capture', 'environment'));
                } catch (e) {}
            };
            apply();
            new MutationObserver(apply).observe(
                window.parent.document.body, { childList: true, subtree: true }
            );
        })();
        </script>
        """,
        height=0,
    )


def _set_mode(mode):
    def cb():
        st.session_state.mode = mode

    return cb


def back_home_button(key: str):
    st.button("⬅️ Back to Home", key=key, on_click=_set_mode("home"), use_container_width=True)


def view_home(loaded_models, thresholds):
    st.markdown(
        """
        <div class="hero">
            <div style="font-size:3rem">🛣️</div>
            <h1>Road Anomaly Detection</h1>
            <p>AI that spots cracks, potholes &amp; speed bumps from a simple photo</p>
            <div class="hero-badges">🧠 2 models &nbsp;·&nbsp; 🛣️ 7 road classes &nbsp;·&nbsp; 📱 phone-ready</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.button(
        "📷 Start Camera",
        type="primary",
        key="go_camera",
        on_click=_set_mode("camera"),
        use_container_width=True,
    )
    st.button(
        "📹 Live Camera",
        type="primary",
        key="go_live",
        on_click=_set_mode("live"),
        use_container_width=True,
    )
    st.button("🖼️ Upload Image", key="go_upload", on_click=_set_mode("upload"), use_container_width=True)
    st.button("🎥 Upload Video", key="go_video", on_click=_set_mode("video"), use_container_width=True)

    st.markdown("### How it works")
    st.markdown(
        """
        1. 📷 **Start Camera** and take a photo of the road (or upload one)
        2. 🧠 The AI models scan it instantly
        3. 🔎 See every crack, pothole and speed bump highlighted
        """
    )

    with st.expander("📱 Open on your phone"):
        st.write("Make sure your phone is on the **same Wi-Fi** as this computer, then open:")
        st.code(f"https://{get_local_ip()}:8501", language=None)
        st.caption(
            "First visit: your browser will warn about the certificate — tap "
            "**Advanced → Proceed / Visit site** (this is normal for a local project). "
            "The secure connection is what lets the **Live Camera** work on your phone."
        )

    with st.expander("⚙️ Advanced Settings"):
        build_model_controls()


def view_camera(loaded_models, thresholds):
    top_bar()
    st.markdown("#### 📷 Take a photo of the road")
    st.caption("Tap the box below — your camera opens. Snap the road and detection runs instantly.")

    img_file = st.file_uploader(
        "Camera",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="cam_upload",
        label_visibility="collapsed",
    )
    force_native_camera_capture()

    if img_file is not None:
        img = decode_uploaded_image(img_file)
        if img is None:
            st.error("Could not read that photo. Please try again.")
        else:
            with st.spinner("Scanning for anomalies..."):
                processed, counts = process_frame(img, loaded_models, thresholds)
            show_counts(counts)
            st.markdown('<div class="result-img">', unsafe_allow_html=True)
            st.image(processed, channels="BGR", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.button(
                "🔄 Take another photo",
                key="retake",
                on_click=lambda: st.session_state.pop("cam_upload", None),
                use_container_width=True,
            )

    st.markdown("")
    back_home_button("back_camera")


def view_upload(loaded_models, thresholds):
    top_bar()
    st.markdown("#### 🖼️ Upload a road image")
    img_file = st.file_uploader(
        "Image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="img_upload",
        label_visibility="collapsed",
    )
    if img_file is not None:
        img = decode_uploaded_image(img_file)
        if img is None:
            st.error("Could not read that image. Please try another file.")
        else:
            with st.spinner("Scanning for anomalies..."):
                processed, counts = process_frame(img, loaded_models, thresholds)
            show_counts(counts)
            st.markdown('<div class="result-img">', unsafe_allow_html=True)
            st.image(processed, channels="BGR", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Pick an image to scan — JPG, PNG, BMP or WEBP.")
    st.markdown("")
    back_home_button("back_upload")


def view_video(loaded_models, thresholds):
    top_bar()
    st.markdown("#### 🎥 Upload a road video")
    st.caption("Processing runs on this computer's CPU — short clips work best.")

    vid_file = st.file_uploader(
        "Video", type=["mp4", "avi", "mov", "mkv"], key="vid_upload", label_visibility="collapsed"
    )
    if vid_file is not None:
        input_tmp_path = None
        output_path = None
        cap = None
        writer = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_tmp:
                input_tmp.write(vid_file.read())
                input_tmp_path = input_tmp.name

            cap = cv2.VideoCapture(input_tmp_path)
            if not cap.isOpened():
                st.error("Could not open that video file.")
                return

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0:
                fps = 30

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as output_tmp:
                output_path = output_tmp.name
            writer = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writer.isOpened():
                st.error("Could not initialize the video writer.")
                return

            prog_bar = st.progress(0, text="Processing video...")
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out_frame, _ = process_frame(frame, loaded_models, thresholds)
                writer.write(out_frame)
                frame_idx += 1
                prog_bar.progress(
                    min(frame_idx / total_frames, 1.0) if total_frames > 0 else 0,
                    text=f"Processing video... {frame_idx}/{total_frames or '?'}",
                )
            prog_bar.progress(1.0, text="Done!")
        except Exception as e:
            st.error(f"Something went wrong while processing the video: {e}")
            logger.error("Video processing failed", exc_info=e)
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            if input_tmp_path and os.path.exists(input_tmp_path):
                try:
                    os.remove(input_tmp_path)
                except OSError:
                    pass

        if output_path and os.path.exists(output_path) and frame_idx > 0:
            with open(output_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)
            st.download_button(
                "⬇️ Download Processed Video",
                data=video_bytes,
                file_name=f"processed_{vid_file.name}",
                mime="video/mp4",
                key="vid_download",
                use_container_width=True,
            )
    else:
        st.info("Pick a video to scan — MP4, AVI, MOV or MKV.")
    st.markdown("")
    back_home_button("back_video")


def top_bar():
    st.markdown("### 🛣️ Road Anomaly Detection")
    st.markdown("")


class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self, models, thresholds, target_width):
        self.models = models
        self.thresholds = thresholds
        self.target_width = target_width

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        original_height, original_width = img.shape[:2]
        img_resized = img
        if self.target_width is not None and original_width > self.target_width:
            aspect_ratio = original_height / original_width
            target_height = int(self.target_width * aspect_ratio)
            img_resized = cv2.resize(
                img, (self.target_width, target_height), interpolation=cv2.INTER_AREA
            )
        annotated_resized, _ = process_frame(
            img_resized, self.models, self.thresholds, imgsz=LIVE_FEED_IMGSZ
        )
        if img_resized is not img:
            final_frame = cv2.resize(
                annotated_resized,
                (original_width, original_height),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            final_frame = annotated_resized
        return av.VideoFrame.from_ndarray(final_frame, format="bgr24")


def build_live_stack():
    """Live mode uses a leaner model stack: Model 1 as a fixed-320 OpenVINO model,
    with Model 2 opt-in (it roughly doubles the per-frame time)."""
    m1_path = LIVE_M1_OV_PATH if Path(LIVE_M1_OV_PATH).exists() else MODEL_PATHS["M1 (Model 1)"]
    live_paths = {"M1 (Model 1)": m1_path}
    if st.checkbox("Also run Model 2 (slower)", value=False, key="use_m2_live"):
        live_paths["M2 (Model 2)"] = MODEL_PATHS["M2 (Model 2)"]

    live_models = {}
    live_thresholds = {}
    for name, path in live_paths.items():
        model, names_map = load_yolo_model(path)
        color = sv.Color.RED if name == "M1 (Model 1)" else sv.Color.BLUE
        box_ann, label_ann = make_annotators(color)
        live_models[name] = (model, names_map, box_ann, label_ann)
        conf_key = "m1_conf" if name.startswith("M1") else "m2_conf"
        live_thresholds[name] = st.session_state.get(conf_key, DEFAULT_CONF[name])
    return live_models, live_thresholds


def view_live():
    top_bar()
    st.markdown("#### 📹 Live Camera")
    st.caption(
        "Allow camera access when your browser asks — detections appear in real time. "
        "Runs on the fast OpenVINO engine; add Model 2 below if you need its crack types."
    )

    live_models, live_thresholds = build_live_stack()

    camera_choice = st.radio(
        "Camera",
        ["📱 Rear camera (phone)", "💻 Laptop webcam"],
        horizontal=True,
        key="camera_choice",
    )
    use_rear = camera_choice.startswith("📱")

    video_constraints = {
        "width": {"ideal": 640},
        "height": {"ideal": 480},
        "frameRate": {"ideal": 15, "max": 20},
    }
    if use_rear:
        video_constraints["facingMode"] = {"exact": "environment"}

    def processor_factory():
        return YOLOVideoProcessor(
            models=live_models,
            thresholds=live_thresholds,
            target_width=LIVE_FEED_TARGET_WIDTH,
        )

    webrtc_streamer(
        key=f"live-camera-{'rear' if use_rear else 'laptop'}",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=processor_factory,
        media_stream_constraints={"video": video_constraints, "audio": False},
        async_processing=True,
        video_receiver_size=1,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )
    st.markdown("")
    back_home_button("back_live")


def build_model_controls():
    use_m1 = st.checkbox("Model 1 — Road defects (cracks, potholes…)", value=True, key="cb_m1")
    use_m2 = st.checkbox("Model 2 — Road damage (crack types, potholes…)", value=True, key="cb_m2")
    if not use_m1 and not use_m2:
        st.warning("Select at least one model.")
    if use_m1:
        st.slider("Model 1 confidence", 0.1, 1.0, DEFAULT_CONF["M1 (Model 1)"], 0.05, key="m1_conf")
    if use_m2:
        st.slider("Model 2 confidence", 0.1, 1.0, DEFAULT_CONF["M2 (Model 2)"], 0.05, key="m2_conf")


def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def load_models_with_controls():
    use_m1 = st.session_state.get("cb_m1", True)
    use_m2 = st.session_state.get("cb_m2", True)
    thresholds = {}
    if use_m1:
        thresholds["M1 (Model 1)"] = st.session_state.get("m1_conf", DEFAULT_CONF["M1 (Model 1)"])
    if use_m2:
        thresholds["M2 (Model 2)"] = st.session_state.get("m2_conf", DEFAULT_CONF["M2 (Model 2)"])

    loaded = {}
    failed = False
    for name in thresholds:
        model, names_map = load_yolo_model(MODEL_PATHS[name])
        if model and names_map:
            color = sv.Color.RED if name == "M1 (Model 1)" else sv.Color.BLUE
            box_ann, label_ann = make_annotators(color)
            loaded[name] = (model, names_map, box_ann, label_ann)
        else:
            failed = True

    if failed:
        st.error("One or more models failed to load. Check the model files and restart.")
        st.stop()
    if not loaded:
        st.warning("Select at least one model in ⚙️ Advanced Settings.")
        st.stop()
    return loaded, thresholds


def main():
    st.set_page_config(
        page_title="Road Anomaly Detection",
        page_icon="🛣️",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    if "mode" not in st.session_state:
        st.session_state.mode = "home"

    loaded_models, thresholds = load_models_with_controls()

    mode = st.session_state.mode
    if mode == "camera":
        view_camera(loaded_models, thresholds)
    elif mode == "live":
        view_live()
    elif mode == "upload":
        view_upload(loaded_models, thresholds)
    elif mode == "video":
        view_video(loaded_models, thresholds)
    else:
        view_home(loaded_models, thresholds)

    st.markdown(
        '<div class="page-footer">🛠️ Built as a personal project · Road Anomaly Detection 2026</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
