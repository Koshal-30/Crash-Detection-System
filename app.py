# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from flask import Flask, request, jsonify, render_template
# import tempfile

# app = Flask(__name__)

# # ── Config (match your training settings exactly) ──────────────────────────
# SEQ_LEN       = 16
# IMG_SIZE      = 112
# CLIPS_PER_VID = 3

# # ── Path to your saved model ───────────────────────────────────────────────
# # Update this to wherever you store the .keras / .h5 file
# MODEL_PATH = os.environ.get("MODEL_PATH", "ccd_cnn_lstm_model.keras")

# # Max upload size: 200 MB
# app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

# # ── Load model once at startup ─────────────────────────────────────────────
# model = None

# def load_model():
#     global model
#     print(f"[INFO] Loading model from: {MODEL_PATH}")
#     model = tf.keras.models.load_model(MODEL_PATH)
#     print("[INFO] Model ready.")

# # ── Preprocessing (mirrors your training pipeline, in-memory) ─────────────
# def extract_clips(video_path, num_clips=CLIPS_PER_VID):
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     if total_frames < SEQ_LEN:
#         cap.release()
#         return []

#     positions = np.linspace(0, total_frames - SEQ_LEN, num_clips, dtype=int)

#     # Only read the frames we actually need
#     needed = set()
#     for start in positions:
#         for i in range(SEQ_LEN):
#             needed.add(start + i)

#     frames_dict = {}
#     current_frame = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if current_frame in needed:
#             frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
#             frames_dict[current_frame] = frame.astype(np.float32) / 255.0
#         if current_frame > max(needed):
#             break
#         current_frame += 1

#     cap.release()

#     clips = []
#     for start in positions:
#         clip = np.array([frames_dict[start + i] for i in range(SEQ_LEN)
#                          if (start + i) in frames_dict])
#         if len(clip) == SEQ_LEN:
#             clips.append(clip)

#     return clips  # list of (16, 112, 112, 3) arrays


# # ── Routes ─────────────────────────────────────────────────────────────────
# @app.route("/")
# def index():
#     return render_template("index.html")


# @app.route("/predict", methods=["POST"])
# def predict():
#     if "video" not in request.files:
#         return jsonify({"error": "No video file provided."}), 400

#     video_file = request.files["video"]

#     # Validate extension
#     allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
#     ext = os.path.splitext(video_file.filename)[1].lower()
#     if ext not in allowed:
#         return jsonify({"error": f"Unsupported format '{ext}'. Use mp4, avi, or mov."}), 400

#     # Save to a temp file (OpenCV needs a real path)
#     suffix = ext if ext else ".mp4"
#     with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
#         video_file.save(tmp.name)
#         tmp_path = tmp.name

#     try:
#         clips = extract_clips(tmp_path)

#         if not clips:
#             return jsonify({
#                 "error": "Video is too short to analyse (needs at least 16 frames)."
#             }), 422

#         # Run inference on each clip
#         clip_scores = []
#         for clip in clips:
#             score = float(model.predict(clip[np.newaxis], verbose=0)[0][0])
#             clip_scores.append(score)

#         # Video-level decision: average clip scores (same as your evaluation logic)
#         avg_score = float(np.mean(clip_scores))
#         label     = "CRASH" if avg_score > 0.5 else "NORMAL"
#         confidence = avg_score if avg_score > 0.5 else (1.0 - avg_score)

#         return jsonify({
#             "prediction":     label,
#             "confidence":     round(confidence * 100, 1),
#             "raw_score":      round(avg_score, 4),
#             "clips_analysed": len(clips),
#             "clip_scores":    [round(s, 4) for s in clip_scores],
#         })

#     except Exception as e:
#         return jsonify({"error": f"Processing failed: {str(e)}"}), 500

#     finally:
#         if os.path.exists(tmp_path):
#             os.unlink(tmp_path)


# # ── Entry point ────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     load_model()
#     app.run(host="0.0.0.0", port=5000, debug=False)
import os
import shutil
import tempfile
from urllib.parse import urlparse

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

try:
    import yt_dlp
except ImportError:
    yt_dlp = None


app = Flask(__name__)

# Config (match your training settings exactly)
SEQ_LEN = 16
IMG_SIZE = 112
CLIPS_PER_VID = 3

# Path to your saved model
MODEL_PATH = os.environ.get("MODEL_PATH", "ccd_cnn_lstm_model.keras")

# Max upload size: 200 MB
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "www.youtu.be",
}

model = None


def load_model():
    global model
    if model is None:
        print(f"[INFO] Loading model from: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("[INFO] Model ready.")


def is_youtube_url(url):
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and parsed.netloc.lower() in YOUTUBE_HOSTS


def download_youtube_video(video_url):
    if yt_dlp is None:
        raise RuntimeError(
            "yt-dlp is not installed. Install it with 'pip install yt-dlp'."
        )

    temp_dir = tempfile.mkdtemp(prefix="ccd_yt_")
    output_template = os.path.join(temp_dir, "source.%(ext)s")
    ydl_opts = {
        "outtmpl": output_template,
        "format": "mp4/bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        downloaded_path = ydl.prepare_filename(info)

    base_path, _ = os.path.splitext(downloaded_path)
    merged_mp4_path = base_path + ".mp4"

    if os.path.exists(merged_mp4_path):
        downloaded_path = merged_mp4_path

    return downloaded_path, temp_dir


def extract_clips(video_path, num_clips=CLIPS_PER_VID):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < SEQ_LEN:
        cap.release()
        return []

    positions = np.linspace(0, total_frames - SEQ_LEN, num_clips, dtype=int)
    needed = set()
    for start in positions:
        for i in range(SEQ_LEN):
            needed.add(start + i)

    frames_dict = {}
    current_frame = 0
    max_needed = max(needed)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame in needed:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frames_dict[current_frame] = frame.astype(np.float32) / 255.0
        if current_frame > max_needed:
            break
        current_frame += 1

    cap.release()

    clips = []
    for start in positions:
        clip = np.array(
            [frames_dict[start + i] for i in range(SEQ_LEN) if (start + i) in frames_dict]
        )
        if len(clip) == SEQ_LEN:
            clips.append(clip)

    return clips


def run_prediction(video_path):
    load_model()
    clips = extract_clips(video_path)

    if not clips:
        return {
            "error": "Video is too short to analyse (needs at least 16 frames)."
        }, 422

    clip_scores = []
    for clip in clips:
        score = float(model.predict(clip[np.newaxis], verbose=0)[0][0])
        clip_scores.append(score)

    avg_score = float(np.mean(clip_scores))
    label = "CRASH" if avg_score > 0.5 else "NORMAL"
    confidence = avg_score if avg_score > 0.5 else (1.0 - avg_score)

    return {
        "prediction": label,
        "confidence": round(confidence * 100, 1),
        "raw_score": round(avg_score, 4),
        "clips_analysed": len(clips),
        "clip_scores": [round(s, 4) for s in clip_scores],
    }, 200


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    tmp_path = None
    temp_dir = None

    try:
        video_url = request.form.get("video_url", "").strip()
        video_file = request.files.get("video")

        if video_url:
            if not is_youtube_url(video_url):
                return jsonify(
                    {
                        "error": "Please enter a valid YouTube video URL."
                    }
                ), 400
            tmp_path, temp_dir = download_youtube_video(video_url)

        elif video_file and video_file.filename:
            ext = os.path.splitext(video_file.filename)[1].lower()
            if ext not in ALLOWED_VIDEO_EXTENSIONS:
                return jsonify(
                    {
                        "error": f"Unsupported format '{ext}'. Use mp4, avi, mov, mkv, or webm."
                    }
                ), 400

            suffix = ext if ext else ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                video_file.save(tmp.name)
                tmp_path = tmp.name

        else:
            return jsonify(
                {
                    "error": "Provide either a local video file or a YouTube video URL."
                }
            ), 400

        result, status_code = run_prediction(tmp_path)
        return jsonify(result), status_code

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False)