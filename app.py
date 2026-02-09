from flask import Flask, render_template, request, Response
import tensorflow as tf
import numpy as np
import os
import cv2

app = Flask(__name__)

model = tf.keras.models.load_model("model/drone_model.h5")

def predict_image_array(img_array):
    img_array = cv2.resize(img_array, (224, 224))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        return "Drone Detected", prediction
    else:
        return "No Drone Detected", prediction

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files["file"]
    os.makedirs("static/uploads", exist_ok=True)
    filepath = os.path.join("static/uploads", file.filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    result, confidence = predict_image_array(img)

    return render_template("result.html",
                           result=result,
                           confidence=round(float(confidence), 2),
                           image_path=filepath)

@app.route("/video-upload")
def video_upload():
    return render_template("video_upload.html")

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0,
                          (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result, confidence = predict_image_array(frame)

        cv2.putText(frame,
                    f"{result} ({round(float(confidence),2)})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        out.write(frame)

    cap.release()
    out.release()

@app.route("/process-video", methods=["POST"])
def process_video_route():
    video = request.files["video"]
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("static/processed", exist_ok=True)

    input_path = os.path.join("static/uploads", video.filename)
    output_path = os.path.join("static/processed", "processed_" + video.filename)

    video.save(input_path)
    process_video(input_path, output_path)

    return render_template("result.html",
                           result="Video Processed Successfully",
                           confidence="",
                           image_path=output_path)

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break

        result, confidence = predict_image_array(frame)

        cv2.putText(frame,
                    f"{result} ({round(float(confidence),2)})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/live")
def live():
    return render_template("live.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
