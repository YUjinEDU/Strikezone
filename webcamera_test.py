# filepath: /c:/Users/Yujin/OneDrive - 한밭대학교/한밭대학교/3학년1학기/지능영상처리/Strikezone/web_tesst.py
from flask import Flask, Response, render_template
import cv2

app = Flask(__name__)

def generate_frames():
    cap = cv2.VideoCapture(2)  # 카메라 장치 ID (0은 기본 카메라)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)