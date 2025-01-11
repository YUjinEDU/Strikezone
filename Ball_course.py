from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# 명령줄 인자 파서를 구성하고 인자를 파싱합니다.
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")

ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
ap.add_argument("-u", "--url", type=str, default="http://172.30.1.18:4747/mjpegfeed",
                help="URL of the DroidCam video feed")


args = vars(ap.parse_args())

# HSV 색상 공간에서 "녹색" 공의 하한과 상한을 정의한 다음
# 추적할 점들의 목록을 초기화합니다.
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

pts = deque(maxlen=args["buffer"])

# 비디오 경로가 제공되지 않은 경우 웹캠을 참조합니다.
if not args.get("video", False):
    # DrodiCam : 'http://172.30.1.18:4747/mjpegfeed'
    vs = VideoStream(src=0).start()
    # 웹캠에서 비디오 스트림을 시작합니다.
else:
    vs = cv2.VideoCapture(args["video"])
    # 비디오 파일에서 비디오 스트림을 시작합니다.

# 카메라 또는 비디오 파일이 예열되도록 2초 동안 대기합니다.
time.sleep(2.0)

# 무한 루프를 통해 프레임을 계속 읽습니다.
while True:
    # 현재 프레임을 가져옵니다.
    frame = vs.read()
    # 비디오 파일에서 프레임을 읽을 경우, 프레임의 첫 번째 요소를 가져옵니다.
    frame = frame[1] if args.get("video", False) else frame
    # 프레임을 가져오지 못한 경우, 비디오가 끝난 것입니다.
    if frame is None:
        break
    # 프레임을 리사이즈하고 블러 처리한 다음 HSV 색상 공간으로 변환합니다.
    frame = imutils.resize(frame, width=1000)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # "녹색" 색상에 대한 마스크를 생성한 다음, 마스크에 대해 여러 번 팽창 및 침식을 수행하여
    # 남아 있는 작은 블롭들을 제거합니다.
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # 마스크에서 윤곽선을 찾아 현재 공의 (x, y) 중심을 초기화합니다.
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    # 최소한 하나의 윤곽선이 발견된 경우에만 진행합니다.
    if len(cnts) > 0:
        # 마스크에서 가장 큰 윤곽선을 찾아 최소한의 둘레 원과 중심을 계산합니다.
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # 반지름이 최소 크기를 만족하는 경우에만 진행합니다.
        if radius > 10:
            # 프레임에 원과 중심을 그린 다음, 추적할 점들의 목록을 업데이트합니다.
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    # 점들의 큐를 업데이트합니다.
    pts.appendleft(center)
    # 추적할 점들의 집합을 반복합니다.
    for i in range(1, len(pts)):
        # 추적할 점들 중 하나라도 None인 경우 무시합니다.
        if pts[i - 1] is None or pts[i] is None:
            continue
        # 두 점 사이의 선의 두께를 계산하고 선을 그립니다.
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
    # 화면에 프레임을 보여줍니다.
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # 'q' 키가 눌리면 루프를 중지합니다.
    if key == ord("q"):
        break

# 비디오 파일을 사용하지 않는 경우, 카메라 비디오 스트림을 중지합니다.
if not args.get("video", False):
    vs.stop()
# 비디오 파일을 사용하는 경우, 카메라를 해제합니다.
else:
    vs.release()
# 모든 창을 닫습니다.
cv2.destroyAllWindows()
