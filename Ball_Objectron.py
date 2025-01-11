import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # BGR -> HSV 변환   
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    greenLower = (29, 86, 6)
    greenUpper = (64, 255, 255)
    # 1) 공 색상에 맞는 범위를 설정해야 함 (예: 흰색 공)
    #    환경에 맞춰 lower/upper를 조절하세요.
    lower_white = np.array([29, 86, 6])
    upper_white = np.array([64, 255, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 노이즈 제거 (침식 & 팽창)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # 컨투어(윤곽선) 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # **가장 큰 컨투어(면적)만 추적**하기
    if contours:
        max_contour = max(contours, key=cv2.contourArea)  # 면적이 가장 큰 컨투어
        
        # 최소 외접원(원의 중심, 반지름) 구하기
        (x, y), radius = cv2.minEnclosingCircle(max_contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # 예: 최소 면적을 정해 가짜 물체(노이즈) 제거
        if radius < 10:
            # 추적용(시각화)으로 원 그리기
            cv2.circle(frame, center, radius, (0, 255, 0), 2)

    cv2.imshow('Single Ball Tracking - Largest Contour', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
