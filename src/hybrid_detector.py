"""
하이브리드 탐지기 (Hybrid Detector)
- FMO (배경 차분): 빠른 공(잔상) 탐지에 강함
- Color (HSV): 느린 공, 정지 상태 탐지에 강함
두 알고리즘을 병렬 실행하여 상황에 맞는 최적 결과 반환
"""

import cv2
import numpy as np
from collections import deque


class HybridDetector:
    def __init__(self, color_lower, color_upper):
        """
        Args:
            color_lower: HSV 하한값 (튜플)
            color_upper: HSV 상한값 (튜플)
        """
        # === FMO (배경 차분) 설정 ===
        self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
            history=300,
            dist2Threshold=400.0,
            detectShadows=False  # 그림자 감지 끔 (연산량 절감)
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # === Color (HSV) 설정 ===
        self.lower_color = np.array(color_lower)
        self.upper_color = np.array(color_upper)
        
        # === 상태 관리 ===
        self.last_valid_color_radius = 0  # FMO radius 보정용
        self.pts = deque(maxlen=64)       # 2D 궤적 (시각화용)
        self.pts_3d = deque(maxlen=64)    # 3D 궤적 (시각화용)
        
        # === 신뢰도 임계값 ===
        self.fmo_score_threshold = 500    # FMO 신뢰도 임계값
        self.min_area = 30                # 최소 컨투어 면적
        self.max_area = 5000              # 최대 컨투어 면적

    def _detect_by_color(self, frame):
        """
        Color(HSV) 기반 탐지
        
        Returns:
            tuple: ((x, y), radius, score) 또는 (None, 0, 0)
        """
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_result = None
        best_score = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area < area < self.max_area:
                ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                
                # 원형도 계산 (1에 가까울수록 원형)
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
                
                # Color는 원형에 가까울수록 높은 점수
                score = area * circularity
                
                if score > best_score and circularity > 0.5:
                    best_score = score
                    best_result = ((int(x), int(y)), radius)
        
        if best_result:
            return best_result[0], best_result[1], best_score
        return None, 0, 0

    def _detect_by_fmo(self, frame):
        """
        FMO(배경 차분) 기반 탐지 - 빠른 움직임(잔상) 감지
        
        Returns:
            tuple: ((x, y), radius, score) 또는 (None, 0, 0)
        """
        mask_fmo = self.bg_subtractor.apply(frame)
        mask_fmo = cv2.morphologyEx(mask_fmo, cv2.MORPH_OPEN, self.kernel)
        
        contours, _ = cv2.findContours(mask_fmo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_result = None
        best_score = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < self.max_area:
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (w, h), angle = rect
                
                minor_axis = min(w, h)
                major_axis = max(w, h)
                aspect_ratio = major_axis / (minor_axis + 1e-6)
                
                # FMO는 길쭉한 형상(잔상)에 가산점
                if aspect_ratio > 1.2:
                    score = area * aspect_ratio
                    
                    if score > best_score:
                        best_score = score
                        # 반지름: 이전 Color radius 사용 (더 정확)
                        # FMO 잔상은 radius가 부정확하므로 보정
                        if self.last_valid_color_radius > 0:
                            radius = self.last_valid_color_radius
                        else:
                            radius = minor_axis / 2
                        best_result = ((int(cx), int(cy)), radius)
        
        if best_result:
            return best_result[0], best_result[1], best_score
        return None, 0, 0

    def detect(self, frame):
        """
        하이브리드 탐지 메인 메서드
        Color와 FMO를 모두 실행하고 최적 결과 반환
        
        Args:
            frame: BGR 이미지
            
        Returns:
            tuple: (center, radius, method)
                - center: (x, y) 또는 None
                - radius: float
                - method: "Color", "FMO", "FMO_Weak", "None"
        """
        # === A. Color 탐지 (항상 실행) ===
        color_center, color_radius, color_score = self._detect_by_color(frame)
        
        # Color 성공 시 radius 저장 (FMO 보정용)
        if color_center and color_radius > 0:
            self.last_valid_color_radius = color_radius
        
        # === B. FMO 탐지 (항상 실행 - 배경 모델 업데이트 필요) ===
        fmo_center, fmo_radius, fmo_score = self._detect_by_fmo(frame)
        
        # === C. 결과 융합 (Decision Logic) ===
        final_center = None
        final_radius = 0
        used_method = "None"
        
        # 우선순위 결정:
        # 1. FMO 점수가 높으면 (빠른 공) → FMO 사용
        # 2. Color가 있으면 → Color 사용 (더 안정적)
        # 3. FMO만 있으면 → 약한 FMO라도 사용
        
        if fmo_center and fmo_score > self.fmo_score_threshold:
            # FMO 신뢰도가 높음 (빠른 움직임 감지)
            final_center = fmo_center
            final_radius = fmo_radius
            used_method = "FMO"
        elif color_center:
            # FMO가 없거나 약하면 Color 사용
            final_center = color_center
            final_radius = color_radius
            used_method = "Color"
        elif fmo_center:
            # Color도 없으면 약한 FMO라도 사용
            final_center = fmo_center
            final_radius = fmo_radius
            used_method = "FMO_Weak"
        
        # 궤적 저장
        if final_center:
            self.pts.appendleft(final_center)
        
        return final_center, final_radius, used_method

    def draw_ball(self, frame, center, radius, method=""):
        """
        감지된 공 시각화
        
        Args:
            frame: 그릴 프레임
            center: (x, y) 중심점
            radius: 반지름
            method: 사용된 탐지 방법 (디버그용)
        """
        if center is None:
            return
        
        # 방법에 따라 색상 구분
        if method == "FMO":
            color = (0, 0, 255)    # 빨강 (빠른 공)
        elif method == "FMO_Weak":
            color = (0, 165, 255)  # 주황 (약한 FMO)
        else:
            color = (0, 255, 255)  # 노랑 (Color)
        
        # 공 그리기
        cv2.circle(frame, center, int(radius), color, 2)
        cv2.circle(frame, center, 3, (255, 0, 0), -1)  # 중심점
        
        # 방법 표시
        if method:
            cv2.putText(frame, method, (center[0] + 15, center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def draw_trajectory(self, frame, color=(0, 255, 0)):
        """
        2D 궤적 그리기
        """
        for i in range(1, len(self.pts)):
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue
            thickness = int(np.sqrt(64 / float(i + 1)) * 2)
            cv2.line(frame, self.pts[i - 1], self.pts[i], color, thickness)

    def reset(self):
        """
        상태 초기화
        """
        self.pts.clear()
        self.pts_3d.clear()
        self.last_valid_color_radius = 0
        # 배경 모델은 유지 (재학습에 시간이 걸리므로)
