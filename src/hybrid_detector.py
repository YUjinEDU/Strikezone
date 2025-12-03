"""
하이브리드 탐지기 (Hybrid Detector)
- FMO (3프레임 메디안 배경): 빠른 공(잔상) 탐지에 강함 (FMO-cpp-demo 참고)
- Color (HSV): 느린 공, 정지 상태 탐지에 강함
두 알고리즘을 병렬 실행하여 상황에 맞는 최적 결과 반환

참고: https://github.com/rozumden/fmo-cpp-demo
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
        # === FMO (3프레임 메디안 배경) 설정 ===
        # FMO-cpp-demo 방식: 최근 3프레임의 메디안으로 배경 생성
        self.frame_buffer = deque(maxlen=3)  # 최근 3프레임 저장
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # FMO 탐지 파라미터 (fmo-cpp-demo의 Config 참고)
        self.diff_threshold = 25           # 차분 이진화 임계값
        self.min_strip_area = 0.05         # 최소 스트립 면적 비율
        self.min_aspect = 1.5              # 최소 종횡비 (잔상은 길쭉해야 함)
        
        # === Color (HSV) 설정 ===
        self.lower_color = np.array(color_lower)
        self.upper_color = np.array(color_upper)
        
        # === 상태 관리 ===
        self.last_valid_color_radius = 0  # FMO radius 보정용
        self.pts = deque(maxlen=64)       # 2D 궤적 (시각화용)
        self.pts_3d = deque(maxlen=64)    # 3D 궤적 (시각화용)
        
        # === 연속성 필터 (FMO-cpp-demo의 matchObjects 참고) ===
        self.last_center = None           # 마지막 감지 위치
        self.prev_centers = deque(maxlen=3)  # 최근 3프레임 중심점 (방향 검증용)
        self.max_jump_dist = 150          # 최대 점프 거리 (픽셀)
        self.lost_frames = 0              # 공을 못 찾은 연속 프레임 수
        self.max_lost_frames = 8          # 이 이상 못 찾으면 리셋
        
        # === 신뢰도 임계값 ===
        self.fmo_score_threshold = 500    # FMO 신뢰도 임계값
        self.min_area = 30                # 최소 컨투어 면적
        self.max_area = 3000              # 최대 컨투어 면적 (손 필터링: 5000→3000)
        
        # === FMO 토글 ===
        self.fmo_enabled = True           # FMO 활성화 여부
    
    def set_color_range(self, color_lower, color_upper):
        """
        색상 범위 동적 변경
        
        Args:
            color_lower: HSV 하한값 (튜플)
            color_upper: HSV 상한값 (튜플)
        """
        self.lower_color = np.array(color_lower)
        self.upper_color = np.array(color_upper)
        print(f"[HybridDetector] 색상 범위 변경: {color_lower} ~ {color_upper}")

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
        FMO(3프레임 메디안 배경) 기반 탐지 - 빠른 움직임(잔상) 감지
        FMO-cpp-demo의 MedianV1/V2 알고리즘 참고
        
        핵심 아이디어:
        1. 최근 3프레임의 메디안으로 배경 생성
        2. 현재 프레임과 배경의 차분으로 움직임 감지
        3. 길쭉한 형상(잔상)만 선택
        
        Returns:
            tuple: ((x, y), radius, score) 또는 (None, 0, 0)
        """
        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 프레임 버퍼에 추가
        self.frame_buffer.append(gray.copy())
        
        # 3프레임 미만이면 탐지 불가
        if len(self.frame_buffer) < 3:
            return None, 0, 0
        
        # === 3프레임 메디안 배경 생성 (FMO-cpp-demo 핵심) ===
        stacked = np.stack(list(self.frame_buffer), axis=2)
        background = np.median(stacked, axis=2).astype(np.uint8)
        
        # === 배경과 현재 프레임 차분 ===
        diff = cv2.absdiff(gray, background)
        
        # 이진화 (차분이 큰 픽셀만 추출)
        _, mask_fmo = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        
        # 노이즈 제거
        mask_fmo = cv2.morphologyEx(mask_fmo, cv2.MORPH_OPEN, self.kernel_small)
        mask_fmo = cv2.morphologyEx(mask_fmo, cv2.MORPH_CLOSE, self.kernel)
        
        contours, _ = cv2.findContours(mask_fmo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area < area < self.max_area:
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (w, h), angle = rect
                
                minor_axis = min(w, h)
                major_axis = max(w, h)
                aspect_ratio = major_axis / (minor_axis + 1e-6)
                
                # === 손 오탐지 방지: 크기 필터링 ===
                # 공의 minor_axis는 보통 10~60픽셀 범위
                # 손은 너무 크거나 (minor > 80) 너무 길쭉함 (aspect > 6)
                if minor_axis < 5 or minor_axis > 80:
                    continue
                
                # === FMO-cpp-demo의 minAspect 검증 ===
                # 잔상은 약간 길쭉해야 함, 하지만 너무 극단적이면 손일 가능성
                if self.min_aspect < aspect_ratio < 6.0:  # 상한 10→6으로 감소
                    # 점수 계산: 면적 * aspect_ratio (잔상 특성 보너스)
                    score = area * min(aspect_ratio, 4.0)
                    
                    # 반지름 결정 (minor_axis 기반)
                    if self.last_valid_color_radius > 0:
                        radius = self.last_valid_color_radius
                    else:
                        radius = minor_axis / 2
                    
                    # === 반지름 범위 검증 (공 크기: 5~40픽셀) ===
                    if radius < 5 or radius > 50:
                        continue
                    
                    # 방향 벡터 계산 (angle에서 추출)
                    direction = (np.cos(np.radians(angle)), np.sin(np.radians(angle)))
                    
                    candidates.append({
                        'center': (int(cx), int(cy)),
                        'radius': radius,
                        'score': score,
                        'area': area,
                        'aspect': aspect_ratio,
                        'direction': direction,
                        'length': major_axis
                    })
        
        if not candidates:
            return None, 0, 0
        
        # === 연속성 기반 후보 선택 (FMO-cpp-demo의 matchObjects 참고) ===
        if self.last_center is not None:
            for cand in candidates:
                dist = np.sqrt((cand['center'][0] - self.last_center[0])**2 + 
                              (cand['center'][1] - self.last_center[1])**2)
                if dist < self.max_jump_dist:
                    # 가까울수록 높은 가산점
                    proximity_bonus = 1 + (self.max_jump_dist - dist) / self.max_jump_dist
                    cand['score'] *= proximity_bonus
                    
                    # === 방향 일관성 검증 (FMO-cpp-demo의 matchAngleMax 참고) ===
                    if len(self.prev_centers) >= 2:
                        # 예상 이동 방향 계산
                        prev1 = self.prev_centers[-1]
                        prev2 = self.prev_centers[-2]
                        expected_dir = (prev1[0] - prev2[0], prev1[1] - prev2[1])
                        expected_len = np.sqrt(expected_dir[0]**2 + expected_dir[1]**2)
                        
                        if expected_len > 5:  # 충분한 이동이 있었을 때만
                            # 현재 이동 방향
                            current_dir = (cand['center'][0] - prev1[0], cand['center'][1] - prev1[1])
                            current_len = np.sqrt(current_dir[0]**2 + current_dir[1]**2)
                            
                            if current_len > 5:
                                # 방향 일관성 (코사인 유사도)
                                dot = expected_dir[0]*current_dir[0] + expected_dir[1]*current_dir[1]
                                cos_sim = dot / (expected_len * current_len + 1e-6)
                                
                                # 같은 방향이면 보너스 (cos_sim > 0)
                                if cos_sim > 0.3:
                                    cand['score'] *= (1 + cos_sim * 0.5)
                                # 반대 방향이면 페널티
                                elif cos_sim < -0.5:
                                    cand['score'] *= 0.3
                else:
                    # 너무 멀면 점수 대폭 감소
                    cand['score'] *= 0.05
        
        # 최고 점수 후보 선택
        best = max(candidates, key=lambda x: x['score'])
        
        return best['center'], best['radius'], best['score']

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
        
        # === B. FMO 탐지 (토글 상태에 따라 실행) ===
        fmo_center, fmo_radius, fmo_score = None, 0, 0
        if self.fmo_enabled:
            fmo_center, fmo_radius, fmo_score = self._detect_by_fmo(frame)
        else:
            # FMO 비활성화여도 프레임 버퍼는 업데이트 (추후 활성화 대비)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            self.frame_buffer.append(gray.copy())
        
        # === C. 결과 융합 (Decision Logic) ===
        final_center = None
        final_radius = 0
        used_method = "None"
        
        # 우선순위 결정:
        # 1. Color가 있으면 → Color 사용 (더 안정적, 색상 기반이라 정확)
        # 2. FMO 점수가 높으면 (빠른 공) → FMO 사용
        # 3. FMO만 있으면 → 약한 FMO라도 사용 (단, 연속성 검사 통과 시)
        
        if color_center:
            # Color가 최우선 (가장 안정적)
            final_center = color_center
            final_radius = color_radius
            used_method = "Color"
        elif fmo_center and fmo_score > self.fmo_score_threshold:
            # FMO 신뢰도가 높음 (빠른 움직임 감지)
            final_center = fmo_center
            final_radius = fmo_radius
            used_method = "FMO"
        elif fmo_center and fmo_score > self.fmo_score_threshold * 0.3:
            # 약한 FMO (연속성 검사 통과 시만 사용)
            if self._check_continuity(fmo_center):
                final_center = fmo_center
                final_radius = fmo_radius
                used_method = "FMO_Weak"
        
        # === D. 연속성 검사 및 점프 방지 ===
        if final_center is not None:
            if not self._check_continuity(final_center):
                # 연속성 실패: 이전 위치에서 너무 멀리 점프
                # lost_frames가 충분히 높으면 새 위치 수락 (공을 놓친 후 다시 찾음)
                if self.lost_frames < 5:
                    # 아직 연속 탐지 중인데 갑자기 점프 → 노이즈로 판단
                    final_center = None
                    final_radius = 0
                    used_method = "None"
        
        # === E. 상태 업데이트 ===
        if final_center is not None:
            self.prev_centers.append(final_center)  # 방향 검증용 히스토리
            self.last_center = final_center
            self.lost_frames = 0
            self.pts.appendleft(final_center)
        else:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                # 오랫동안 못 찾으면 상태 리셋
                self.last_center = None
                self.prev_centers.clear()
        
        return final_center, final_radius, used_method
    
    def _check_continuity(self, new_center):
        """
        연속성 검사: 새 위치가 이전 위치에서 합리적 거리 내인지 확인
        
        Args:
            new_center: (x, y) 새 감지 위치
            
        Returns:
            bool: 연속성 통과 여부
        """
        if self.last_center is None:
            return True  # 첫 감지는 항상 통과
        
        dist = np.sqrt((new_center[0] - self.last_center[0])**2 + 
                      (new_center[1] - self.last_center[1])**2)
        
        # lost_frames에 따라 허용 거리 증가 (오래 못 찾았으면 더 멀리 허용)
        allowed_dist = self.max_jump_dist * (1 + self.lost_frames * 0.3)
        
        return dist < allowed_dist

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
    
    def toggle_fmo(self):
        """
        FMO 탐지 활성화/비활성화 토글
        
        Returns:
            bool: 현재 FMO 활성화 상태
        """
        self.fmo_enabled = not self.fmo_enabled
        return self.fmo_enabled
    
    def set_fmo_enabled(self, enabled):
        """
        FMO 탐지 활성화 상태 설정
        
        Args:
            enabled: True/False
        """
        self.fmo_enabled = enabled

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
        상태 초기화 (새 투구 시작 시 호출)
        """
        self.pts.clear()
        self.pts_3d.clear()
        self.last_valid_color_radius = 0
        self.last_center = None
        self.prev_centers.clear()
        self.lost_frames = 0
        # 프레임 버퍼는 유지 (배경 모델 보존)
        # 배경 모델은 유지 (재학습에 시간이 걸리므로)
