import cv2
import numpy as np
import imutils
import mediapipe as mp
from collections import deque

class KalmanTracker:
    def __init__(self):
        self.kalman_filter = self.init_kalman_filter()
    
    def init_kalman_filter(self):
        """칼만 필터 초기화"""
        kf = cv2.KalmanFilter(6, 3)  # stateDim=6, measDim=3

        # Transition matrix (A)
        kf.transitionMatrix = np.array([
            [1,0,0,1,0,0],
            [0,1,0,0,1,0],
            [0,0,1,0,0,1],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1],
        ], dtype=np.float32)

        # 측정행렬 (H)
        kf.measurementMatrix = np.array([
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,1,0,0,0]
        ], dtype=np.float32)

        # 공정 잡음, 측정 잡음
        kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.005
        kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.5
        
        # 초기 오차 공분산
        kf.errorCovPost = np.eye(6, dtype=np.float32)

        # 초기 상태 추정치
        kf.statePost = np.zeros((6,1), dtype=np.float32)

        return kf
    
    def update_with_gating(self, measurement, gating_threshold=7.81):
        """게이팅 로직을 적용한 칼만 필터 업데이트"""
        kf = self.kalman_filter
        
        # 예측 단계
        predicted_state = kf.transitionMatrix @ kf.statePost
        predicted_P = kf.transitionMatrix @ kf.errorCovPost @ kf.transitionMatrix.T + kf.processNoiseCov

        # 측정 예측: H * predicted_state
        measurement_prediction = kf.measurementMatrix @ predicted_state

        # 혁신
        innovation = measurement.reshape(-1, 1) - measurement_prediction

        # 혁신 공분산: S = H*P*H^T + R
        S = kf.measurementMatrix @ predicted_P @ kf.measurementMatrix.T + kf.measurementNoiseCov

        # 마할라노비스 거리 계산
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # S가 singular하면 업데이트하지 않습니다.
            return

        mahalanobis_distance = (innovation.T @ S_inv @ innovation).item()

        # 게이팅: 임계값보다 작아야 업데이트 진행
        if (mahalanobis_distance < gating_threshold):
            # 칼만 이득 계산
            K = predicted_P @ kf.measurementMatrix.T @ S_inv

            # 상태 갱신
            kf.statePost = predicted_state + K @ innovation

            # 오차 공분산 갱신
            kf.errorCovPost = (np.eye(kf.statePost.shape[0]) - K @ kf.measurementMatrix) @ predicted_P
        else:
            # 이상치로 판단하여 측정 업데이트를 건너뛰고 예측 결과만 사용
            kf.statePost = predicted_state
            kf.errorCovPost = predicted_P
            print("측정값 이상치 감지: 갱신 단계 생략 (마할라노비스 거리: {:.2f})".format(mahalanobis_distance))
    
    def get_filtered_position(self):
        """필터링된 위치 반환"""
        return self.kalman_filter.statePost[:3].flatten()

class BallDetector:
    def __init__(self, lower_color, upper_color, use_motion=False, use_consistency=False, 
                motion_threshold=10, circularity_threshold=0.5, 
                min_area=30, max_area=8000):
        self.lower_color = lower_color
        self.upper_color = upper_color
        self.use_motion = use_motion
        self.use_consistency = use_consistency
        self.motion_threshold = motion_threshold
        self.circularity_threshold = circularity_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.prev_frame = None
        self.recent_centers = deque(maxlen=10)
        self.pts = deque(maxlen=64)

    

    def detect(self, frame, debug=False):
        """프레임에서 볼 감지 - 단계적 적용 가능"""
        # 1. 색상 기반 필터링 (기본 기능)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        
        # 2. 움직임 감지 적용 (선택적)
        if self.use_motion:
            mask = self._apply_motion_detection(frame, color_mask)
        else:
            mask = color_mask
        
        # 모폴로지 연산 (공통)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 침식+팽창으로 노이즈 제거
        mask = cv2.dilate(mask, kernel, iterations=2)  # 확실한 팽창
        
        # 3. 윤곽선 검출 및 원형도 필터링 (공통)
        center, radius = self._detect_contours(mask)
        
        # 4. 다중 프레임 일관성 검사 (선택적)
        if self.use_consistency:
            center, radius = self._check_trajectory_consistency(center, radius)
        
        # 디버그 정보 준비
        debug_info = None
        if debug:
            debug_info = self._prepare_debug_info(frame, color_mask, mask, center, radius)
        
        return center, radius, mask, debug_info if debug else None
    
    def _apply_motion_detection(self, frame, color_mask):
        """움직임 감지 적용"""
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = curr_gray
            return color_mask
        
        # 이전 프레임과 차이 계산
        diff = cv2.absdiff(self.prev_frame, curr_gray)
        _, motion_mask = cv2.threshold(diff, 7, 255, cv2.THRESH_BINARY)

         # 노이즈 제거 후 팽창 연산 늘리기
        motion_mask = cv2.medianBlur(motion_mask, 3)  # 노이즈 제거 추가
        motion_mask = cv2.dilate(motion_mask, None, iterations=3)
        
        # 현재 프레임 저장
        self.prev_frame = curr_gray

        
        # OR 연산으로 변경 - 조건을 더 관대하게
        combined_mask = cv2.bitwise_or(color_mask, motion_mask)
        # 모폴로지 연산으로 노이즈 제거
        kernel = np.ones((3,3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        return combined_mask

    def _detect_contours(self, mask):
        """윤곽선 검출 및 원형도 필터링"""
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
        radius = 0
        
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            if 3 < area < 10000 and circularity > 0.5:
                # 모멘트 계산:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                else:
                    center = (int(x), int(y))
        
        return center, radius

    def _check_trajectory_consistency(self, center, radius):
        """다중 프레임 일관성 검사"""
        if center is None:
            self.recent_centers.append(None)
            return None, 0
        
        # 이전 중심점과 거리 계산
        if len(self.recent_centers) > 0 and self.recent_centers[-1] is not None:
            last_center = self.recent_centers[-1]
            distance = np.linalg.norm(np.array(center) - np.array(last_center))
            
            # 거리가 임계값보다 크면 공이 아닌 것으로 판단
            max_distance = max(100, radius * 3)  # 공 크기에 따른 동적 임계값
            if distance > max_distance:
                self.recent_centers.append(None)
                return None, 0
        
        # 현재 중심점 기록
        self.recent_centers.append(center)
        
        # 충분한 기록이 있을 때 전체 궤적 일관성 검사
        if len(self.recent_centers) >= 5:
            valid_centers = [c for c in self.recent_centers if c is not None]
            if len(valid_centers) >= 3:
                distances = [np.linalg.norm(np.array(valid_centers[i]) - np.array(valid_centers[i+1])) 
                            for i in range(len(valid_centers)-1)]
                
                if max(distances) > 100 or np.std(distances) > 30:
                    return None, 0
        
        return center, radius

    def _prepare_debug_info(self, frame, color_mask, final_mask, center, radius):
        """디버그 정보 준비"""
        debug_info = {
            "원본 프레임": frame.copy(),
            "색상 마스크": cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR),
            "최종 마스크": cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR),
            "결과": frame.copy()
        }
        
        if center and radius > 0:
            cv2.circle(debug_info["결과"], center, int(radius), (0, 255, 255), 2)
            cv2.circle(debug_info["결과"], center, 5, (0, 0, 255), -1)
        
        return debug_info

    def draw_ball(self, frame, center, radius):
        """공 시각화"""
        if center and radius > 0.5:
            cv2.circle(frame, (int(center[0]), int(center[1])), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            return True
        return False
            
    def track_trajectory(self, center):
        """궤적 추적"""
        self.pts.appendleft(center)
            
    def draw_trajectory(self, frame, color=(0, 255, 0), fade=True):
        """
        궤적 그리기 (개선된 버전)
        
        Parameters:
        - frame: 궤적을 그릴 프레임
        - color: 궤적 색상 (B, G, R)
        - fade: 점점 흐려지는 효과 적용 여부
        """
        # 유효한 점이 없으면 바로 반환
        if len(self.pts) < 2:
            return frame
        
        # 유효한 점만 필터링 (None이 아닌 점)
        valid_pts = [pt for pt in self.pts if pt is not None]
        
        # 유효한 점이 2개 미만이면 그리기 불가
        if len(valid_pts) < 2:
            return frame
        
        # 궤적 그리기
        for i in range(1, len(valid_pts)):
            # 현재점과 이전점
            pt1 = tuple(map(int, valid_pts[i-1]))
            pt2 = tuple(map(int, valid_pts[i]))
            
            if fade:
                # 점점 옅어지는 효과 (최근 궤적이 더 두껍고 선명)
                alpha = 1.0 - float(i) / len(valid_pts)
                thickness = max(1, int(np.sqrt(64 / float(i + 1)) * 2.5))
                
                # 컬러 블렌딩 (점점 흐려지는 효과)
                curr_color = (
                    int(color[0] * alpha + 255 * (1-alpha)),
                    int(color[1] * alpha + 255 * (1-alpha)),
                    int(color[2] * alpha)
                )
            else:
                # 일정한 색상과 두께
                thickness = 2
                curr_color = color
            
            # 선 그리기
            cv2.line(frame, pt1, pt2, curr_color, thickness)
        
        return frame

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
    
    def estimate(self, frame):
        """프레임에서 포즈 추정"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results.pose_landmarks

class HandDetector:
    """손 감지 및 제스처 인식 클래스"""
    
    def __init__(self, max_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands, 
            min_detection_confidence=min_detection_confidence, 
            min_tracking_confidence=min_tracking_confidence
        )
        
    def find_hands(self, frame):
        """프레임에서 손 감지"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_frame)
        return self.results
    

    def is_hand_open(self):
        """손바닥이 펴진 상태인지 확인"""
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                # 손가락 끝 위치
                tips = [hand_landmarks.landmark[i] for i in [8, 12, 16, 20]]
                # 손가락이 펴졌는지 간단 체크
                if all(tip.y < hand_landmarks.landmark[i - 2].y for tip, i in zip(tips, [8, 12, 16, 20])):
                    return True
        return False
    
    def is_index_finger_only(self):
        """검지 손가락만 펴져 있는지 확인"""
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                index_tip = hand_landmarks.landmark[8]   # 검지 손가락 끝
                index_mcp = hand_landmarks.landmark[5]   # 검지 손가락 관절

                # 검지만 올라갔는지 확인
                other_fingers_folded = all(
                    hand_landmarks.landmark[i].y > hand_landmarks.landmark[i - 2].y
                    for i in [12, 16, 20]  # 중지, 약지, 새끼 손가락
                )

                if index_tip.y < index_mcp.y and other_fingers_folded:
                    return True
        return False

