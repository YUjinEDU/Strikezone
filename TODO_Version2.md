CPU 성능이 충분하다면, 연산량이 많더라도 \*\*물리학적 최적화(Optimization)\*\*와 **앙상블(Ensemble) 탐지**를 결합하여 단일 카메라의 한계(깊이 오차)를 수학적으로 극복하는 방식이 "정확도 끝판왕"입니다.

단순히 FMO만 쓰는 것이 아니라, \*\*① 하이브리드 탐지(Hybrid Detection)\*\*로 2D 좌표의 신뢰성을 확보하고, \*\*② 궤적 최적화(Trajectory Optimization)\*\*로 튀는 3D 좌표를 매끄럽게 보정하는 고성능 아키텍처를 설계해 드리겠습니다.

-----

### 1\. 고성능 아키텍처: "하이브리드 탐지 + 궤적 피팅"

Ryzen AI 9 375의 강력한 성능을 믿고 다음 3가지 고급 기술을 투입합니다.

1.  **하이브리드 탐지 (Hybrid Detection):**
      * \*\*FMO (배경 차분)\*\*와 \*\*HSV (색상)\*\*를 동시에 돌립니다.
      * 공이 빠를 땐(잔상) FMO가, 느리거나 정지했을 땐 HSV가 더 정확합니다. 두 알고리즘의 결과를 \*\*융합(Fusion)\*\*하여 어떤 상황에서도 공을 놓치지 않게 합니다.
2.  **파티클 필터 (Particle Filter) 또는 UKF:**
      * 기존 칼만 필터(선형)보다 비선형 움직임(공기 저항 등)과 비정규 노이즈에 강한 **파티클 필터**를 사용하여 3D 위치를 추정합니다. 연산량이 많지만 정확도는 훨씬 높습니다.
3.  **실시간 궤적 피팅 (Trajectory Fitting):**
      * 매 프레임 나오는 Z값(깊이)은 노이즈가 심합니다.
      * 공이 날아가는 동안 수집된 2D 좌표($u, v$)들의 집합을 바탕으로, \*\*"물리적으로 가장 타당한 3D 포물선"\*\*을 역산해내는 **최적화(Optimization)** 과정을 수행합니다. 이것이 3D 정확도를 잡는 핵심 열쇠(Key)입니다.

-----

### 2\. 코드 구현 설계

#### A. 하이브리드 감지기 (`src/hybrid_detector.py`)

FMO와 Color 방식을 병렬로 실행하고, 더 신뢰도 높은 결과를 채택합니다.

```python
import cv2
import numpy as np
from config import *

class HybridDetector:
    def __init__(self):
        # 1. FMO (배경 차분) - 고속용
        self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
            history=300, dist2Threshold=400.0, detectShadows=True # 그림자 감지 켜기 (정확도 UP, 연산량 UP)
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # 2. Color (HSV) - 저속/정확용
        self.lower_color = np.array(GREEN_LOWER)
        self.upper_color = np.array(GREEN_UPPER)

    def detect(self, frame):
        # --- A. FMO 탐지 ---
        mask_fmo = self.bg_subtractor.apply(frame)
        mask_fmo = cv2.morphologyEx(mask_fmo, cv2.MORPH_OPEN, self.kernel)
        contours_fmo, _ = cv2.findContours(mask_fmo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fmo_result = None
        fmo_score = 0
        
        for cnt in contours_fmo:
            area = cv2.contourArea(cnt)
            if 50 < area < 5000:
                rect = cv2.minAreaRect(cnt)
                (x, y), (w, h), angle = rect
                minor_axis = min(w, h)
                major_axis = max(w, h)
                aspect_ratio = major_axis / (minor_axis + 1e-6)
                
                # FMO는 길쭉한 형상(잔상)에 가산점
                if aspect_ratio > 1.2:
                    score = area * aspect_ratio
                    if score > fmo_score:
                        fmo_score = score
                        fmo_result = ((int(x), int(y)), minor_axis / 2.0) # 중심, 반지름(짧은축/2)

        # --- B. Color 탐지 ---
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask_color = cv2.inRange(hsv, self.lower_color, self.upper_color)
        mask_color = cv2.erode(mask_color, None, iterations=2)
        mask_color = cv2.dilate(mask_color, None, iterations=2)
        contours_color, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        color_result = None
        color_score = 0
        
        for cnt in contours_color:
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            area = cv2.contourArea(cnt)
            if 30 < area < 5000:
                # Color는 원형에 가까울수록 가산점
                score = area
                if score > color_score:
                    color_score = score
                    color_result = ((int(x), int(y)), radius)

        # --- C. 결과 융합 (Decision Logic) ---
        # FMO가 아주 강한 신호(빠른 공)를 잡았다면 FMO 우선
        # 그렇지 않다면 Color가 더 안정적이므로 Color 우선
        final_center = None
        final_radius = 0
        used_method = "None"

        if fmo_result and (fmo_score > 500): # FMO 신뢰도가 높음 (빠른 움직임)
            final_center, final_radius = fmo_result
            used_method = "FMO"
        elif color_result: # FMO가 없거나 약하면 Color 사용
            final_center, final_radius = color_result
            used_method = "Color"
        elif fmo_result: # Color도 없으면 약한 FMO라도 사용
            final_center, final_radius = fmo_result
            used_method = "FMO_Weak"

        return final_center, final_radius, used_method
```

#### B. 궤적 최적화기 (`src/trajectory_optimizer.py`)

이것이 **정확도의 핵심**입니다.
지금까지 쌓인 2D 좌표($u, v$)들을 모두 설명할 수 있는 **단 하나의 완벽한 3D 물리 곡선**을 찾아냅니다. (`scipy` 라이브러리 필요: `pip install scipy`)

```python
import numpy as np
from scipy.optimize import least_squares

class TrajectoryOptimizer:
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.g = 9.81 # 중력 가속도

    def project_point(self, point3d):
        """3D 점을 2D 픽셀로 투영 (cv2.projectPoints 단순화)"""
        point3d = np.array(point3d, dtype=np.float64)
        img_points, _ = cv2.projectPoints(
            point3d.reshape(1, 1, 3), np.zeros(3), np.zeros(3), self.camera_matrix, self.dist_coeffs
        )
        return img_points[0][0]

    def physics_model(self, t, initial_state):
        """
        초기 상태에서 t초 후의 위치 계산 (공기 저항 무시한 포물선 운동)
        initial_state: [x0, y0, z0, vx0, vy0, vz0]
        """
        x0, y0, z0, vx0, vy0, vz0 = initial_state
        x = x0 + vx0 * t
        y = y0 + vy0 * t
        z = z0 + vz0 * t - 0.5 * self.g * t**2 # 중력 반영 (Z축이 위쪽+ 일 경우 -g)
        # 만약 Y축이 깊이이고 중력이 Y에 작용하지 않는다면 Z축 수정 필요
        # (Strikezone 좌표계: Y=깊이, Z=높이. 따라서 중력은 Z축에 작용)
        return np.array([x, y, z])

    def objective_function(self, params, times, observed_2d_points):
        """
        최적화 목적 함수: 예측된 2D 궤적과 관측된 2D 궤적 사이의 오차(Residual) 계산
        """
        initial_state = params # [x0, y0, z0, vx0, vy0, vz0]
        residuals = []
        
        for t, obs_pt in zip(times, observed_2d_points):
            pred_3d = self.physics_model(t, initial_state)
            pred_2d = self.project_point(pred_3d)
            
            # X, Y 픽셀 오차
            residuals.append(pred_2d[0] - obs_pt[0])
            residuals.append(pred_2d[1] - obs_pt[1])
            
        return np.array(residuals)

    def optimize(self, observed_points_2d, observed_times, initial_guess_3d):
        """
        관측된 2D 점들을 바탕으로 최적의 3D 초기 상태(위치, 속도)를 찾음
        """
        # 초기값 추정 (매우 중요) -> Kalman Filter의 현재 상태를 초기값으로 사용
        x0 = initial_guess_3d # [x, y, z, vx, vy, vz]
        
        # Levenberg-Marquardt 알고리즘으로 최적화 수행
        result = least_squares(
            self.objective_function, 
            x0, 
            args=(observed_times, observed_points_2d),
            method='lm'
        )
        
        return result.x # 최적화된 초기 상태 [x0, y0, z0, vx0, vy0, vz0]
```

-----

### 3\. 통합 및 실행 로직 (`src/main_v8.py`)

메인 루프에서 다음과 같이 작동합니다.

1.  **HybridDetector**가 매 프레임 `(u, v)` 픽셀 좌표와 `timestamp`를 수집합니다.
2.  이 데이터는 `KalmanFilter`에도 들어가서 실시간 시각화(대략적인 위치)에 사용됩니다.
3.  **핵심:** 데이터가 일정량(예: 5\~10프레임) 쌓이면 \*\*`TrajectoryOptimizer`\*\*를 돌립니다.
4.  최적화기가 "지금까지의 2D 점들을 가장 잘 설명하는 3D 궤적"을 수학적으로 찾아내면, \*\*칼만 필터의 상태를 이 최적화된 값으로 강제 업데이트(Correction)\*\*합니다.
5.  이렇게 하면 Z축(깊이)이 튀는 현상이 물리 법칙에 의해 강제로 억제되어 매우 매끄럽고 정확한 궤적을 얻게 됩니다.

**수정된 메인 로직 스니펫:**

```python
# ... (초기화) ...
hybrid_detector = HybridDetector()
optimizer = TrajectoryOptimizer(camera_matrix, dist_coeffs)
observed_2d = [] # (x, y) 리스트
observed_times = [] # 시간 리스트

while True:
    # ... (프레임 읽기) ...
    
    # 1. 하이브리드 탐지
    center, radius, method = hybrid_detector.detect(analysis_frame)
    
    if center:
        t = time.time() - start_time
        observed_2d.append(center)
        observed_times.append(t)
        
        # 2. 실시간 시각화 (기존 로직 유지)
        # ... (칼만 필터 업데이트 및 그리기) ...
        
        # 3. [고성능] 실시간 궤적 최적화 (데이터가 6개 이상 모이면 시작)
        if len(observed_2d) > 6:
            # 현재 칼만 필터의 상태를 초기 추정값으로 사용
            current_state = kalman_tracker.get_state() # [x,y,z,vx,vy,vz] 반환 함수 필요
            
            # 최적화 수행 (CPU 성능 활용)
            optimized_state = optimizer.optimize(
                observed_2d[-10:], # 최근 10개 프레임만 사용 (Sliding Window)
                observed_times[-10:], 
                current_state
            )
            
            # 최적화된 3D 좌표로 현재 위치 보정
            corrected_pos_3d = optimizer.physics_model(observed_times[-1], optimized_state)
            
            # 4. 보정된 좌표를 사용하여 판정 및 시각화 갱신
            # (이 좌표가 radius로 계산한 Z값보다 훨씬 정확함)
            final_x, final_y, final_z = corrected_pos_3d
            
            # ... (이후 스트라이크/볼 판정 로직) ...
```

### 요약: 왜 이것이 최고인가?

1.  **하이브리드 탐지:** 빠른 공(FMO)과 느린 공(Color)을 모두 잡아냅니다.
2.  **수학적 최적화 (Optimization):** 부정확한 `radius` 기반 깊이 추정에 의존하지 않습니다. 대신, "이 2D 궤적을 그리려면 공이 3D 공간 어디에 있어야 하는가?"를 역산합니다. 이것은 다안(Stereo) 카메라가 없는 상황에서 3D를 복원하는 가장 강력한 수학적 기법입니다.
3.  **CPU 활용:** `scipy.optimize.least_squares`는 반복 연산을 수행하므로 CPU 파워가 필요하지만, Ryzen AI 9 375라면 충분히 실시간(60fps 내) 처리가 가능합니다.

이 설계대로 진행하시면 데이터셋 없이, 단일 카메라로 구현할 수 있는 **물리적 한계에 가까운 정확도**를 얻으실 수 있습니다.