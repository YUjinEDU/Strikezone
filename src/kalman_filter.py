import numpy as np
import cv2


class KalmanFilter3D:
    """3D 공간에서의 물체 추적을 위한 칼만 필터 클래스"""
    
    def __init__(self):
        """상태: (x, y, z, vx, vy, vz) -> 6차원, 측정: (x, y, z) -> 3차원"""
        self.kf = cv2.KalmanFilter(6, 3)  # stateDim=6, measDim=3
        
        # 전이 행렬 (A)
        # 등속도 모델: x' = x + vx, y' = y + vy, z' = z + vz, vx' = vx, vy' = vy, vz' = vz
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        
        # 측정 행렬 (H) - 관측: x, y, z만 측정
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)
        
        # 공정 잡음, 측정 잡음
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.1
        
        # 초기 오차 공분산
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        
        # 초기 상태 추정치 (x,y,z,vx,vy,vz)
        self.kf.statePost = np.zeros((6, 1), dtype=np.float32)
    
    def update_with_gating(self, measurement, gating_threshold=7.81):
        """
        게이팅을 사용하여 칼만 필터 업데이트 수행
        
        Args:
            measurement: 측정값 (x, y, z)
            gating_threshold: 허용 가능한 마할라노비스 거리 임계값
            
        Returns:
            필터링된 상태 (x, y, z)
        """
        # 예측 단계
        predicted_state = self.kf.transitionMatrix @ self.kf.statePost
        predicted_P = self.kf.transitionMatrix @ self.kf.errorCovPost @ self.kf.transitionMatrix.T + self.kf.processNoiseCov
        
        # 측정 예측: H * predicted_state
        measurement_prediction = self.kf.measurementMatrix @ predicted_state
        
        # 혁신 (Innovation)
        innovation = measurement.reshape(-1, 1) - measurement_prediction
        
        # 혁신 공분산: S = H*P*H^T + R
        S = self.kf.measurementMatrix @ predicted_P @ self.kf.measurementMatrix.T + self.kf.measurementNoiseCov
        
        # 마할라노비스 거리 계산
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # S가 특이행렬이면 업데이트하지 않음
            self.kf.statePost = predicted_state
            self.kf.errorCovPost = predicted_P
            return self.get_position()
        
        mahalanobis_distance = (innovation.T @ S_inv @ innovation).item()
        
        # 게이팅: 임계값보다 작으면 업데이트 진행
        if mahalanobis_distance < gating_threshold:
            # 칼만 이득 계산
            K = predicted_P @ self.kf.measurementMatrix.T @ S_inv
            
            # 상태 갱신
            self.kf.statePost = predicted_state + K @ innovation
            
            # 오차 공분산 갱신
            self.kf.errorCovPost = (np.eye(self.kf.statePost.shape[0]) - K @ self.kf.measurementMatrix) @ predicted_P
        else:
            # 이상치로 판단하여 예측 결과만 사용
            self.kf.statePost = predicted_state
            self.kf.errorCovPost = predicted_P
            print(f"측정값 이상치 감지: 갱신 단계 생략 (마할라노비스 거리: {mahalanobis_distance:.2f})")
        
        return self.get_position()
    
    def predict(self):
        """
        상태 예측 수행
        
        Returns:
            예측된 위치 (x, y, z)
        """
        predicted = self.kf.predict()
        return predicted[:3].flatten()
    
    def get_position(self):
        """
        현재 추정된 위치 반환
        
        Returns:
            현재 추정된 위치 (x, y, z)
        """
        return self.kf.statePost[:3].flatten()
    
    def get_velocity(self):
        """
        현재 추정된 속도 반환
        
        Returns:
            현재 추정된 속도 (vx, vy, vz)
        """
        return self.kf.statePost[3:].flatten()
    
    def get_state(self):
        """
        전체 상태 벡터 반환 (위치 + 속도)
        
        Returns:
            전체 상태 벡터 [x, y, z, vx, vy, vz]
        """
        return self.kf.statePost.flatten()
    
    def reset(self):
        """필터 상태 초기화"""
        self.kf.statePost = np.zeros((6, 1), dtype=np.float32)
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) 