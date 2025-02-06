#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import plotly.graph_objs as go
import cv2

def init_kalman_3d():
    """
    6차원 상태: (x, y, z, vx, vy, vz)
    3차원 측정: (x, y, z)
    등속도 모델
    """
    kf = cv2.KalmanFilter(6, 3)  # stateDim=6, measureDim=3

    # Transition Matrix A
    # x' = x + vx
    # y' = y + vy
    # z' = z + vz
    # vx' = vx
    # vy' = vy
    # vz' = vz
    kf.transitionMatrix = np.array([
        [1,0,0,1,0,0],
        [0,1,0,0,1,0],
        [0,0,1,0,0,1],
        [0,0,0,1,0,0],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1],
    ], dtype=np.float32)

    # 측정행렬 H: (3x6)
    # 관측: x,y,z
    kf.measurementMatrix = np.array([
        [1,0,0,0,0,0],
        [0,1,0,0,0,0],
        [0,0,1,0,0,0]
    ], dtype=np.float32)

    # 공정 잡음, 측정 잡음 대략 설정 (상황에 맞게 조정)
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.01
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.1
    kf.errorCovPost = np.eye(6, dtype=np.float32) * 1

    # 초기 상태
    kf.statePost = np.zeros((6,1), dtype=np.float32)

    return kf

def main():
    # ---------------------
    # 1) 가상의 3D 곡선 (참값) 생성
    # ---------------------
    n_points = 50
    t = np.linspace(0, 2*np.pi, n_points)
    # 예: 나선형(helix)  x=cos(t), y=sin(t), z= t
    true_x = np.cos(t)
    true_y = np.sin(t)
    true_z = t / (2*np.pi) * 3  # z 범위 대략 0~3

    # 3D 좌표 (n_points, 3)
    true_3d = np.stack([true_x, true_y, true_z], axis=1)

    # ---------------------
    # 2) 측정값 = 참값 + 노이즈
    # ---------------------
    rng = np.random.default_rng(seed=42)
    noise_std = 0.2
    measured_3d = true_3d + rng.normal(0, noise_std, size=true_3d.shape)

    # ---------------------
    # 3) 칼만 필터 초기화
    # ---------------------
    kf = init_kalman_3d()

    # 첫 관측값으로 초기 설정 (statePost[:3] = x0,y0,z0)
    kf.statePost[0,0] = measured_3d[0,0]
    kf.statePost[1,0] = measured_3d[0,1]
    kf.statePost[2,0] = measured_3d[0,2]

    # ---------------------
    # 4) main loop - predict/correct
    # ---------------------
    filtered_3d = []  # (n_points, 3) 보정 결과

    for i in range(n_points):
        # predict
        pred = kf.predict()  # shape=(6,1)
        # px_pred, py_pred, pz_pred = pred[0], pred[1], pred[2]

        # 측정값
        mx, my, mz = measured_3d[i]

        # correct
        meas = np.array([[mx],[my],[mz]], dtype=np.float32)
        estimated = kf.correct(meas)
        ex, ey, ez, evx, evy, evz = estimated.ravel()

        # 저장
        filtered_3d.append((ex, ey, ez))

    filtered_3d = np.array(filtered_3d)  # (n_points,3)

    # ---------------------
    # 5) Plotly로 시각화
    # ---------------------
    # - true (실제 곡선) : 파란색 라인
    # - measured (노이즈) : 빨간색 점
    # - filtered (칼만) : 두꺼운 반투명 흰색 라인
    trace_true = go.Scatter3d(
        x=true_3d[:,0], y=true_3d[:,1], z=true_3d[:,2],
        mode='lines',
        line=dict(color='blue', width=2),
        name='True Path'
    )
    trace_meas = go.Scatter3d(
        x=measured_3d[:,0], y=measured_3d[:,1], z=measured_3d[:,2],
        mode='markers',
        marker=dict(color='red', size=4),
        name='Measured'
    )
    trace_filt = go.Scatter3d(
        x=filtered_3d[:,0], y=filtered_3d[:,1], z=filtered_3d[:,2],
        mode='lines',
        line=dict(
            color='rgba(255,255,255,0.6)',  # 흰색 + 60% 불투명
            width=6
        ),
        name='Filtered (Kalman)'
    )

    fig = go.Figure(data=[trace_true, trace_meas, trace_filt])
    fig.update_layout(
        scene=dict(aspectmode='cube'),
        title='3D Kalman Filter Demo'
    )

    # HTML로 저장
    fig.write_html("kalman_3d_demo.html")
    print("kalman_3d_demo.html 생성. 브라우저로 열어보세요.")

if __name__ == "__main__":
    main()



