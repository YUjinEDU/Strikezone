# -*- coding: utf-8 -*-
"""
GUI 통합 메인 실행 파일
기존 main_v7.py의 검출 로직 + PyQt5 GUI
"""

import os
import sys
import cv2
import numpy as np
import threading
import time
import logging
import json
from collections import deque

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject

# 내부 모듈
from config import *
from camera import CameraManager
from aruco_detector import ArucoDetector
from tracker_v1 import KalmanTracker
from hybrid_detector import HybridDetector
from kalman_filter import KalmanFilter3D
from baseball_scoreboard import BaseballScoreboard

# GUI 모듈
from gui_config import vis_config, game_config, window_config
from gui_app import MainWindow, VideoThread

# 설정 파일 경로
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), 'user_settings.json')


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_user_settings():
    """저장된 사용자 설정 불러오기"""
    default_settings = {
        'vis_settings': {
            'zone': True, 'plane1': True, 'plane2': True,
            'grid': True, 'trajectory': True, 'speed': True,
            'scoreboard': True, 'aruco': True, 'axes': False,
            'fmo': False
        },
        'game_mode_enabled': False,
        'target_zone': None
    }
    
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                saved = json.load(f)
                # 기존 설정과 병합 (새로운 키가 추가될 경우 대비)
                for key in default_settings:
                    if key not in saved:
                        saved[key] = default_settings[key]
                    elif isinstance(default_settings[key], dict):
                        for sub_key in default_settings[key]:
                            if sub_key not in saved[key]:
                                saved[key][sub_key] = default_settings[key][sub_key]
                return saved
        except Exception as e:
            print(f"[설정] 불러오기 실패: {e}, 기본값 사용")
    
    return default_settings


def save_user_settings(vis_settings, game_mode_enabled=False, target_zone=None):
    """사용자 설정 저장"""
    settings = {
        'vis_settings': vis_settings,
        'game_mode_enabled': game_mode_enabled,
        'target_zone': target_zone
    }
    
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        print(f"[설정] 저장 완료: {SETTINGS_FILE}")
    except Exception as e:
        print(f"[설정] 저장 실패: {e}")


class PitchAnalyzer(QObject):
    """
    투구 분석기 클래스
    ArUco 감지, 공 추적, 스트라이크/볼 판정을 담당
    """
    
    # 시그널
    pitch_detected = pyqtSignal(dict)  # 투구 결과
    frame_processed = pyqtSignal(np.ndarray)  # 처리된 프레임
    speed_updated = pyqtSignal(float)  # 구속 업데이트
    
    def __init__(self, camera_matrix, dist_coeffs):
        super().__init__()
        
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # ArUco 검출기
        self.aruco_detector = ArucoDetector(ARUCO_MARKER_SIZE, camera_matrix, dist_coeffs)
        
        # 공 검출기 (하이브리드)
        self.ball_detector = HybridDetector(GREEN_LOWER, GREEN_UPPER)
        self.ball_detector.fmo_enabled = False  # 기본값: FMO 비활성화 (GUI에서 토글)
        
        # 칼만 필터
        self.kalman_tracker = KalmanFilter3D()
        
        # 스코어보드
        self.scoreboard = BaseballScoreboard(width=0.3, height=0.24, offset_x=-0.4, offset_y=0.0, offset_z=0.1)
        
        # 존 설정
        self.ball_zone_corners = BALL_ZONE_CORNERS.copy()
        self.ball_zone_corners2 = BALL_ZONE_CORNERS.copy()
        self.ball_zone_corners2[:, 1] += ZONE_DEPTH
        self.box_corners_3d = self.aruco_detector.get_box_corners_3d(BOX_MIN, BOX_MAX)
        
        # 평면 법선 방향: 투수가 -Y에 있으므로 공이 -Y→+Y로 날아옴
        # 따라서 깊이 축을 -Y로 설정해야 d가 양수→음수로 변화함
        self.desired_depth_axis = np.array([0, -1, 0], dtype=np.float32)
        
        # 상태 변수 초기화
        self._reset_state()
        
        # 궤적 버퍼
        self.traj_x, self.traj_y, self.traj_z = [], [], []
        
        # 기록된 투구들
        self.impact_points = []
        
        # 사용자 설정 불러오기
        user_settings = load_user_settings()
        
        # 시각화 설정
        self.vis_settings = user_settings.get('vis_settings', {
            'zone': True, 'plane1': True, 'plane2': True,
            'grid': True, 'trajectory': True, 'speed': True,
            'scoreboard': True, 'aruco': True, 'axes': False,
            'fmo': False
        })
        
        # 게임 모드 설정
        self.game_mode_enabled = user_settings.get('game_mode_enabled', False)
        self.target_zone = user_settings.get('target_zone', None)
        
        # 왜곡 보정
        self.undistort_map1 = None
        self.undistort_map2 = None
    
    def save_settings(self):
        """현재 설정 저장"""
        save_user_settings(self.vis_settings, self.game_mode_enabled, self.target_zone)
        
    def _reset_state(self):
        """상태 초기화"""
        self.prev_distance_to_plane1 = None
        self.prev_distance_to_plane2 = None
        self.prev_time_perf = None
        self.t_cross_plane1 = None
        self.t_cross_plane2 = None
        self.plane1_crossed = False
        self.plane2_crossed = False  # plane2 통과 상태 추가
        self.plane1_in_zone = False
        self.plane2_in_zone = False  # plane2 스트라이크존 내부 여부
        self.cross_point_p1_saved = None
        self.cross_point_p2_saved = None  # plane2 통과 지점
        self.first_plane_time = None  # 첫 번째 통과 시간
        self.display_velocity = 0.0
        self.realtime_speed_kmh = 0.0  # 실시간 속도
        
        # 판정 쿨다운 (연속 판정 방지)
        self.last_judgment_time = 0.0
        self.judgment_cooldown = 3.0  # 3초 쿨다운
        
        # 판정 완료 후 궤적 기록 중단 플래그
        self.pitch_completed = False
        self.pitch_completed_time = 0.0
        self.wait_for_reset = 1.5  # 판정 후 대기 시간 (초)
        
        # 프레임 간 속도 측정용
        self.prev_ball_pos_marker = None
        self.prev_ball_time = None
        self.frame_speed_buffer = []
        self.MAX_SPEED_BUFFER = 5
        
    def setup_undistort(self, frame_w, frame_h):
        """왜곡 보정 맵 설정"""
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, 
            (frame_w, frame_h), alpha=0
        )
        self.undistort_map1, self.undistort_map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, new_K, 
            (frame_w, frame_h), cv2.CV_16SC2
        )
        
    def update_vis_settings(self, settings):
        """시각화 설정 업데이트"""
        self.vis_settings = settings
        
        # FMO 모드 설정을 ball_detector에 전달
        fmo_enabled = settings.get('fmo', False)
        if hasattr(self, 'ball_detector') and self.ball_detector is not None:
            self.ball_detector.fmo_enabled = fmo_enabled
    
    def update_ball_color(self, color_name):
        """공 색상 변경"""
        from config import BALL_COLOR_PRESETS
        
        if color_name in BALL_COLOR_PRESETS:
            preset = BALL_COLOR_PRESETS[color_name]
            lower = preset["lower"]
            upper = preset["upper"]
            
            if hasattr(self, 'ball_detector') and self.ball_detector is not None:
                self.ball_detector.set_color_range(lower, upper)
                print(f"[PitchAnalyzer] 공 색상 변경: {color_name} ({lower} ~ {upper})")
    
    def update_game_mode(self, enabled, target_zone=None):
        """게임 모드 설정 업데이트"""
        self.game_mode_enabled = enabled
        self.target_zone = target_zone
        
    def process_frame(self, frame):
        """
        프레임 처리 및 분석
        
        Returns:
            overlay_frame: 시각화가 적용된 프레임
        """
        # 왜곡 보정
        if self.undistort_map1 is not None:
            frame = cv2.remap(frame, self.undistort_map1, self.undistort_map2, 
                            interpolation=cv2.INTER_LINEAR)
        
        overlay_frame = frame.copy()
        
        # ArUco 탐지
        corners, ids, rejected = self.aruco_detector.detect_markers(frame)
        
        if ids is None:
            return overlay_frame
            
        rvecs, tvecs = self.aruco_detector.estimate_pose(corners)
        
        for rvec, tvec in zip(rvecs, tvecs):
            # 시각화: 좌표축
            if self.vis_settings.get('axes', False):
                self.aruco_detector.draw_axes(overlay_frame, rvec, tvec, size=0.1)
            
            # 시각화: 스트라이크 존 3D 박스 (앞뒤 판정면 + 연결선)
            if self.vis_settings.get('zone', True):
                # 앞면 (plane1)
                projected_front = self.aruco_detector.project_points(self.ball_zone_corners, rvec, tvec)
                cv2.polylines(overlay_frame, [projected_front], True, (0, 255, 0), 2)
                # 뒷면 (plane2)
                projected_back = self.aruco_detector.project_points(self.ball_zone_corners2, rvec, tvec)
                cv2.polylines(overlay_frame, [projected_back], True, (0, 255, 0), 2)
                # 앞뒤 연결선 (4개 모서리)
                for i in range(4):
                    pt1 = tuple(projected_front[i])
                    pt2 = tuple(projected_back[i])
                    cv2.line(overlay_frame, pt1, pt2, (0, 255, 0), 1)
            
            # 시각화: plane1 (앞 판정면 - 시안색, zone과 독립)
            if self.vis_settings.get('plane1', True):
                projected_points = self.aruco_detector.project_points(self.ball_zone_corners, rvec, tvec)
                cv2.polylines(overlay_frame, [projected_points], True, (0, 255, 255), 3)
            
            # 시각화: plane2 (뒤 판정면 - 주황색, zone과 독립)
            if self.vis_settings.get('plane2', True):
                projected_points2 = self.aruco_detector.project_points(self.ball_zone_corners2, rvec, tvec)
                cv2.polylines(overlay_frame, [projected_points2], True, (255, 100, 0), 3)
                
                # 9분할 그리드는 plane2에만 적용
                if self.vis_settings.get('grid', True):
                    self._draw_grid_on_plane2(overlay_frame, rvec, tvec)
                
                # 게임모드 타겟 구역 하이라이트 (plane2에만)
                if self.game_mode_enabled and self.target_zone is not None:
                    self._draw_target_zone_on_plane2(overlay_frame, rvec, tvec, self.target_zone)
            
            # 3D 박스 그리기 (zone 옵션 - 스트라이크 존)
            if self.vis_settings.get('zone', True):
                pts2d_box = self.aruco_detector.project_points(self.box_corners_3d, rvec, tvec)
                self.aruco_detector.draw_3d_box(overlay_frame, pts2d_box, BOX_EDGES, color=(0,0,0), thickness=2)
            
            # 스코어보드 (scoreboard 옵션)
            if self.vis_settings.get('scoreboard', True):
                self.scoreboard.draw(overlay_frame, self.aruco_detector, rvec, tvec)
            
            # 기록된 투구 지점 표시 (ball_markers 옵션)
            # plane2에서의 실제 통과 위치를 표시 (카메라 위치와 무관하게 정확한 위치)
            if self.vis_settings.get('ball_markers', True):
                for impact in self.impact_points:
                    # plane2 좌표를 직접 투영 (plane2_point 사용)
                    point_3d = impact.get('plane2_point', impact['point_3d'])
                    self.aruco_detector.draw_impact_point_on_plane2(
                        overlay_frame,
                        point_3d,
                        self.ball_zone_corners2,
                        rvec, tvec,
                        circle_radius=12,
                        circle_color=(255, 255, 0),
                        circle_thickness=2,
                        number_text=impact['number'],
                        text_color=(255, 0, 0)
                    )
            
            # 궤적 표시
            if self.vis_settings.get('trajectory', True) and len(self.traj_x) >= 2:
                trajectory_3d = [[self.traj_x[i], self.traj_y[i], self.traj_z[i]] 
                               for i in range(len(self.traj_x))]
                self.aruco_detector.draw_trajectory_3d(
                    overlay_frame, trajectory_3d, rvec, tvec,
                    color=(0, 255, 0), thickness=2
                )
            
            # === 판정 완료 후 대기 시간 체크 ===
            current_time = time.time()
            if self.pitch_completed:
                # 판정 후 대기 시간이 지나면 새 투구 추적 시작
                if (current_time - self.pitch_completed_time) >= self.wait_for_reset:
                    self.pitch_completed = False
                    print("[추적 재개] 새 투구 대기 중...")
                else:
                    # 대기 중에는 공 추적/궤적 기록 건너뜀
                    self.prev_distance_to_plane1 = None
                    self.prev_distance_to_plane2 = None
                    self.prev_time_perf = None
                    continue  # 다음 마커로
            
            # 공 검출
            center, radius, detect_method = self.ball_detector.detect(frame)
            
            if center and radius > 0.4:
                # 공 표시
                self.ball_detector.draw_ball(overlay_frame, center, radius, detect_method)
                
                # === 깊이 추정 보정 ===
                # 기본 깊이 추정 (공 크기 기반)
                estimated_Z_ball = (self.camera_matrix[0, 0] * BALL_RADIUS_REAL) / max(1e-6, radius)
                
                # ArUco 마커 기준 깊이 (tvec[2])를 참조하여 보정
                marker_depth = float(tvec[2][0]) if tvec is not None else 1.0
                
                # 깊이 보정: 투수 마운드 거리까지 확장 (마커 앞 1m ~ 뒤 20m)
                # 투수 마운드는 약 18.44m 뒤에 있으므로 충분한 범위 확보
                min_depth = max(0.1, marker_depth - 1.0)
                max_depth = marker_depth + 20.0  # 1.0 → 20.0으로 확장
                estimated_Z = np.clip(estimated_Z_ball, min_depth, max_depth)
                
                # 카메라 좌표계 3D 복원
                ball_3d_cam = np.array([
                    (center[0] - self.camera_matrix[0,2]) * estimated_Z / self.camera_matrix[0,0],
                    (center[1] - self.camera_matrix[1,2]) * estimated_Z / self.camera_matrix[1,1],
                    estimated_Z
                ], dtype=np.float32)
                
                # 칼만 필터
                filtered_point_kalman = self.kalman_tracker.update_with_gating(ball_3d_cam)
                filtered_point = self.aruco_detector.point_to_marker_coord(
                    np.array(filtered_point_kalman, dtype=np.float32), rvec, tvec
                )
                
                # === 프레임 간 속도 계산 (마커 좌표계 Y축 = 깊이 방향) ===
                now_time = time.perf_counter()
                if self.prev_ball_pos_marker is not None and self.prev_ball_time is not None:
                    dt_frame = now_time - self.prev_ball_time
                    if dt_frame > 0.001:  # 1ms 이상일 때만
                        # Y축(깊이) 이동 거리로 속도 계산 (투구 방향)
                        dy = filtered_point[1] - self.prev_ball_pos_marker[1]
                        
                        # 주로 Y축 이동이면 유효한 속도
                        if abs(dy) > 0.01:  # 1cm 이상 깊이 이동
                            speed_mps = abs(dy) / dt_frame  # Y축 속도
                            frame_speed_kmh = speed_mps * 3.6
                            
                            # 합리적 범위 (1~200 km/h)만 버퍼에 추가
                            if 1.0 < frame_speed_kmh < 200.0:
                                self.frame_speed_buffer.append(frame_speed_kmh)
                                if len(self.frame_speed_buffer) > self.MAX_SPEED_BUFFER:
                                    self.frame_speed_buffer.pop(0)
                
                self.prev_ball_pos_marker = filtered_point.copy()
                self.prev_ball_time = now_time
                
                # 이동평균 속도
                if len(self.frame_speed_buffer) > 0:
                    self.realtime_speed_kmh = sum(self.frame_speed_buffer) / len(self.frame_speed_buffer)
                else:
                    self.realtime_speed_kmh = 0.0
                
                # === 궤적 기록 필터링 ===
                # 넓은 범위 내의 공만 기록 (X: ±0.8m, Z: -0.2~1.2m)
                x, y, z = filtered_point[0], filtered_point[1], filtered_point[2]
                in_tracking_zone = (-0.8 < x < 0.8) and (-0.2 < z < 1.2)
                
                # 공이 plane1 앞쪽에 있을 때 기록
                approaching_zone = (y > -0.5)  # plane1 앞 0.5m 이내
                
                if in_tracking_zone and approaching_zone:
                    self.traj_x.append(float(x))
                    self.traj_y.append(float(y))
                    self.traj_z.append(float(z))
                    # 디버깅: 첫 번째와 마지막 궤적 점 확인
                    if len(self.traj_x) == 1:
                        print(f"[DEBUG 궤적 시작] X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
                
                # 통과 감지 및 판정
                self._check_crossing(filtered_point, rvec, tvec)
        
        # 구속 표시
        if self.vis_settings.get('speed', True) and self.display_velocity > 0:
            self._draw_speed(overlay_frame)
        
        self.frame_processed.emit(overlay_frame)
        return overlay_frame
        
    def _draw_grid_on_plane2(self, frame, rvec, tvec):
        """9분할 그리드를 plane2에만 그리기"""
        zone = self.ball_zone_corners2  # plane2 사용
        
        # 수직선 (3등분)
        for i in range(1, 3):
            t = i / 3.0
            p1 = zone[0] + t * (zone[3] - zone[0])
            p2 = zone[1] + t * (zone[2] - zone[1])
            pts = self.aruco_detector.project_points(np.array([p1, p2]), rvec, tvec)
            cv2.line(frame, tuple(pts[0]), tuple(pts[1]), (128, 128, 128), 1)
            
        # 수평선 (3등분)
        for i in range(1, 3):
            t = i / 3.0
            p1 = zone[0] + t * (zone[1] - zone[0])
            p2 = zone[3] + t * (zone[2] - zone[3])
            pts = self.aruco_detector.project_points(np.array([p1, p2]), rvec, tvec)
            cv2.line(frame, tuple(pts[0]), tuple(pts[1]), (128, 128, 128), 1)
    
    def _draw_target_zone_on_plane2(self, frame, rvec, tvec, target_zone, color=(0, 165, 255), alpha=0.4):
        """
        게임모드에서 plane2의 타겟 구역을 반투명으로 하이라이트
        target_zone: 1~9 (좌상부터 우하까지)
        
        구역 배치:
        1 | 2 | 3
        ---------
        4 | 5 | 6
        ---------
        7 | 8 | 9
        """
        if target_zone is None or target_zone < 1 or target_zone > 9:
            return
        
        zone = self.ball_zone_corners2  # plane2 사용
        
        # 타겟 구역의 행/열 (0-indexed)
        # 야구 구역: 1,2,3(상단=높은Z), 4,5,6(중단), 7,8,9(하단=낮은Z)
        zone_idx = target_zone - 1
        row = zone_idx // 3  # 0, 1, 2 (0=상단, 2=하단)
        col = zone_idx % 3   # 0, 1, 2 (0=왼쪽, 2=오른쪽)
        
        # STRIKE_ZONE_CORNERS 배열 구조:
        # zone[0] = Bottom-left (Z=0.25, 낮음)
        # zone[1] = Bottom-right (Z=0.25, 낮음)
        # zone[2] = Top-right (Z=0.65, 높음)
        # zone[3] = Top-left (Z=0.65, 높음)
        
        # 구역의 4개 코너 계산 (3D)
        def interpolate_point_3d(t_col, t_row):
            """bilinear 보간으로 3D 점 계산
            t_row: 0=상단(높은Z), 1=하단(낮은Z)
            t_col: 0=왼쪽, 1=오른쪽
            """
            # 상단 변 (높은 Z): zone[3]=좌상, zone[2]=우상
            top = zone[3] * (1 - t_col) + zone[2] * t_col
            # 하단 변 (낮은 Z): zone[0]=좌하, zone[1]=우하
            bottom = zone[0] * (1 - t_col) + zone[1] * t_col
            # 수직 보간 (t_row=0이면 top, t_row=1이면 bottom)
            return top * (1 - t_row) + bottom * t_row
        
        # 구역 코너 (좌상, 우상, 우하, 좌하)
        c1 = col / 3.0
        c2 = (col + 1) / 3.0
        r1 = row / 3.0
        r2 = (row + 1) / 3.0
        
        zone_corners_3d = np.array([
            interpolate_point_3d(c1, r1),  # 좌상
            interpolate_point_3d(c2, r1),  # 우상
            interpolate_point_3d(c2, r2),  # 우하
            interpolate_point_3d(c1, r2),  # 좌하
        ], dtype=np.float32)
        
        # 3D 좌표를 2D로 투영
        projected = self.aruco_detector.project_points(zone_corners_3d, rvec, tvec)
        
        # 반투명 채우기
        overlay = frame.copy()
        cv2.fillPoly(overlay, [projected], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # 테두리
        cv2.polylines(frame, [projected], True, color, 2)
            
    def _draw_speed(self, frame):
        """구속 표시"""
        h, w = frame.shape[:2]
        speed_text = f"{self.display_velocity:.1f} km/h"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        
        (text_w, text_h), _ = cv2.getTextSize(speed_text, font, font_scale, thickness)
        x = w - text_w - 20
        y = 50
        
        # 배경
        cv2.rectangle(frame, (x - 10, y - text_h - 10), 
                     (x + text_w + 10, y + 10), (0, 0, 0), -1)
        # 텍스트
        cv2.putText(frame, speed_text, (x, y), font, font_scale, 
                   (0, 255, 255), thickness)
        
    def _check_crossing(self, filtered_point, rvec, tvec):
        """평면 통과 감지 및 판정 - main_v7.py와 동일한 로직"""
        p1_0, p1_1, p1_2 = self.ball_zone_corners[0], self.ball_zone_corners[1], self.ball_zone_corners[2]
        p2_0, p2_1, p2_2 = self.ball_zone_corners2[0], self.ball_zone_corners2[1], self.ball_zone_corners2[2]
        
        d1 = self.aruco_detector.signed_distance_to_plane_oriented(
            filtered_point, p1_0, p1_1, p1_2, desired_dir=self.desired_depth_axis
        )
        d2 = self.aruco_detector.signed_distance_to_plane_oriented(
            filtered_point, p2_0, p2_1, p2_2, desired_dir=self.desired_depth_axis
        )
        
        # [DEBUG] 좌표계 확인용 - 필요시 주석 해제
        # print(f"[좌표] Y={filtered_point[1]:.3f}m | d1(plane1)={d1:.3f} | d2(plane2)={d2:.3f} | plane1_Y=0 | plane2_Y={ZONE_DEPTH:.2f}")
        
        now_perf = time.perf_counter()
        current_time = time.time()
        
        # === 통과 감지 (정방향만) ===
        crossed_p1 = False
        crossed_p2 = False
        cross_point_p1 = None
        cross_point_p2 = None
        alpha1, alpha2 = 0.0, 0.0
        
        if self.prev_time_perf is not None and self.prev_distance_to_plane1 is not None:
            # plane1 통과 감지: 정방향만 (앞→뒤)
            forward_cross = (self.prev_distance_to_plane1 > 0.0) and (d1 <= 0.0)
            
            if forward_cross:
                crossed_p1 = True
                alpha1 = abs(self.prev_distance_to_plane1) / (abs(self.prev_distance_to_plane1) + abs(d1) + 1e-9)
                
                if len(self.traj_x) >= 2:
                    prev_pt = np.array([self.traj_x[-2], self.traj_y[-2], self.traj_z[-2]], dtype=np.float32)
                    curr_pt = np.array([self.traj_x[-1], self.traj_y[-1], self.traj_z[-1]], dtype=np.float32)
                    cross_point_p1 = prev_pt + alpha1 * (curr_pt - prev_pt)
                else:
                    cross_point_p1 = filtered_point.copy()
                    
        if self.prev_time_perf is not None and self.prev_distance_to_plane2 is not None:
            # plane2 통과 감지: 정방향만 (앞→뒤)
            forward_cross2 = (self.prev_distance_to_plane2 > 0.0) and (d2 <= 0.0)
            
            if forward_cross2:
                crossed_p2 = True
                alpha2 = abs(self.prev_distance_to_plane2) / (abs(self.prev_distance_to_plane2) + abs(d2) + 1e-9)
                
                if len(self.traj_x) >= 2:
                    prev_pt = np.array([self.traj_x[-2], self.traj_y[-2], self.traj_z[-2]], dtype=np.float32)
                    curr_pt = np.array([self.traj_x[-1], self.traj_y[-1], self.traj_z[-1]], dtype=np.float32)
                    cross_point_p2 = prev_pt + alpha2 * (curr_pt - prev_pt)
                else:
                    cross_point_p2 = filtered_point.copy()
        
        # === plane1 통과 처리 ===
        if crossed_p1 and (not self.plane1_crossed) and cross_point_p1 is not None:
            # 좌표 범위 검증: 넓은 범위 내인지 확인
            x, z = cross_point_p1[0], cross_point_p1[2]
            in_valid_range = (-0.8 < x < 0.8) and (-0.2 < z < 1.2)
            
            if in_valid_range:
                self.plane1_crossed = True
                self.t_cross_plane1 = self.prev_time_perf + alpha1 * (now_perf - self.prev_time_perf)
                self.plane1_in_zone = self.aruco_detector.is_point_in_strike_zone_3d(
                    cross_point_p1, self.ball_zone_corners
                )
                self.cross_point_p1_saved = cross_point_p1.copy()
                
                if self.first_plane_time is None:
                    self.first_plane_time = self.t_cross_plane1
                
                zone_status = "스트라이크존 내부 ✓" if self.plane1_in_zone else "스트라이크존 밖 ✗"
                print(f"[plane1 통과] X={cross_point_p1[0]:.3f}, Z={cross_point_p1[2]:.3f} → {zone_status}")
        
        # === plane2 통과 처리 ===
        if crossed_p2 and (not self.plane2_crossed) and cross_point_p2 is not None:
            # 좌표 범위 검증: 넓은 범위 내인지 확인
            x, z = cross_point_p2[0], cross_point_p2[2]
            in_valid_range = (-0.8 < x < 0.8) and (-0.2 < z < 1.2)
            
            if in_valid_range:
                self.plane2_crossed = True
                self.t_cross_plane2 = self.prev_time_perf + alpha2 * (now_perf - self.prev_time_perf)
                self.plane2_in_zone = self.aruco_detector.is_point_in_strike_zone_3d(
                    cross_point_p2, self.ball_zone_corners2
                )
                self.cross_point_p2_saved = cross_point_p2.copy()
                
                if self.first_plane_time is None:
                    self.first_plane_time = self.t_cross_plane2
                
                print(f"[plane2 통과] X={cross_point_p2[0]:.3f}, Z={cross_point_p2[2]:.3f} → 스트라이크존 내부: {self.plane2_in_zone}")
        
        # === 판정: 두 평면 모두 통과했을 때 ===
        cooldown_passed = (current_time - self.last_judgment_time) >= self.judgment_cooldown
        
        if self.plane1_crossed and self.plane2_crossed and cooldown_passed:
            # 스트라이크 조건: plane1 OR plane2 중 하나라도 스트라이크존 내부 통과
            # (야구 규칙: 공이 스트라이크존의 일부라도 통과하면 스트라이크)
            is_strike = self.plane1_in_zone or self.plane2_in_zone
            
            print(f"[판정] plane1(존내부:{self.plane1_in_zone}), plane2(존내부:{self.plane2_in_zone})")
            print(f"[판정] 결과: {'스트라이크' if is_strike else '볼'}")
            
            self.last_judgment_time = current_time
            
            # 속도 계산
            v_kmh = 0.0
            if self.t_cross_plane1 is not None and self.t_cross_plane2 is not None:
                dt = abs(self.t_cross_plane2 - self.t_cross_plane1)
                
                MAX_VALID_DT = 0.5
                MIN_VALID_DT = 0.003
                
                if dt < MIN_VALID_DT or dt > MAX_VALID_DT:
                    v_kmh = self.realtime_speed_kmh if hasattr(self, 'realtime_speed_kmh') and self.realtime_speed_kmh > 0 else 0.0
                else:
                    v_depth_mps = ZONE_DEPTH / dt
                    v_kmh = v_depth_mps * 3.6
                    if v_kmh > 200:
                        v_kmh = self.realtime_speed_kmh if hasattr(self, 'realtime_speed_kmh') and self.realtime_speed_kmh > 0 else 0.0
                
                if v_kmh > 0:
                    self.display_velocity = v_kmh
                    self.speed_updated.emit(v_kmh)
            
            # 스코어보드 업데이트
            if is_strike:
                self.scoreboard.add_strike()
            else:
                self.scoreboard.add_ball()
            
            # 교차 지점 투영
            if self.cross_point_p2_saved is not None:
                try:
                    point_on_plane2 = self.aruco_detector.project_point_onto_plane(
                        self.cross_point_p2_saved, p2_0, p2_1, p2_2
                    )
                except:
                    point_on_plane2 = self.cross_point_p2_saved.copy()
            elif self.cross_point_p1_saved is not None:
                point_on_plane2 = self.cross_point_p1_saved.copy()
            else:
                point_on_plane2 = filtered_point.copy()
            
            pitch_number = len(self.impact_points) + 1
            self.impact_points.append({
                'point_3d': point_on_plane2,
                'number': pitch_number
            })
            
            # 궤적 데이터 복사 (리셋 전에)
            trajectory_data = list(zip(self.traj_x, self.traj_y, self.traj_z))
            
            # 궤적 마지막 점을 최종 공 위치로 보정 (마커와 궤적 일치)
            if trajectory_data:
                final_x = float(point_on_plane2[0])
                final_y = float(point_on_plane2[1]) if len(point_on_plane2) > 1 else 0.0
                final_z = float(point_on_plane2[2])
                trajectory_data.append((final_x, final_y, final_z))
            
            # 디버깅 로그
            if trajectory_data:
                print(f"[DEBUG] 궤적 데이터: {len(trajectory_data)}개 점")
                print(f"[DEBUG] 궤적 마지막: ({trajectory_data[-1][0]:.3f}, {trajectory_data[-1][2]:.3f})")
                print(f"[DEBUG] 최종 위치: ({final_x:.3f}, {final_z:.3f})")
            
            # 시그널 발생 (궤적 포함)
            self.pitch_detected.emit({
                'is_strike': is_strike,
                'x': float(point_on_plane2[0]),
                'z': float(point_on_plane2[2]),
                'speed': float(v_kmh),
                'number': pitch_number,
                'trajectory': trajectory_data  # 3D 궤적 추가
            })
            
            # 상태 리셋
            self._reset_pitch_state()
            
            # 판정 완료 플래그 설정
            self.pitch_completed = True
            self.pitch_completed_time = current_time
        
        # === 한쪽 평면만 통과 후 옆으로 빠진 경우 → 볼 판정 ===
        # 단, 유효한 궤적 데이터가 있을 때만
        one_plane_only = (self.plane1_crossed and not self.plane2_crossed) or (self.plane2_crossed and not self.plane1_crossed)
        # plane1만 통과한 경우: d2가 음수가 되어야 함 (plane2까지 도달 후 판정)
        # plane2만 통과한 경우: d1이 음수가 되어야 함 (plane1까지 도달 후 판정)
        if self.plane1_crossed and not self.plane2_crossed:
            passed_through = (d2 < -0.05)  # plane2를 확실히 지나갔는지 확인
        elif self.plane2_crossed and not self.plane1_crossed:
            passed_through = (d1 < -0.05)  # plane1을 확실히 지나갔는지 확인
        else:
            passed_through = False
        has_valid_trajectory = len(self.traj_x) >= 3  # 최소 3개 이상의 궤적 포인트
        
        if one_plane_only and passed_through and cooldown_passed and has_valid_trajectory:
            if self.plane1_crossed and not self.plane2_crossed:
                print(f"[판정] plane1 통과 후 plane2 미통과 (옆으로 빠짐) → 볼")
            else:
                print(f"[판정] plane2 통과 후 plane1 미통과 (옆으로 빠짐) → 볼")
            
            self.last_judgment_time = current_time
            self.scoreboard.add_ball()
            
            point_on_plane2 = filtered_point.copy()
            # plane2 통과 좌표가 있으면 사용, 없으면 plane1 좌표 사용
            plane2_point = self.cross_point_p2_saved.copy() if self.cross_point_p2_saved is not None else (
                self.cross_point_p1_saved.copy() if self.cross_point_p1_saved is not None else filtered_point.copy()
            )
            
            pitch_number = len(self.impact_points) + 1
            self.impact_points.append({
                'point_3d': point_on_plane2,
                'plane2_point': plane2_point,
                'number': pitch_number
            })
            
            # 궤적 데이터 복사 (리셋 전에)
            trajectory_data = list(zip(self.traj_x, self.traj_y, self.traj_z))
            
            self.pitch_detected.emit({
                'is_strike': False,
                'x': float(point_on_plane2[0]),
                'z': float(point_on_plane2[2]),
                'speed': 0.0,
                'number': pitch_number,
                'trajectory': trajectory_data  # 3D 궤적 추가
            })
            
            self._reset_pitch_state()
            
            # 판정 완료 플래그 설정
            self.pitch_completed = True
            self.pitch_completed_time = current_time
        
        # 상태 업데이트
        self.prev_distance_to_plane1 = d1
        self.prev_distance_to_plane2 = d2
        self.prev_time_perf = now_perf
        
    def _reset_pitch_state(self):
        """투구 상태 리셋 - main_v7.py와 동일"""
        # plane1 상태
        self.plane1_crossed = False
        self.plane1_in_zone = False
        self.cross_point_p1_saved = None
        self.t_cross_plane1 = None
        
        # plane2 상태
        self.plane2_crossed = False
        self.plane2_in_zone = False
        self.cross_point_p2_saved = None
        self.t_cross_plane2 = None
        
        # 평면 거리 상태
        self.prev_distance_to_plane1 = None
        self.prev_distance_to_plane2 = None
        self.prev_time_perf = None
        
        # 시간 상태
        self.first_plane_time = None
        
        # 궤적 클리어
        self.traj_x.clear()
        self.traj_y.clear()
        self.traj_z.clear()
        
        # 속도 관련
        self.frame_speed_buffer.clear()
        self.prev_ball_pos_marker = None
        self.prev_ball_time = None
        
        # 볼 디텍터 리셋
        if hasattr(self, 'ball_detector') and self.ball_detector is not None:
            self.ball_detector.reset()
        
    def reset(self):
        """전체 리셋"""
        self._reset_state()
        self.traj_x.clear()
        self.traj_y.clear()
        self.traj_z.clear()
        self.impact_points.clear()
        self.scoreboard.reset()


class IntegratedMainWindow(MainWindow):
    """
    통합 메인 윈도우
    기존 MainWindow + PitchAnalyzer 통합
    """
    
    def __init__(self, camera_matrix, dist_coeffs):
        # 분석기 초기화
        self.pitch_analyzer = PitchAnalyzer(camera_matrix, dist_coeffs)
        
        # 부모 클래스 초기화
        super().__init__()
        
        # 시그널 연결
        self._connect_analyzer_signals()
        
        # 저장된 설정을 UI에 반영
        self._apply_saved_settings()
        
    def _apply_saved_settings(self):
        """저장된 설정을 UI에 반영"""
        # 시각화 설정을 SettingsDialog에 반영
        saved_vis = self.pitch_analyzer.vis_settings
        for key, value in saved_vis.items():
            if key in self.settings_dialog.vis_checkboxes:
                self.settings_dialog.vis_checkboxes[key].setChecked(value)
        
        # 게임 모드 설정 반영
        if self.pitch_analyzer.game_mode_enabled:
            self.game_mode_enabled = True
            self.target_zone = self.pitch_analyzer.target_zone
            self.game_widget.game_mode_cb.setChecked(True)
        
    def _connect_analyzer_signals(self):
        """분석기 시그널 연결"""
        self.pitch_analyzer.pitch_detected.connect(self._on_pitch_detected)
        self.pitch_analyzer.speed_updated.connect(self._on_speed_updated)
        
    def _on_frame_ready(self, frame):
        """프레임 수신 (오버라이드)"""
        # 분석기로 프레임 처리
        processed = self.pitch_analyzer.process_frame(frame)
        self.current_frame = processed
        
    def _on_vis_changed(self, settings):
        """시각화 설정 변경 (오버라이드)"""
        super()._on_vis_changed(settings)
        self.pitch_analyzer.update_vis_settings(settings)
    
    def _on_ball_color_changed(self, color_name):
        """공 색상 변경 (오버라이드)"""
        super()._on_ball_color_changed(color_name)
        self.pitch_analyzer.update_ball_color(color_name)
    
    def _on_game_mode_toggled(self, enabled):
        """게임 모드 토글 (오버라이드)"""
        super()._on_game_mode_toggled(enabled)
        # 분석기에 게임 모드 및 타겟 구역 전달
        self.pitch_analyzer.update_game_mode(enabled, self.target_zone)
    
    def _on_pitch_detected(self, pitch_data):
        """투구 감지 (오버라이드)"""
        super()._on_pitch_detected(pitch_data)
        # 게임 모드일 경우 새 타겟 구역을 분석기에 전달
        if self.game_mode_enabled:
            self.pitch_analyzer.update_game_mode(True, self.target_zone)
        
    def _on_reset(self):
        """리셋 (오버라이드)"""
        reply = QMessageBox.question(
            self, "리셋 확인",
            "모든 기록을 초기화하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.record_sheet.clear_records()
            self.pitch_list.clear_pitches()  # 투구 리스트도 초기화
            self.scoreboard.reset_all()
            self.stats_widget.reset()
            self.game_widget.reset()
            self.pitch_analyzer.reset()
            self.statusBar.showMessage("초기화됨")
            
    def _on_speed_updated(self, speed):
        """구속 업데이트"""
        self.statusBar.showMessage(f"구속: {speed:.1f} km/h")
    
    def closeEvent(self, event):
        """윈도우 종료 시 설정 저장"""
        # 현재 설정 저장
        self.pitch_analyzer.save_settings()
        print("[종료] 사용자 설정 저장 완료")
        
        # 부모 클래스 closeEvent 호출
        super().closeEvent(event)


def main():
    """메인 함수"""
    # 로그 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    print("=" * 50)
    print("  ⚾ AR Strike Zone Analyzer (GUI 버전)")
    print("=" * 50)
    
    # 캘리브레이션 로드 (기본)
    camera_matrix = None
    dist_coeffs = None
    
    try:
        calib_data = np.load(CALIBRATION_PATH)
        camera_matrix = calib_data["camera_matrix"]
        dist_coeffs = calib_data["dist_coeffs"]
        print(f"✅ 캘리브레이션 로드 완료: {CALIBRATION_PATH}")
    except Exception as e:
        print(f"⚠️ 캘리브레이션 데이터 로드 실패: {e}")
        print("기본값을 사용합니다.")
        # 기본 카메라 매트릭스 (추정값)
        camera_matrix = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        dist_coeffs = np.zeros(5, dtype=np.float32)
    
    # PyQt5 앱 시작
    app = QApplication(sys.argv)
    
    # 메인 윈도우 생성
    window = IntegratedMainWindow(camera_matrix, dist_coeffs)
    window.show()
    
    print("\n✅ GUI 시작됨")
    print("- 입력 소스: GUI 상단 '입력 소스' 메뉴에서 선택")
    print("- 왼쪽: 비디오 + 시각화")
    print("- 오른쪽: 기록지 / 게임 모드")
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
