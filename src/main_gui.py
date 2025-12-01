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


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


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
        
        # 칼만 필터
        self.kalman_tracker = KalmanFilter3D()
        
        # 스코어보드
        self.scoreboard = BaseballScoreboard(width=0.3, height=0.24, offset_x=-0.4, offset_y=0.0, offset_z=0.1)
        
        # 존 설정
        self.ball_zone_corners = BALL_ZONE_CORNERS.copy()
        self.ball_zone_corners2 = BALL_ZONE_CORNERS.copy()
        self.ball_zone_corners2[:, 1] += ZONE_DEPTH
        self.box_corners_3d = self.aruco_detector.get_box_corners_3d(BOX_MIN, BOX_MAX)
        
        # 평면 법선 방향
        self.desired_depth_axis = np.array([0, 1, 0], dtype=np.float32)
        
        # 상태 변수 초기화
        self._reset_state()
        
        # 궤적 버퍼
        self.traj_x, self.traj_y, self.traj_z = [], [], []
        
        # 기록된 투구들
        self.impact_points = []
        
        # 시각화 설정
        self.vis_settings = {
            'zone': True, 'plane1': True, 'plane2': True,
            'grid': True, 'trajectory': True, 'speed': True,
            'aruco': True, 'axes': False
        }
        
        # 왜곡 보정
        self.undistort_map1 = None
        self.undistort_map2 = None
        
    def _reset_state(self):
        """상태 초기화"""
        self.prev_distance_to_plane1 = None
        self.prev_distance_to_plane2 = None
        self.prev_time_perf = None
        self.t_cross_plane1 = None
        self.t_cross_plane2 = None
        self.plane1_crossed = False
        self.plane1_in_zone = False
        self.cross_point_p1_saved = None
        self.display_velocity = 0.0
        
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
            
            # 시각화: 스트라이크 존 (plane1)
            if self.vis_settings.get('zone', True) or self.vis_settings.get('plane1', True):
                projected_points = self.aruco_detector.project_points(self.ball_zone_corners, rvec, tvec)
                cv2.polylines(overlay_frame, [projected_points], True, (0, 255, 255), 3)
            
            # 시각화: plane2
            if self.vis_settings.get('plane2', True):
                projected_points2 = self.aruco_detector.project_points(self.ball_zone_corners2, rvec, tvec)
                cv2.polylines(overlay_frame, [projected_points2], True, (255, 100, 0), 3)
            
            # 9분할 그리드
            if self.vis_settings.get('grid', True):
                self._draw_grid(overlay_frame, rvec, tvec)
            
            # 박스 그리기
            pts2d_box = self.aruco_detector.project_points(self.box_corners_3d, rvec, tvec)
            self.aruco_detector.draw_3d_box(overlay_frame, pts2d_box, BOX_EDGES, color=(0,0,0), thickness=2)
            
            # 스코어보드
            self.scoreboard.draw(overlay_frame, self.aruco_detector, rvec, tvec)
            
            # 기록된 투구 지점 표시
            for impact in self.impact_points:
                self.aruco_detector.draw_impact_point_on_plane(
                    overlay_frame,
                    impact['point_3d'],
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
            
            # 공 검출
            center, radius, detect_method = self.ball_detector.detect(frame)
            
            if center and radius > 0.4:
                # 공 표시
                self.ball_detector.draw_ball(overlay_frame, center, radius, detect_method)
                
                # 3D 위치 추정
                estimated_Z = (self.camera_matrix[0, 0] * BALL_RADIUS_REAL) / max(1e-6, radius)
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
                
                # 궤적 기록
                self.traj_x.append(filtered_point[0])
                self.traj_y.append(filtered_point[1])
                self.traj_z.append(filtered_point[2])
                
                # 통과 감지 및 판정
                self._check_crossing(filtered_point, rvec, tvec)
        
        # 구속 표시
        if self.vis_settings.get('speed', True) and self.display_velocity > 0:
            self._draw_speed(overlay_frame)
        
        self.frame_processed.emit(overlay_frame)
        return overlay_frame
        
    def _draw_grid(self, frame, rvec, tvec):
        """9분할 그리드 그리기"""
        zone = self.ball_zone_corners
        
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
        """평면 통과 감지 및 판정"""
        p1_0, p1_1, p1_2 = self.ball_zone_corners[0], self.ball_zone_corners[1], self.ball_zone_corners[2]
        p2_0, p2_1, p2_2 = self.ball_zone_corners2[0], self.ball_zone_corners2[1], self.ball_zone_corners2[2]
        
        d1 = self.aruco_detector.signed_distance_to_plane_oriented(
            filtered_point, p1_0, p1_1, p1_2, desired_dir=self.desired_depth_axis
        )
        d2 = self.aruco_detector.signed_distance_to_plane_oriented(
            filtered_point, p2_0, p2_1, p2_2, desired_dir=self.desired_depth_axis
        )
        
        now_perf = time.perf_counter()
        
        crossed_p1 = False
        crossed_p2 = False
        cross_point_p1 = None
        cross_point_p2 = None
        alpha1, alpha2 = 0.0, 0.0
        
        if self.prev_time_perf is not None and self.prev_distance_to_plane1 is not None:
            # plane1 통과 감지
            if (self.prev_distance_to_plane1 > 0.0) and (d1 <= 0.0):
                crossed_p1 = True
                alpha1 = self.prev_distance_to_plane1 / (self.prev_distance_to_plane1 - d1 + 1e-9)
                
                if len(self.traj_x) >= 2:
                    prev_pt = np.array([self.traj_x[-2], self.traj_y[-2], self.traj_z[-2]], dtype=np.float32)
                    curr_pt = np.array([self.traj_x[-1], self.traj_y[-1], self.traj_z[-1]], dtype=np.float32)
                    cross_point_p1 = prev_pt + alpha1 * (curr_pt - prev_pt)
                else:
                    cross_point_p1 = filtered_point.copy()
                    
        if self.prev_time_perf is not None and self.prev_distance_to_plane2 is not None:
            # plane2 통과 감지
            if (self.prev_distance_to_plane2 > 0.0) and (d2 <= 0.0):
                crossed_p2 = True
                alpha2 = self.prev_distance_to_plane2 / (self.prev_distance_to_plane2 - d2 + 1e-9)
                
                if len(self.traj_x) >= 2:
                    prev_pt = np.array([self.traj_x[-2], self.traj_y[-2], self.traj_z[-2]], dtype=np.float32)
                    curr_pt = np.array([self.traj_x[-1], self.traj_y[-1], self.traj_z[-1]], dtype=np.float32)
                    cross_point_p2 = prev_pt + alpha2 * (curr_pt - prev_pt)
                else:
                    cross_point_p2 = filtered_point.copy()
        
        # plane1 통과 처리
        if crossed_p1 and (not self.plane1_crossed) and cross_point_p1 is not None:
            self.plane1_crossed = True
            self.t_cross_plane1 = self.prev_time_perf + alpha1 * (now_perf - self.prev_time_perf)
            self.plane1_in_zone = self.aruco_detector.is_point_in_strike_zone_3d(
                cross_point_p1, self.ball_zone_corners
            )
            self.cross_point_p1_saved = cross_point_p1.copy()
        
        # plane2 통과 처리 (판정)
        if crossed_p2 and cross_point_p2 is not None:
            self.t_cross_plane2 = self.prev_time_perf + alpha2 * (now_perf - self.prev_time_perf)
            plane2_in_zone = self.aruco_detector.is_point_in_strike_zone_3d(
                cross_point_p2, self.ball_zone_corners2
            )
            
            is_strike = self.plane1_crossed and self.plane1_in_zone and plane2_in_zone
            
            # 속도 계산
            v_kmh = 0.0
            if self.plane1_crossed and (self.t_cross_plane1 is not None):
                dt = max(1e-6, (self.t_cross_plane2 - self.t_cross_plane1))
                v_depth_mps = ZONE_DEPTH / dt
                v_kmh = v_depth_mps * 3.6
                self.display_velocity = v_kmh
                self.speed_updated.emit(v_kmh)
            
            # 스코어보드 업데이트
            if is_strike:
                self.scoreboard.add_strike()
            else:
                self.scoreboard.add_ball()
            
            # 교차 지점 투영
            try:
                point_on_plane2 = self.aruco_detector.project_point_onto_plane(
                    cross_point_p2, p2_0, p2_1, p2_2
                )
            except:
                point_on_plane2 = cross_point_p2.copy()
            
            pitch_number = len(self.impact_points) + 1
            self.impact_points.append({
                'point_3d': point_on_plane2,
                'number': pitch_number
            })
            
            # 시그널 발생
            self.pitch_detected.emit({
                'is_strike': is_strike,
                'x': float(point_on_plane2[0]),
                'z': float(point_on_plane2[2]),
                'speed': float(v_kmh),
                'number': pitch_number
            })
            
            # 상태 리셋
            self._reset_pitch_state()
            
        # plane1 통과 후 plane2 미통과 (옆으로 빠진 경우)
        elif self.plane1_crossed and (not crossed_p2) and (d2 < -0.05):
            v_kmh = 0.0
            self.scoreboard.add_ball()
            
            point_on_plane2 = filtered_point.copy()
            pitch_number = len(self.impact_points) + 1
            self.impact_points.append({
                'point_3d': point_on_plane2,
                'number': pitch_number
            })
            
            self.pitch_detected.emit({
                'is_strike': False,
                'x': float(point_on_plane2[0]),
                'z': float(point_on_plane2[2]),
                'speed': 0.0,
                'number': pitch_number
            })
            
            self._reset_pitch_state()
        
        # 상태 업데이트
        self.prev_distance_to_plane1 = d1
        self.prev_distance_to_plane2 = d2
        self.prev_time_perf = now_perf
        
    def _reset_pitch_state(self):
        """투구 상태 리셋"""
        self.plane1_crossed = False
        self.plane1_in_zone = False
        self.cross_point_p1_saved = None
        self.t_cross_plane1 = None
        self.t_cross_plane2 = None
        self.prev_distance_to_plane1 = None
        self.prev_distance_to_plane2 = None
        self.prev_time_perf = None
        self.traj_x.clear()
        self.traj_y.clear()
        self.traj_z.clear()
        
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
            self.scoreboard.reset_all()
            self.stats_widget.reset()
            self.game_widget.reset()
            self.pitch_analyzer.reset()
            self.statusBar.showMessage("초기화됨")
            
    def _on_speed_updated(self, speed):
        """구속 업데이트"""
        self.statusBar.showMessage(f"구속: {speed:.1f} km/h")


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
    
    # 입력 소스 선택
    print("\n입력 소스 선택:")
    print("1: 카메라")
    print("2: 비디오 파일")
    user_input = input("> ").strip()
    
    camera_matrix = None
    dist_coeffs = None
    video_source = None
    is_video_mode = False
    
    if user_input == "1":
        # 카메라 모드
        shutdown_event = threading.Event()
        camera_manager = CameraManager(shutdown_event)
        selected_camera = camera_manager.create_camera_selection_gui()
        
        if selected_camera is None:
            print("카메라가 선택되지 않았습니다.")
            return
            
        if not camera_manager.open_camera(selected_camera):
            print(f"카메라 {selected_camera}를 열 수 없습니다.")
            return
            
        if not camera_manager.load_calibration(CALIBRATION_PATH):
            print("캘리브레이션 데이터 로드 실패")
            return
            
        camera_matrix = camera_manager.camera_matrix
        dist_coeffs = camera_manager.dist_coeffs
        video_source = selected_camera
        
    elif user_input == "2":
        # 비디오 모드
        video_path = input("비디오 파일 경로 (기본: ./video/video_BBS.mp4): ").strip()
        if not video_path:
            video_path = "./video/video_BBS.mp4"
            
        if not os.path.exists(video_path):
            print(f"파일을 찾을 수 없습니다: {video_path}")
            return
            
        # 캘리브레이션 로드
        try:
            calib_data = np.load(CALIBRATION_PATH)
            camera_matrix = calib_data["camera_matrix"]
            dist_coeffs = calib_data["dist_coeffs"]
        except Exception as e:
            print(f"캘리브레이션 데이터 로드 실패: {e}")
            return
            
        video_source = video_path
        is_video_mode = True
    else:
        print("잘못된 입력입니다.")
        return
    
    # PyQt5 앱 시작
    app = QApplication(sys.argv)
    
    # 메인 윈도우 생성
    window = IntegratedMainWindow(camera_matrix, dist_coeffs)
    
    # 왜곡 보정 설정
    cap_temp = cv2.VideoCapture(video_source)
    ret, temp_frame = cap_temp.read()
    if ret:
        h, w = temp_frame.shape[:2]
        window.pitch_analyzer.setup_undistort(w, h)
    cap_temp.release()
    
    # 소스 설정 및 시작
    if is_video_mode:
        window.control_panel.sourceChanged.emit(f"file:{video_source}")
    else:
        window.control_panel.sourceChanged.emit(f"camera:{video_source}")
    
    window.show()
    
    print("\n✅ GUI 시작됨")
    print("- 왼쪽: 비디오 + 시각화")
    print("- 오른쪽: 기록지 / 게임 모드")
    print("- 하단: 스코어보드 + 통계")
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
