# -*- coding: utf-8 -*-
"""
메인 GUI 애플리케이션
PyQt5 기반 스트라이크 존 분석 GUI
"""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QCheckBox, QGroupBox,
    QFrame, QSplitter, QComboBox, QSlider, QFileDialog,
    QMessageBox, QStatusBar, QTabWidget, QSizePolicy,
    QDialog, QDialogButtonBox, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor

from gui_config import (
    vis_config, game_config, record_config,
    scoreboard_config, window_config
)
from gui_widgets import (
    RecordSheet2D, Scoreboard, GameModeWidget, StatsWidget, PitchListWidget
)


class VisualizationSettingsDialog(QDialog):
    """
    시각화 설정 다이얼로그 (별도 창)
    """
    
    settingsChanged = pyqtSignal(dict)  # 설정 변경 시그널
    ballColorChanged = pyqtSignal(str)  # 공 색상 변경 시그널
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ 시각화 설정")
        self.setMinimumSize(280, 450)
        self.setModal(False)  # 모달리스 - 메인 창과 동시 조작 가능
        
        self.vis_checkboxes = {}
        self._init_ui()
        self._apply_style()
    
    def _apply_style(self):
        """다이얼로그 스타일 적용"""
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
            }
            QLabel {
                color: #ffffff;
            }
            QCheckBox {
                color: #ddd;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #666;
                background-color: #444;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #4a9eff;
                background-color: #4a9eff;
                border-radius: 3px;
            }
            QGroupBox {
                color: #fff;
                font-weight: bold;
                border: 1px solid #555;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
            }
        """)
        
    def _init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # 타이틀
        title = QLabel("⚙️ 시각화 설정")
        title.setFont(QFont(window_config.FONT_FAMILY, 14, QFont.Bold))
        title.setStyleSheet("color: #ffffff;")
        layout.addWidget(title)
        
        # 구분선
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #555;")
        layout.addWidget(line)
        
        # === 공 색상 선택 ===
        color_group = QGroupBox("🎾 공 색상")
        color_group.setStyleSheet("""
            QGroupBox {
                color: #fff;
                font-weight: bold;
                border: 1px solid #555;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
            }
        """)
        color_layout = QVBoxLayout(color_group)
        
        self.ball_color_combo = QComboBox()
        self.ball_color_combo.addItems(["형광공", "노랑공", "하얀공"])
        self.ball_color_combo.setStyleSheet("""
            QComboBox {
                background-color: #444;
                color: #fff;
                border: 1px solid #666;
                border-radius: 4px;
                padding: 5px;
                min-height: 25px;
            }
            QComboBox:hover {
                border: 1px solid #888;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #fff;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #444;
                color: #fff;
                selection-background-color: #666;
            }
        """)
        self.ball_color_combo.currentTextChanged.connect(self._on_ball_color_changed)
        color_layout.addWidget(self.ball_color_combo)
        
        # 색상 설명 라벨
        self.color_desc_label = QLabel("형광 연두색 공 (기본값)")
        self.color_desc_label.setStyleSheet("color: #aaa; font-size: 10px;")
        color_layout.addWidget(self.color_desc_label)
        
        layout.addWidget(color_group)
        
        # 구분선
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("background-color: #555;")
        layout.addWidget(line2)
        
        # 시각화 옵션들 - 그룹화
        # === 영역 표시 그룹 ===
        zone_group = QGroupBox("📍 영역 표시")
        zone_layout = QVBoxLayout(zone_group)
        
        zone_options = [
            ("zone", "스트라이크 존", True),
            ("plane1", "판정면 1 (앞)", True),
            ("plane2", "판정면 2 (뒤)", True),
            ("grid", "9분할 그리드", True),
        ]
        
        for key, label, default in zone_options:
            cb = QCheckBox(label)
            cb.setChecked(default)
            cb.stateChanged.connect(self._on_setting_changed)
            zone_layout.addWidget(cb)
            self.vis_checkboxes[key] = cb
        
        layout.addWidget(zone_group)
        
        # === 공 표시 그룹 ===
        ball_group = QGroupBox("⚾ 공 표시")
        ball_layout = QVBoxLayout(ball_group)
        
        ball_options = [
            ("trajectory", "공 궤적", True),
            ("ball_markers", "공 위치 마커 (넘버링)", True),
            ("speed", "구속 표시", True),
        ]
        
        for key, label, default in ball_options:
            cb = QCheckBox(label)
            cb.setChecked(default)
            cb.stateChanged.connect(self._on_setting_changed)
            ball_layout.addWidget(cb)
            self.vis_checkboxes[key] = cb
        
        layout.addWidget(ball_group)
        
        # === 기타 표시 그룹 ===
        misc_group = QGroupBox("🔧 기타")
        misc_layout = QVBoxLayout(misc_group)
        
        misc_options = [
            ("scoreboard", "스코어보드", True),
            ("aruco", "ArUco 마커", True),
            ("axes", "좌표축", False),
            ("fmo", "FMO 모드", False),
        ]
        
        for key, label, default in misc_options:
            cb = QCheckBox(label)
            cb.setChecked(default)
            cb.stateChanged.connect(self._on_setting_changed)
            misc_layout.addWidget(cb)
            self.vis_checkboxes[key] = cb
        
        layout.addWidget(misc_group)
        
        layout.addStretch()
        
        # 닫기 버튼
        close_btn = QPushButton("닫기")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                color: #fff;
                border: none;
                border-radius: 5px;
                padding: 8px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
        """)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
    
    def _on_ball_color_changed(self, color_name):
        """공 색상 변경 이벤트"""
        from config import BALL_COLOR_PRESETS
        if color_name in BALL_COLOR_PRESETS:
            desc = BALL_COLOR_PRESETS[color_name]["description"]
            self.color_desc_label.setText(desc)
            self.ballColorChanged.emit(color_name)
        
    def _on_setting_changed(self):
        """설정 변경 이벤트"""
        settings = {}
        for key, cb in self.vis_checkboxes.items():
            settings[key] = cb.isChecked()
        self.settingsChanged.emit(settings)
        
    def get_settings(self):
        """현재 설정 반환"""
        settings = {}
        for key, cb in self.vis_checkboxes.items():
            settings[key] = cb.isChecked()
        return settings


class VisualizationRenderer:
    """
    시각화 렌더링 클래스
    프레임에 스트라이크존, 판정면, 그리드 등을 그림
    """
    
    # 스트라이크 존 범위 (미터)
    ZONE_X_MIN, ZONE_X_MAX = -0.15, 0.15
    ZONE_Z_MIN, ZONE_Z_MAX = 0.25, 0.65
    
    # 색상 설정 (BGR)
    COLOR_ZONE = (0, 255, 0)       # 스트라이크존: 녹색
    COLOR_PLANE1 = (0, 255, 255)   # plane1: 시안
    COLOR_PLANE2 = (255, 100, 0)   # plane2: 파랑
    COLOR_GRID = (128, 128, 128)   # 그리드: 회색
    COLOR_TARGET = (0, 165, 255)   # 타겟: 주황
    
    @staticmethod
    def draw_zone(frame, corners_2d, color=None, thickness=2):
        """
        스트라이크 존 테두리 그리기
        corners_2d: 4개의 2D 투영 좌표 [(x,y), ...]
        """
        if color is None:
            color = VisualizationRenderer.COLOR_ZONE
        if corners_2d is not None and len(corners_2d) >= 4:
            pts = np.array(corners_2d, dtype=np.int32)
            cv2.polylines(frame, [pts], True, color, thickness)
    
    @staticmethod
    def draw_plane(frame, corners_2d, color, thickness=2, fill_alpha=0.0):
        """
        판정면 그리기 (plane1 또는 plane2)
        fill_alpha > 0 이면 반투명 채우기
        """
        if corners_2d is None or len(corners_2d) < 4:
            return
        pts = np.array(corners_2d, dtype=np.int32)
        
        if fill_alpha > 0:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)
        
        cv2.polylines(frame, [pts], True, color, thickness)
    
    @staticmethod
    def draw_9grid_on_plane2(frame, plane2_corners_2d, color=None, thickness=1):
        """
        plane2에만 9분할 그리드 그리기
        plane2_corners_2d: 4개의 2D 투영 좌표 [(x,y), ...]
                           순서: 좌상, 우상, 우하, 좌하 (또는 시계/반시계)
        """
        if color is None:
            color = VisualizationRenderer.COLOR_GRID
        if plane2_corners_2d is None or len(plane2_corners_2d) < 4:
            return
        
        pts = np.array(plane2_corners_2d, dtype=np.float32)
        # 코너: 0=좌상, 1=우상, 2=우하, 3=좌하 가정
        # 수직선 (3등분)
        for i in range(1, 3):
            t = i / 3.0
            # 상단 점: pts[0] ~ pts[1] 사이
            top = pts[0] * (1 - t) + pts[1] * t
            # 하단 점: pts[3] ~ pts[2] 사이
            bottom = pts[3] * (1 - t) + pts[2] * t
            cv2.line(frame, tuple(top.astype(int)), tuple(bottom.astype(int)), color, thickness)
        
        # 수평선 (3등분)
        for i in range(1, 3):
            t = i / 3.0
            # 좌측 점: pts[0] ~ pts[3] 사이
            left = pts[0] * (1 - t) + pts[3] * t
            # 우측 점: pts[1] ~ pts[2] 사이
            right = pts[1] * (1 - t) + pts[2] * t
            cv2.line(frame, tuple(left.astype(int)), tuple(right.astype(int)), color, thickness)
    
    @staticmethod
    def draw_target_zone_highlight(frame, plane2_corners_2d, target_zone, color=None, alpha=0.4):
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
        if color is None:
            color = VisualizationRenderer.COLOR_TARGET
        if plane2_corners_2d is None or len(plane2_corners_2d) < 4:
            return
        if target_zone is None or target_zone < 1 or target_zone > 9:
            return
        
        pts = np.array(plane2_corners_2d, dtype=np.float32)
        # 코너: 0=좌상, 1=우상, 2=우하, 3=좌하 가정
        
        # 타겟 구역의 행/열 (0-indexed)
        zone_idx = target_zone - 1
        row = zone_idx // 3  # 0, 1, 2
        col = zone_idx % 3   # 0, 1, 2
        
        # 구역의 4개 코너 계산
        def interpolate_point(t_col, t_row):
            """2D bilinear 보간으로 구역 내 점 계산"""
            # 상단 변에서의 점
            top = pts[0] * (1 - t_col) + pts[1] * t_col
            # 하단 변에서의 점
            bottom = pts[3] * (1 - t_col) + pts[2] * t_col
            # 수직 보간
            return top * (1 - t_row) + bottom * t_row
        
        # 구역 코너 (좌상, 우상, 우하, 좌하)
        c1 = col / 3.0
        c2 = (col + 1) / 3.0
        r1 = row / 3.0
        r2 = (row + 1) / 3.0
        
        zone_corners = np.array([
            interpolate_point(c1, r1),  # 좌상
            interpolate_point(c2, r1),  # 우상
            interpolate_point(c2, r2),  # 우하
            interpolate_point(c1, r2),  # 좌하
        ], dtype=np.int32)
        
        # 반투명 채우기
        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone_corners], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # 테두리
        cv2.polylines(frame, [zone_corners], True, color, 2)


class VideoThread(QThread):
    """비디오 프레임 처리 스레드"""
    
    frame_ready = pyqtSignal(np.ndarray)  # 원본 프레임
    processed_ready = pyqtSignal(np.ndarray)  # 처리된 프레임
    pitch_detected = pyqtSignal(dict)  # 투구 감지 결과
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.paused = False
        self.cap = None
        self.is_video_mode = False
        
    def set_source(self, source, is_video=False):
        """비디오 소스 설정"""
        self.is_video_mode = is_video
        if isinstance(source, str):
            # 비디오 파일 경로
            self.cap = cv2.VideoCapture(source)
        elif isinstance(source, int):
            # 카메라 인덱스
            self.cap = cv2.VideoCapture(source)
        elif isinstance(source, cv2.VideoCapture):
            # 이미 VideoCapture 객체
            self.cap = source
        else:
            # 기타 (예: 카메라 인덱스로 시도)
            self.cap = cv2.VideoCapture(int(source) if source else 0)
            
    def run(self):
        """스레드 실행"""
        self.running = True
        while self.running:
            if self.paused:
                self.msleep(50)
                continue
                
            if self.cap is None or not self.cap.isOpened():
                self.msleep(50)
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                if self.is_video_mode:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self.msleep(50)
                    continue
                    
            self.frame_ready.emit(frame)
            self.msleep(16)  # ~60fps
            
    def stop(self):
        """스레드 정지"""
        self.running = False
        self.wait()
        if self.cap:
            self.cap.release()


class VideoDisplay(QLabel):
    """비디오 디스플레이 위젯 (크기 축소)"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)  # 축소
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1a1a1a; border: 2px solid #333;")
        self.setText("비디오 소스를 선택하세요")
        self.setFont(QFont(window_config.FONT_FAMILY, 14))
        
    def update_frame(self, frame):
        """프레임 업데이트"""
        if frame is None:
            return
            
        # BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        
        # QImage로 변환
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 위젯 크기에 맞게 스케일
        scaled = q_img.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(QPixmap.fromImage(scaled))


class ControlPanel(QFrame):
    """컨트롤 패널 위젯 (간소화 버전)"""
    
    # 시그널
    sourceChanged = pyqtSignal(str)  # 소스 변경
    visualizationChanged = pyqtSignal(dict)  # 시각화 설정 변경
    gameModeToggled = pyqtSignal(bool)  # 게임 모드 토글
    resetRequested = pyqtSignal()  # 리셋 요청
    settingsToggled = pyqtSignal(bool)  # 설정 패널 토글
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self._init_ui()
        
    def _init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # === 소스 선택 섹션 ===
        source_group = QGroupBox("입력 소스")
        source_layout = QVBoxLayout(source_group)
        
        # 카메라/비디오 선택
        self.source_combo = QComboBox()
        self.source_combo.addItems(["카메라 0", "카메라 1", "비디오 파일..."])
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        source_layout.addWidget(self.source_combo)
        
        # 비디오 파일 선택 버튼
        self.file_btn = QPushButton("📁 파일 열기")
        self.file_btn.clicked.connect(self._on_file_open)
        source_layout.addWidget(self.file_btn)
        
        layout.addWidget(source_group)
        
        # === 게임 모드 섹션 ===
        game_group = QGroupBox("게임 모드")
        game_layout = QVBoxLayout(game_group)
        
        self.game_mode_cb = QCheckBox("🎯 타겟 연습 모드")
        self.game_mode_cb.stateChanged.connect(self._on_game_mode_changed)
        game_layout.addWidget(self.game_mode_cb)
        
        layout.addWidget(game_group)
        
        # === 제어 버튼 섹션 ===
        control_group = QGroupBox("제어")
        control_layout = QGridLayout(control_group)
        
        self.reset_btn = QPushButton("🔄 리셋")
        self.reset_btn.clicked.connect(self._on_reset)
        control_layout.addWidget(self.reset_btn, 0, 0)
        
        self.pause_btn = QPushButton("⏸ 일시정지")
        self.pause_btn.setCheckable(True)
        control_layout.addWidget(self.pause_btn, 0, 1)
        
        # 설정 패널 토글 버튼
        self.settings_btn = QPushButton("⚙️ 시각화 설정")
        self.settings_btn.setCheckable(True)
        self.settings_btn.clicked.connect(self._on_settings_toggled)
        control_layout.addWidget(self.settings_btn, 1, 0, 1, 2)
        
        layout.addWidget(control_group)
        
        # 스페이서
        layout.addStretch()
        
    def _on_source_changed(self, index):
        """소스 변경 이벤트"""
        if index == 0:
            self.sourceChanged.emit("camera:0")
        elif index == 1:
            self.sourceChanged.emit("camera:1")
        # index == 2는 파일 선택
            
    def _on_file_open(self):
        """파일 열기 다이얼로그"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "비디오 파일 선택", "",
            "비디오 파일 (*.mp4 *.avi *.mov *.mkv);;모든 파일 (*.*)"
        )
        if file_path:
            self.source_combo.setCurrentIndex(2)
            self.sourceChanged.emit(f"file:{file_path}")
            
    def _on_game_mode_changed(self, state):
        """게임 모드 토글"""
        self.gameModeToggled.emit(state == Qt.Checked)
        
    def _on_reset(self):
        """리셋 버튼"""
        self.resetRequested.emit()
        
    def _on_settings_toggled(self, checked):
        """설정 패널 토글"""
        self.settingsToggled.emit(checked)


class MainWindow(QMainWindow):
    """메인 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("⚾ AR Strike Zone Analyzer")
        self.setMinimumSize(window_config.WINDOW_WIDTH, window_config.WINDOW_HEIGHT)
        
        # 상태 변수
        self.vis_settings = {
            'zone': True, 'plane1': True, 'plane2': True,
            'grid': True, 'trajectory': True, 'speed': True,
            'scoreboard': True, 'aruco': True, 'axes': False,
            'fmo': False
        }
        self.game_mode_enabled = False
        self.current_frame = None
        self.target_zone = None  # 게임모드 타겟 구역
        
        # 시각화 렌더러
        self.renderer = VisualizationRenderer()
        
        # 테스트용 plane 좌표 (실제 AR 시스템 연동 시 업데이트됨)
        self.zone_corners_2d = None  # 스트라이크 존 2D 투영 좌표
        self.plane1_corners_2d = None  # plane1 2D 투영 좌표
        self.plane2_corners_2d = None  # plane2 2D 투영 좌표
        
        # 컴포넌트 초기화
        self._init_components()
        self._init_ui()
        self._connect_signals()
        
        # 스타일시트 적용
        self._apply_style()
        
    def _init_components(self):
        """컴포넌트 초기화"""
        # 비디오 스레드
        self.video_thread = VideoThread(self)
        
        # 타이머 (프레임 업데이트용)
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._process_frame)
        
    def _init_ui(self):
        """UI 초기화"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃 (수평: 왼쪽 + 오른쪽)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(8, 8, 8, 8)
        
        # === 왼쪽: 비디오 + 하단(컨트롤|스코어보드|통계) ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(8)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # 비디오 디스플레이
        self.video_display = VideoDisplay()
        self.video_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.video_display, stretch=4)
        
        # 왼쪽 하단: 컨트롤 | 스코어보드 | 통계
        left_bottom_panel = QWidget()
        left_bottom_layout = QHBoxLayout(left_bottom_panel)
        left_bottom_layout.setContentsMargins(0, 0, 0, 0)
        left_bottom_layout.setSpacing(8)
        
        # 컨트롤 패널
        self.control_panel = ControlPanel()
        left_bottom_layout.addWidget(self.control_panel, stretch=2)
        
        # 스코어보드
        self.scoreboard = Scoreboard()
        left_bottom_layout.addWidget(self.scoreboard, stretch=1)
        
        # 통계
        self.stats_widget = StatsWidget()
        left_bottom_layout.addWidget(self.stats_widget, stretch=1)
        
        left_layout.addWidget(left_bottom_panel, stretch=1)
        
        main_layout.addWidget(left_panel, stretch=2)  # 왼쪽 비율 축소 (3→2)
        
        # === 오른쪽: 기록지 탭 + 투구 리스트 ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # 탭 위젯 (기록지 / 게임모드)
        self.tab_widget = QTabWidget()
        
        # 기록지 탭
        record_tab = QWidget()
        record_layout = QVBoxLayout(record_tab)
        record_layout.setSpacing(5)
        
        # 기록지 (세로가 더 긴 비율)
        self.record_sheet = RecordSheet2D()
        self.record_sheet.setMinimumSize(280, 350)  # 세로가 더 긴 비율
        record_layout.addWidget(self.record_sheet, stretch=3)  # 기록지가 더 큼
        
        # 투구 리스트
        self.pitch_list = PitchListWidget()
        self.pitch_list.setMinimumHeight(120)  # 최소 높이 설정
        record_layout.addWidget(self.pitch_list, stretch=2)  # 기록지보다 작게
        
        self.tab_widget.addTab(record_tab, "📋 기록지")
        
        # 게임 모드 탭
        game_tab = QWidget()
        game_layout = QVBoxLayout(game_tab)
        self.game_widget = GameModeWidget()
        game_layout.addWidget(self.game_widget)
        game_layout.addStretch()
        self.tab_widget.addTab(game_tab, "🎯 게임 모드")
        
        right_layout.addWidget(self.tab_widget)
        
        # 시각화 설정 다이얼로그 (별도 창)
        self.settings_dialog = VisualizationSettingsDialog(self)
        
        main_layout.addWidget(right_panel, stretch=2)  # 오른쪽 비율 확대 (1→2)
        
        # 상태바
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("준비됨")
        
    def _connect_signals(self):
        """시그널 연결"""
        # 비디오 스레드
        self.video_thread.frame_ready.connect(self._on_frame_ready)
        self.video_thread.pitch_detected.connect(self._on_pitch_detected)
        
        # 컨트롤 패널
        self.control_panel.sourceChanged.connect(self._on_source_changed)
        self.control_panel.gameModeToggled.connect(self._on_game_mode_toggled)
        self.control_panel.resetRequested.connect(self._on_reset)
        self.control_panel.settingsToggled.connect(self._on_settings_toggled)
        
        # 일시정지 버튼
        self.control_panel.pause_btn.toggled.connect(self._on_pause_toggled)
        
        # 시각화 설정 다이얼로그
        self.settings_dialog.settingsChanged.connect(self._on_vis_changed)
        self.settings_dialog.ballColorChanged.connect(self._on_ball_color_changed)
        
        # 기록지 ↔ 투구 리스트 연동
        self.record_sheet.pitchSelected.connect(self._on_pitch_selected_from_sheet)
        self.pitch_list.pitchSelected.connect(self._on_pitch_selected_from_list)
        
    def _on_settings_toggled(self, visible):
        """설정 다이얼로그 토글"""
        if visible:
            self.settings_dialog.show()
            self.settings_dialog.raise_()
            self.settings_dialog.activateWindow()
        else:
            self.settings_dialog.hide()
        
    def _on_pitch_selected_from_sheet(self, number):
        """기록지에서 공 선택"""
        self.pitch_list.select_pitch(number)
        
    def _on_pitch_selected_from_list(self, number):
        """리스트에서 공 선택"""
        self.record_sheet.select_pitch(number)
        
    def _apply_style(self):
        """스타일시트 적용"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QLabel {
                color: #ffffff;
            }
            QGroupBox {
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 8px 16px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #555555;
            }
            QPushButton:checked {
                background-color: #0078d4;
            }
            QCheckBox {
                color: #ffffff;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QComboBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 5px;
            }
            QTabWidget::pane {
                border: 1px solid #555;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #3d3d3d;
                color: #ffffff;
                padding: 8px 16px;
                border: 1px solid #555;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
            }
            QStatusBar {
                background-color: #1a1a1a;
                color: #ffffff;
            }
        """)
        
    def _on_frame_ready(self, frame):
        """프레임 수신"""
        self.current_frame = frame
        
    def _process_frame(self):
        """프레임 처리 및 표시"""
        if self.current_frame is None:
            return
            
        display_frame = self.current_frame.copy()
        
        # === 시각화 적용 (각 옵션 독립적으로 처리) ===
        
        # 스트라이크 존 (zone 옵션 - plane1/plane2와 독립)
        if self.vis_settings.get('zone', True):
            if self.zone_corners_2d is not None:
                self.renderer.draw_zone(display_frame, self.zone_corners_2d)
        
        # plane1 (판정면 1 - 앞)
        if self.vis_settings.get('plane1', True):
            if self.plane1_corners_2d is not None:
                self.renderer.draw_plane(
                    display_frame, self.plane1_corners_2d,
                    color=VisualizationRenderer.COLOR_PLANE1, thickness=3
                )
        
        # plane2 (판정면 2 - 뒤)
        if self.vis_settings.get('plane2', True):
            if self.plane2_corners_2d is not None:
                self.renderer.draw_plane(
                    display_frame, self.plane2_corners_2d,
                    color=VisualizationRenderer.COLOR_PLANE2, thickness=3
                )
                
                # 9분할 그리드는 plane2에만 적용
                if self.vis_settings.get('grid', True):
                    self.renderer.draw_9grid_on_plane2(
                        display_frame, self.plane2_corners_2d,
                        thickness=1
                    )
                
                # 게임모드 타겟 구역 반투명 표시 (plane2에만)
                if self.game_mode_enabled and self.target_zone is not None:
                    self.renderer.draw_target_zone_highlight(
                        display_frame, self.plane2_corners_2d,
                        self.target_zone,
                        color=(0, 165, 255),  # 주황색
                        alpha=0.4
                    )
        
        # 비디오 디스플레이 업데이트
        self.video_display.update_frame(display_frame)
        
    def _on_source_changed(self, source):
        """소스 변경"""
        self.video_thread.stop()
        
        if source.startswith("camera:"):
            cam_id = int(source.split(":")[1])
            self.video_thread.set_source(cam_id, is_video=False)
            self.statusBar.showMessage(f"카메라 {cam_id} 연결됨")
        elif source.startswith("file:"):
            file_path = source[5:]
            self.video_thread.set_source(file_path, is_video=True)
            self.statusBar.showMessage(f"비디오 로드됨: {file_path}")
            
        self.video_thread.start()
        self.update_timer.start(16)  # ~60fps
        
    def _on_vis_changed(self, settings):
        """시각화 설정 변경"""
        self.vis_settings = settings
    
    def _on_ball_color_changed(self, color_name):
        """공 색상 변경 (하위 클래스에서 오버라이드)"""
        print(f"[MainWindow] 공 색상 변경: {color_name}")
        
    def _on_game_mode_toggled(self, enabled):
        """게임 모드 토글"""
        self.game_mode_enabled = enabled
        if enabled:
            self.tab_widget.setCurrentIndex(1)  # 게임 모드 탭으로 전환
            self.target_zone = self.game_widget.set_random_target()
            self.record_sheet.set_target_zone(self.target_zone)  # 기록지에 타겟 표시
            self.statusBar.showMessage(f"🎯 게임 모드 활성화됨 - 목표: {self.target_zone}구역")
        else:
            self.tab_widget.setCurrentIndex(0)  # 기록지 탭으로 전환
            self.target_zone = None
            self.record_sheet.set_target_zone(None)  # 타겟 해제
            self.statusBar.showMessage("게임 모드 비활성화됨")
            
    def _on_pause_toggled(self, paused):
        """일시정지 토글"""
        self.video_thread.paused = paused
        if paused:
            self.control_panel.pause_btn.setText("▶ 재생")
            self.statusBar.showMessage("일시정지됨")
        else:
            self.control_panel.pause_btn.setText("⏸ 일시정지")
            self.statusBar.showMessage("재생 중")
            
    def _on_reset(self):
        """리셋"""
        reply = QMessageBox.question(
            self, "리셋 확인",
            "모든 기록을 초기화하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.record_sheet.clear_records()
            self.pitch_list.clear_pitches()
            self.scoreboard.reset_all()
            self.stats_widget.reset()
            self.game_widget.reset()
            self.statusBar.showMessage("초기화됨")
            
    def _on_pitch_detected(self, pitch_data):
        """투구 감지"""
        is_strike = pitch_data.get('is_strike', False)
        x = pitch_data.get('x', 0)
        z = pitch_data.get('z', 0)
        speed = pitch_data.get('speed', 0)
        trajectory = pitch_data.get('trajectory', None)  # 궤적 데이터
        
        # 기록지 업데이트 (궤적 포함)
        number = self.record_sheet.add_record(x, z, is_strike, speed, trajectory)
        
        # 투구 리스트 업데이트
        self.pitch_list.add_pitch(number, is_strike, speed)
        
        # GUI 스코어보드 업데이트 (PitchAnalyzer의 scoreboard와 별도 객체)
        if is_strike:
            self.scoreboard.add_strike()
        else:
            self.scoreboard.add_ball()
            
        # 통계 업데이트
        self.stats_widget.add_pitch(is_strike, speed)
        
        # 게임 모드 타겟 구역을 기록지에 전달
        if self.game_mode_enabled:
            self.record_sheet.set_target_zone(self.target_zone)
            
            zone = self._calculate_zone(x, z)
            is_hit = self.game_widget.check_hit(zone)
            if is_hit:
                self.statusBar.showMessage(f"🎯 명중! 구역 {zone}")
            else:
                self.statusBar.showMessage(f"❌ 실패 (구역 {zone})")
            self.target_zone = self.game_widget.set_random_target()  # 다음 타겟
            self.record_sheet.set_target_zone(self.target_zone)  # 새 타겟 표시
            
    def _calculate_zone(self, x, z):
        """X, Z 좌표로 9분할 구역 계산"""
        # 스트라이크 존 범위
        x_min, x_max = -0.15, 0.15
        z_min, z_max = 0.25, 0.65
        
        # 정규화
        nx = (x - x_min) / (x_max - x_min)
        nz = (z - z_min) / (z_max - z_min)
        
        # 구역 계산 (1~9)
        col = min(2, max(0, int(nx * 3)))
        row = min(2, max(0, int((1 - nz) * 3)))  # Z는 위가 높음
        
        return row * 3 + col + 1
        
    def closeEvent(self, event):
        """종료 이벤트"""
        self.video_thread.stop()
        self.update_timer.stop()
        event.accept()


def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    
    # 폰트 설정
    font = QFont(window_config.FONT_FAMILY, window_config.FONT_SIZE_NORMAL)
    app.setFont(font)
    
    # 메인 윈도우 생성 및 표시
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
