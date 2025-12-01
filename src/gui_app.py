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
    QMessageBox, QStatusBar, QTabWidget, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor

from gui_config import (
    vis_config, game_config, record_config,
    scoreboard_config, window_config
)
from gui_widgets import (
    RecordSheet2D, Scoreboard, GameModeWidget, StatsWidget
)


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
            self.cap = cv2.VideoCapture(source)
        else:
            self.cap = source
            
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
    """비디오 디스플레이 위젯"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
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
    """컨트롤 패널 위젯"""
    
    # 시그널
    sourceChanged = pyqtSignal(str)  # 소스 변경
    visualizationChanged = pyqtSignal(dict)  # 시각화 설정 변경
    gameModeToggled = pyqtSignal(bool)  # 게임 모드 토글
    resetRequested = pyqtSignal()  # 리셋 요청
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self._init_ui()
        
    def _init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
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
        
        # === 시각화 옵션 섹션 ===
        vis_group = QGroupBox("시각화 옵션")
        vis_layout = QVBoxLayout(vis_group)
        
        self.vis_checkboxes = {}
        vis_options = [
            ("zone", "스트라이크 존", True),
            ("plane1", "판정면 1 (앞)", True),
            ("plane2", "판정면 2 (뒤)", True),
            ("grid", "9분할 그리드", True),
            ("trajectory", "공 궤적", True),
            ("speed", "구속 표시", True),
            ("aruco", "ArUco 마커", True),
            ("axes", "좌표축", False),
        ]
        
        for key, label, default in vis_options:
            cb = QCheckBox(label)
            cb.setChecked(default)
            cb.stateChanged.connect(self._on_vis_changed)
            vis_layout.addWidget(cb)
            self.vis_checkboxes[key] = cb
            
        layout.addWidget(vis_group)
        
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
            
    def _on_vis_changed(self):
        """시각화 옵션 변경 이벤트"""
        vis_settings = {}
        for key, cb in self.vis_checkboxes.items():
            vis_settings[key] = cb.isChecked()
        self.visualizationChanged.emit(vis_settings)
        
    def _on_game_mode_changed(self, state):
        """게임 모드 토글"""
        self.gameModeToggled.emit(state == Qt.Checked)
        
    def _on_reset(self):
        """리셋 버튼"""
        self.resetRequested.emit()


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
            'aruco': True, 'axes': False
        }
        self.game_mode_enabled = False
        self.current_frame = None
        
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
        
        # 메인 레이아웃
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # === 왼쪽: 비디오 + 하단 스코어보드 ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        
        # 비디오 디스플레이
        self.video_display = VideoDisplay()
        self.video_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.video_display, stretch=4)
        
        # 하단 패널 (스코어보드 + 통계)
        bottom_panel = QWidget()
        bottom_layout = QHBoxLayout(bottom_panel)
        
        # 스코어보드
        self.scoreboard = Scoreboard()
        bottom_layout.addWidget(self.scoreboard)
        
        # 통계
        self.stats_widget = StatsWidget()
        bottom_layout.addWidget(self.stats_widget)
        
        left_layout.addWidget(bottom_panel, stretch=1)
        
        main_layout.addWidget(left_panel, stretch=3)
        
        # === 오른쪽: 기록지 + 컨트롤 + 게임모드 ===
        right_panel = QWidget()
        right_panel.setFixedWidth(window_config.RIGHT_PANEL_WIDTH)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)
        
        # 탭 위젯 (기록지 / 게임모드)
        self.tab_widget = QTabWidget()
        
        # 기록지 탭
        record_tab = QWidget()
        record_layout = QVBoxLayout(record_tab)
        self.record_sheet = RecordSheet2D()
        record_layout.addWidget(self.record_sheet)
        record_layout.addStretch()
        self.tab_widget.addTab(record_tab, "📋 기록지")
        
        # 게임 모드 탭
        game_tab = QWidget()
        game_layout = QVBoxLayout(game_tab)
        self.game_widget = GameModeWidget()
        game_layout.addWidget(self.game_widget)
        game_layout.addStretch()
        self.tab_widget.addTab(game_tab, "🎯 게임 모드")
        
        right_layout.addWidget(self.tab_widget)
        
        # 컨트롤 패널
        self.control_panel = ControlPanel()
        right_layout.addWidget(self.control_panel)
        
        main_layout.addWidget(right_panel)
        
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
        self.control_panel.visualizationChanged.connect(self._on_vis_changed)
        self.control_panel.gameModeToggled.connect(self._on_game_mode_toggled)
        self.control_panel.resetRequested.connect(self._on_reset)
        
        # 일시정지 버튼
        self.control_panel.pause_btn.toggled.connect(self._on_pause_toggled)
        
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
            
        # 여기서 시각화 처리를 추가할 수 있음
        # (현재는 원본 프레임만 표시)
        display_frame = self.current_frame.copy()
        
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
        
    def _on_game_mode_toggled(self, enabled):
        """게임 모드 토글"""
        self.game_mode_enabled = enabled
        if enabled:
            self.tab_widget.setCurrentIndex(1)  # 게임 모드 탭으로 전환
            self.game_widget.set_random_target()
            self.statusBar.showMessage("🎯 게임 모드 활성화됨")
        else:
            self.tab_widget.setCurrentIndex(0)  # 기록지 탭으로 전환
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
        
        # 기록지 업데이트
        self.record_sheet.add_record(x, z, is_strike, speed)
        
        # 스코어보드 업데이트
        if is_strike:
            self.scoreboard.add_strike()
        else:
            self.scoreboard.add_ball()
            
        # 통계 업데이트
        self.stats_widget.add_pitch(is_strike, speed)
        
        # 게임 모드일 경우 타겟 체크
        if self.game_mode_enabled:
            zone = self._calculate_zone(x, z)
            is_hit = self.game_widget.check_hit(zone)
            if is_hit:
                self.statusBar.showMessage(f"🎯 명중! 구역 {zone}")
            else:
                self.statusBar.showMessage(f"❌ 실패 (구역 {zone})")
            self.game_widget.set_random_target()  # 다음 타겟
            
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
