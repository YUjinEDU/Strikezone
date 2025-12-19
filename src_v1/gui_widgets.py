# -*- coding: utf-8 -*-
"""
GUI 위젯 모듈
2D 기록지, 스코어보드, 9분할 뷰 등의 커스텀 위젯
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QFrame, QSizePolicy, QListWidget, QListWidgetItem,
    QScrollArea
)
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal, QTimer, QPropertyAnimation, QVariantAnimation
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, 
    QPainterPath, QLinearGradient, QRadialGradient
)
import math

from gui_config import (
    record_config, scoreboard_config, 
    game_config, window_config
)
from config import ZONE_HALF_WIDTH, ZONE_TOP, ZONE_BOTTOM


# gui_widgets.py 의 RecordSheet2D 클래스를 아래 코드로 교체하세요.

class RecordSheet2D(QWidget):
    """
    2D 기록지 위젯 (투수 시점 - Pitcher's View)
    투수가 포수/타자를 바라보는 시점입니다.
    - 공의 시작점(투수 손): 화면 앞쪽 (크고, 넓게 퍼짐)
    - 공의 도착점(스트라이크 존): 화면 안쪽 (정확한 위치)
    """
    
    pitchSelected = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.records = []
        self.setMinimumSize(280, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # === [설정] 뷰 파라미터 ===
        # 스트라이크 존 표시 범위 (미터)
        self.zone_x_min = -0.6
        self.zone_x_max = 0.6
        self.zone_z_min = 0.0
        self.zone_z_max = 1.2
        
        # 실제 스트라이크 존 (규격)
        self.strike_x_min = -ZONE_HALF_WIDTH
        self.strike_x_max = ZONE_HALF_WIDTH
        self.strike_z_min = ZONE_BOTTOM
        self.strike_z_max = ZONE_TOP
        
        # 깊이 설정 (Y축: 투수 < 0, 포수 ~ 0)
        # 투수의 릴리스 포인트(약 -1.5m ~ -2m)부터 포수(0m)까지
        self.depth_y_release = -2.0 
        self.depth_y_plate = 0.0
        
        self.selected_pitch = None
        self.target_zone = None
        self.trajectory_points_count = 20  # 궤적 포인트 개수 늘림 (더 부드럽게)
        
        # 애니메이션
        self.animation_progress = 1.0
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._update_animation)
        self.is_animating = False

    # ... [add_record, clear_records 등 데이터 관리 메서드는 기존과 동일 유지] ...
    def add_record(self, x, z, is_strike, speed=None, trajectory=None):
        """기록 추가 (기존 코드 유지)"""
        number = len(self.records) + 1
        x_clamped = max(self.zone_x_min, min(self.zone_x_max, x))
        z_clamped = max(self.zone_z_min, min(self.zone_z_max, z))
        
        traj_3d = []
        if trajectory and len(trajectory) > 0:
            n = min(self.trajectory_points_count, len(trajectory))
            for pt in trajectory[-n:]:
                if len(pt) >= 3:
                    traj_3d.append(pt) # 있는 그대로 저장
                    
        self.records.append((x_clamped, z_clamped, is_strike, number, speed, traj_3d))
        
        if len(self.records) > record_config.MAX_DISPLAY_COUNT:
            self.records.pop(0)
            for i, (rx, rz, ris_s, _, rspd, rtraj) in enumerate(self.records):
                self.records[i] = (rx, rz, ris_s, i + 1, rspd, rtraj)
        
        self.selected_pitch = None
        self._start_animation()
        self.update()
        return number

    def clear_records(self):
        self.records = []
        self.selected_pitch = None
        self.target_zone = None
        self.update()

    def _start_animation(self):
        self.animation_progress = 0.0
        self.is_animating = True
        self.animation_timer.start(record_config.TRAJECTORY_ANIMATION_SPEED)

    def _update_animation(self):
        self.animation_progress += 0.05 # 속도 조절
        if self.animation_progress >= 1.0:
            self.animation_progress = 1.0
            self.is_animating = False
            self.animation_timer.stop()
        self.update()

    def set_target_zone(self, zone):
        self.target_zone = zone
        self.update()

    def select_pitch(self, number):
        self.selected_pitch = number
        self.update()

    # === [핵심 수정] 좌표 변환 로직 ===
    def _world_to_widget_flat(self, x, z):
        """평면(스트라이크존 위치)에서의 2D 좌표 변환"""
        margin = record_config.MARGIN
        w = self.width() - 2 * margin
        h = self.height() - 2 * margin
        
        # 정규화 (0~1)
        nx = (x - self.zone_x_min) / (self.zone_x_max - self.zone_x_min)
        nz = (z - self.zone_z_min) / (self.zone_z_max - self.zone_z_min)
        
        # 화면 좌표 (Z는 위가 0이므로 1-nz)
        wx = margin + nx * w
        wy = margin + (1 - nz) * h
        return wx, wy

    def _perspective_transform(self, x, y, z):
        """
        [3D 투수 시점 변환]
        공이 Y축(깊이)에서 멀어질수록(투수 쪽, Y < 0) 화면 중앙(소실점)에서 멀어지게 처리하여
        공이 화면 밖에서 안으로 들어오는 효과를 냄.
        """
        # 1. 기본 평면 좌표 (도착지점 기준)
        base_wx, base_wy = self._world_to_widget_flat(x, z)
        
        # 2. 소실점 (Vanishing Point): 화면 정중앙
        center_x = self.width() / 2
        center_y = self.height() / 2
        
        # 3. 깊이 비율 계산 (0: 포수/존, 1: 투수/릴리스)
        # y는 0(존) ~ -2.0(투수) 범위라고 가정
        depth = abs(y) 
        # depth가 클수록(투수쪽) 왜곡을 심하게 줌
        
        # 4. 원근 왜곡 계수 (Zoom Factor)
        # 투수 쪽에 있을수록 좌표를 소실점 바깥으로 밀어냄 (광각 렌즈 효과)
        # 1.0 = 존 위치, > 1.0 = 투수 위치
        zoom = 1.0 + (depth * 0.4) # 0.4는 원근감 강도 (조절 가능)
        
        # 5. 소실점 기준으로 좌표 확장
        final_x = center_x + (base_wx - center_x) * zoom
        final_y = center_y + (base_wy - center_y) * zoom
        
        # 6. 스케일(크기) 계수 반환 (투수 쪽일수록 큼)
        scale = 1.0 + (depth * 0.5)
        
        return final_x, final_y, scale

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 배경
        painter.fillRect(self.rect(), QColor(*record_config.COLOR_BACKGROUND))
        
        # === 1. 스트라이크 존 그리기 (가장 뒤쪽 레이어) ===
        # 존 좌표 계산
        zl, zt = self._world_to_widget_flat(self.strike_x_min, self.strike_z_max)
        zr, zb = self._world_to_widget_flat(self.strike_x_max, self.strike_z_min)
        zw, zh = zr - zl, zb - zt
        
        # 타겟 구역 (게임모드)
        if self.target_zone:
            self._draw_target_zone(painter, zl, zt, zw, zh)
            
        # 존 내부 채우기
        painter.fillRect(QRectF(zl, zt, zw, zh), QColor(*record_config.COLOR_ZONE_FILL, 150))
        
        # 9분할 그리드
        painter.setPen(QPen(QColor(*record_config.COLOR_GRID), 1, Qt.DashLine))
        for i in range(1, 3):
            # 수직
            x = zl + zw * i / 3
            painter.drawLine(QPointF(x, zt), QPointF(x, zb))
            # 수평
            y = zt + zh * i / 3
            painter.drawLine(QPointF(zl, y), QPointF(zr, y))
            
        # 존 외곽선
        painter.setPen(QPen(QColor(*record_config.COLOR_ZONE_BORDER), 2))
        painter.drawRect(QRectF(zl, zt, zw, zh))
        
        # === 2. 궤적 그리기 (중간 레이어) ===
        # [수정됨] 주석 해제 및 로직 개선
        latest_idx = len(self.records) 
        for x, z, is_strike, number, speed, trajectory in self.records:
            is_selected = (number == self.selected_pitch)
            is_latest = (number == latest_idx)
            
            # 선택된 공이 없으면 최신 공, 있으면 선택된 공만 궤적 표시
            should_draw_traj = is_selected or (self.selected_pitch is None and is_latest)
            
            if should_draw_traj and trajectory and len(trajectory) > 1:
                 # 애니메이션 진행률
                progress = self.animation_progress if (is_latest and not self.selected_pitch) else 1.0
                self._draw_trajectory_3d(painter, trajectory, is_strike, is_selected, progress)

        # === 3. 공 착탄 위치 마커 (맨 앞 레이어) ===
        for x, z, is_strike, number, speed, trajectory in self.records:
            is_selected = (number == self.selected_pitch)
            # 궤적이 그려지는 중이면 마커를 나중에 그림 (애니메이션 끝날 때)
            is_latest = (number == latest_idx)
            
            if is_latest and self.is_animating:
                if self.animation_progress > 0.9: # 애니메이션 거의 끝날 때 등장
                    self._draw_marker(painter, x, z, is_strike, number, is_selected)
            else:
                self._draw_marker(painter, x, z, is_strike, number, is_selected)
                
        # 타이틀
        painter.setPen(QColor(*record_config.COLOR_TEXT))
        painter.setFont(QFont(window_config.FONT_FAMILY, 12, QFont.Bold))
        painter.drawText(10, 20, "⚾ 투구 궤적 (Pitcher View)")

    def _draw_trajectory_3d(self, painter, trajectory, is_strike, is_selected, progress):
        """3D 입체 궤적 그리기"""
        if len(trajectory) < 2: return
        
        # 색상 설정
        if is_selected:
            color = QColor(255, 220, 100) # 금색
        elif is_strike:
            color = QColor(0, 255, 100) # 녹색
        else:
            color = QColor(255, 80, 80) # 빨강
            
        # 궤적 점 변환
        points_2d = []
        scales = []
        
        # 표시할 포인트 개수 제한 (애니메이션)
        limit = int(len(trajectory) * progress)
        if limit < 2: return
        
        draw_traj = trajectory[:limit]
        
        for tx, ty, tz in draw_traj:
            wx, wy, s = self._perspective_transform(tx, ty, tz)
            points_2d.append(QPointF(wx, wy))
            scales.append(s)
            
        # 선 그리기 (Segment별로 두께와 투명도 달리함)
        for i in range(len(points_2d) - 1):
            p1 = points_2d[i]
            p2 = points_2d[i+1]
            s = scales[i] # 투수 쪽일수록 큼 (> 1.0)
            
            # 투수 쪽(시작점)일수록 투명하게, 도착점일수록 진하게
            # s가 1.0(존)에 가까우면 alpha 255, s가 클수록 alpha 감소
            alpha = int(255 / s)
            alpha = max(50, min(255, alpha))
            
            # 두께: 가까운 쪽(투수)이 굵어 보임 (원근법)
            width = 3 * s 
            
            current_color = QColor(color)
            current_color.setAlpha(alpha)
            
            pen = QPen(current_color, width)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(p1, p2)
            
    def _draw_marker(self, painter, x, z, is_strike, number, is_selected):
        """착탄 마커 그리기 (2D 평면 기준)"""
        cx, cy = self._world_to_widget_flat(x, z)
        
        radius = 8
        if is_selected:
            radius = 12
            painter.setPen(QPen(Qt.yellow, 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), radius+2, radius+2)
            
        color = QColor(*record_config.COLOR_STRIKE) if is_strike else QColor(*record_config.COLOR_BALL)
        painter.setPen(Qt.white)
        painter.setBrush(color)
        painter.drawEllipse(QPointF(cx, cy), radius, radius)
        
        # 번호
        painter.drawText(QRectF(cx-radius, cy-radius, 2*radius, 2*radius), 
                         Qt.AlignCenter, str(number))

    # ... [mousePressEvent, _draw_target_zone 등 기존 메서드 유지] ...
    def mousePressEvent(self, event):
        """마우스 클릭: 공 선택"""
        if event.button() == Qt.LeftButton:
            # 평면 좌표계 기준으로 거리 계산
            min_dist = 20
            clicked_num = None
            
            for x, z, _, num, _, _ in self.records:
                wx, wy = self._world_to_widget_flat(x, z)
                dist = ((event.x() - wx)**2 + (event.y() - wy)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    clicked_num = num
            
            if clicked_num:
                self.selected_pitch = clicked_num if self.selected_pitch != clicked_num else None
                self.pitchSelected.emit(clicked_num if self.selected_pitch else 0)
                self.update()

    def _draw_target_zone(self, painter, xl, yt, w, h):
        """게임모드 타겟 그리기 (기존 로직 유지)"""
        if not self.target_zone: return
        
        r = (self.target_zone - 1) // 3
        c = (self.target_zone - 1) % 3
        
        cw, ch = w/3, h/3
        tx = xl + c * cw
        ty = yt + r * ch
        
        painter.fillRect(QRectF(tx, ty, cw, ch), QColor(255, 165, 0, 100))
        painter.setPen(QPen(QColor(255, 165, 0), 2))
        painter.drawRect(QRectF(tx, ty, cw, ch))

class Scoreboard(QFrame):
    """
    스코어보드 위젯 (간소화 버전)
    B-S-O 카운트만 표시 (이닝, 점수 제거)
    """
    
    countChanged = pyqtSignal(int, int, int)  # balls, strikes, outs
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(2)
        
        self.balls = 0
        self.strikes = 0
        self.outs = 0
        
        self._init_ui()
        
    def _init_ui(self):
        """UI 초기화"""
        layout = QHBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 10, 15, 10)
        
        # 카운트 섹션
        count_frame = QFrame()
        count_layout = QGridLayout(count_frame)
        count_layout.setSpacing(5)
        
        # B-S-O 라벨
        font_label = QFont(window_config.FONT_FAMILY, 14, QFont.Bold)
        font_count = QFont(window_config.FONT_FAMILY, 20, QFont.Bold)
        
        # Ball
        lbl_b = QLabel("B")
        lbl_b.setFont(font_label)
        lbl_b.setStyleSheet("color: #00AA00;")
        count_layout.addWidget(lbl_b, 0, 0)
        
        self.ball_indicators = []
        for i in range(4):
            indicator = QLabel("●" if i < self.balls else "○")
            indicator.setFont(font_count)
            indicator.setStyleSheet("color: #00AA00;")
            count_layout.addWidget(indicator, 0, i + 1)
            self.ball_indicators.append(indicator)
            
        # Strike
        lbl_s = QLabel("S")
        lbl_s.setFont(font_label)
        lbl_s.setStyleSheet("color: #DDDD00;")
        count_layout.addWidget(lbl_s, 1, 0)
        
        self.strike_indicators = []
        for i in range(3):
            indicator = QLabel("●" if i < self.strikes else "○")
            indicator.setFont(font_count)
            indicator.setStyleSheet("color: #DDDD00;")
            count_layout.addWidget(indicator, 1, i + 1)
            self.strike_indicators.append(indicator)
            
        # Out
        lbl_o = QLabel("O")
        lbl_o.setFont(font_label)
        lbl_o.setStyleSheet("color: #DD0000;")
        count_layout.addWidget(lbl_o, 2, 0)
        
        self.out_indicators = []
        for i in range(3):
            indicator = QLabel("●" if i < self.outs else "○")
            indicator.setFont(font_count)
            indicator.setStyleSheet("color: #DD0000;")
            count_layout.addWidget(indicator, 2, i + 1)
            self.out_indicators.append(indicator)
        
        layout.addWidget(count_frame)
        layout.addStretch()
        
    def add_strike(self):
        """스트라이크 추가"""
        self.strikes += 1
        if self.strikes >= 3:
            self.add_out()
            self.strikes = 0
            self.balls = 0
        self._update_display()
        
    def add_ball(self):
        """볼 추가"""
        self.balls += 1
        if self.balls >= 4:
            # 볼넷
            self.balls = 0
            self.strikes = 0
        self._update_display()
        
    def add_out(self):
        """아웃 추가"""
        self.outs += 1
        if self.outs >= 3:
            self.outs = 0
        self._update_display()
        
    def reset_count(self):
        """카운트 리셋"""
        self.balls = 0
        self.strikes = 0
        self._update_display()
        
    def reset_all(self):
        """전체 리셋"""
        self.balls = 0
        self.strikes = 0
        self.outs = 0
        self._update_display()
        
    def _update_display(self):
        """디스플레이 업데이트"""
        # Ball indicators
        for i, indicator in enumerate(self.ball_indicators):
            indicator.setText("●" if i < self.balls else "○")
            
        # Strike indicators
        for i, indicator in enumerate(self.strike_indicators):
            indicator.setText("●" if i < self.strikes else "○")
            
        # Out indicators
        for i, indicator in enumerate(self.out_indicators):
            indicator.setText("●" if i < self.outs else "○")
        
        self.countChanged.emit(self.balls, self.strikes, self.outs)


class GameModeWidget(QFrame):
    """
    게임 모드 위젯
    9구역 타겟 연습 모드
    """
    
    targetHit = pyqtSignal(int, bool)  # zone, is_hit
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(2)
        
        self.target_zone = None
        self.score = 0
        self.attempts = 0
        self.max_attempts = 10
        self.hits = 0
        
        self._init_ui()
        
    def _init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)
        
        # 타이틀
        title = QLabel("🎯 타겟 모드")
        title.setFont(QFont(window_config.FONT_FAMILY, 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 현재 타겟 표시
        self.target_label = QLabel("목표: -")
        self.target_label.setFont(QFont(window_config.FONT_FAMILY, 14))
        self.target_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.target_label)
        
        # 점수
        score_layout = QHBoxLayout()
        
        self.score_label = QLabel(f"점수: {self.score}")
        self.score_label.setFont(QFont(window_config.FONT_FAMILY, 14, QFont.Bold))
        score_layout.addWidget(self.score_label)
        
        self.attempts_label = QLabel(f"시도: {self.attempts}/{self.max_attempts}")
        self.attempts_label.setFont(QFont(window_config.FONT_FAMILY, 12))
        score_layout.addWidget(self.attempts_label)
        
        layout.addLayout(score_layout)
        
        # 명중률
        self.accuracy_label = QLabel("명중률: 0%")
        self.accuracy_label.setFont(QFont(window_config.FONT_FAMILY, 12))
        self.accuracy_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.accuracy_label)
        
    def set_target(self, zone):
        """목표 구역 설정"""
        self.target_zone = zone
        zone_name = game_config.ZONE_NAMES.get(zone, str(zone))
        self.target_label.setText(f"목표: {zone} ({zone_name})")
        self.target_label.setStyleSheet("color: #FF6600; font-weight: bold;")
        
    def set_random_target(self):
        """랜덤 목표 설정"""
        import random
        zone = random.randint(1, 9)
        self.set_target(zone)
        return zone
        
    def check_hit(self, actual_zone):
        """명중 체크"""
        self.attempts += 1
        is_hit = (actual_zone == self.target_zone)
        
        if is_hit:
            self.hits += 1
            zone_score = game_config.ZONE_SCORES.get(self.target_zone, 5)
            self.score += zone_score
            
        self._update_display()
        self.targetHit.emit(self.target_zone, is_hit)
        
        return is_hit
        
    def reset(self):
        """리셋"""
        self.target_zone = None
        self.score = 0
        self.attempts = 0
        self.hits = 0
        self.target_label.setText("목표: -")
        self.target_label.setStyleSheet("")
        self._update_display()
        
    def _update_display(self):
        """디스플레이 업데이트"""
        self.score_label.setText(f"점수: {self.score}")
        self.attempts_label.setText(f"시도: {self.attempts}/{self.max_attempts}")
        
        if self.attempts > 0:
            accuracy = (self.hits / self.attempts) * 100
            self.accuracy_label.setText(f"명중률: {accuracy:.1f}%")
        else:
            self.accuracy_label.setText("명중률: 0%")


class StatsWidget(QFrame):
    """
    통계 위젯
    평균 구속, 스트라이크 비율 등 표시
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        
        self.total_pitches = 0
        self.strikes = 0
        self.balls = 0
        self.speeds = []
        
        self._init_ui()
        
    def _init_ui(self):
        """UI 초기화"""
        layout = QGridLayout(self)
        
        font_label = QFont(window_config.FONT_FAMILY, 11)
        font_value = QFont(window_config.FONT_FAMILY, 13, QFont.Bold)
        
        # 총 투구수
        layout.addWidget(QLabel("총 투구:"), 0, 0)
        self.total_label = QLabel("0")
        self.total_label.setFont(font_value)
        layout.addWidget(self.total_label, 0, 1)
        
        # 스트라이크
        layout.addWidget(QLabel("스트라이크:"), 1, 0)
        self.strike_label = QLabel("0")
        self.strike_label.setFont(font_value)
        self.strike_label.setStyleSheet("color: #00AA00;")
        layout.addWidget(self.strike_label, 1, 1)
        
        # 볼
        layout.addWidget(QLabel("볼:"), 2, 0)
        self.ball_label = QLabel("0")
        self.ball_label.setFont(font_value)
        self.ball_label.setStyleSheet("color: #DD0000;")
        layout.addWidget(self.ball_label, 2, 1)
        
        # 스트라이크 비율
        layout.addWidget(QLabel("S%:"), 0, 2)
        self.strike_pct_label = QLabel("0%")
        self.strike_pct_label.setFont(font_value)
        layout.addWidget(self.strike_pct_label, 0, 3)
        
        # 평균 구속
        layout.addWidget(QLabel("평균 구속:"), 1, 2)
        self.avg_speed_label = QLabel("- km/h")
        self.avg_speed_label.setFont(font_value)
        self.avg_speed_label.setStyleSheet("color: #FF6600;")
        layout.addWidget(self.avg_speed_label, 1, 3)
        
        # 최고 구속
        layout.addWidget(QLabel("최고 구속:"), 2, 2)
        self.max_speed_label = QLabel("- km/h")
        self.max_speed_label.setFont(font_value)
        self.max_speed_label.setStyleSheet("color: #FF0000;")
        layout.addWidget(self.max_speed_label, 2, 3)
        
    def add_pitch(self, is_strike, speed=None):
        """투구 기록 추가"""
        self.total_pitches += 1
        
        if is_strike:
            self.strikes += 1
        else:
            self.balls += 1
            
        if speed is not None and speed > 0:
            self.speeds.append(speed)
            
        self._update_display()
        
    def reset(self):
        """리셋"""
        self.total_pitches = 0
        self.strikes = 0
        self.balls = 0
        self.speeds = []
        self._update_display()
        
    def _update_display(self):
        """디스플레이 업데이트"""
        self.total_label.setText(str(self.total_pitches))
        self.strike_label.setText(str(self.strikes))
        self.ball_label.setText(str(self.balls))
        
        if self.total_pitches > 0:
            strike_pct = (self.strikes / self.total_pitches) * 100
            self.strike_pct_label.setText(f"{strike_pct:.1f}%")
        else:
            self.strike_pct_label.setText("0%")
            
        if self.speeds:
            avg_speed = sum(self.speeds) / len(self.speeds)
            max_speed = max(self.speeds)
            self.avg_speed_label.setText(f"{avg_speed:.1f} km/h")
            self.max_speed_label.setText(f"{max_speed:.1f} km/h")
        else:
            self.avg_speed_label.setText("- km/h")
            self.max_speed_label.setText("- km/h")


class PitchListWidget(QFrame):
    """
    투구 리스트 위젯
    던진 공들의 목록을 표시하고 선택 가능
    """
    
    pitchSelected = pyqtSignal(int)  # 선택된 공 번호
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(1)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 확장 가능
        
        self.pitches = []  # [(number, is_strike, speed), ...]
        self.selected_number = None  # 현재 선택된 공 번호
        
        self._init_ui()
        
    def _init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 타이틀
        title = QLabel("📜 투구 목록")
        title.setFont(QFont(window_config.FONT_FAMILY, 11, QFont.Bold))
        title.setStyleSheet("color: #ffffff;")
        layout.addWidget(title)
        
        # 리스트 위젯
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #2a2a2a;
                border: 1px solid #444;
                border-radius: 3px;
                color: #ffffff;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #3a3a3a;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
            }
            QListWidget::item:hover {
                background-color: #3a3a3a;
            }
        """)
        # 높이 제한 제거 - 확장 가능하게
        self.list_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.list_widget)
        
    def add_pitch(self, number, is_strike, speed):
        """투구 추가"""
        self.pitches.append((number, is_strike, speed))
        
        # 결과 표시
        result = "S" if is_strike else "B"
        result_color = "#00CC66" if is_strike else "#FF6666"
        
        # 구속 표시
        speed_text = f"{speed:.1f}km/h" if speed and speed > 0 else "-"
        
        # 리스트 아이템 생성
        item_text = f"#{number}  [{result}]  {speed_text}"
        item = QListWidgetItem(item_text)
        
        # 색상 설정
        if is_strike:
            item.setForeground(QColor(0, 200, 100))
        else:
            item.setForeground(QColor(255, 100, 100))
            
        self.list_widget.addItem(item)
        
        # 스크롤을 최신 항목으로
        self.list_widget.scrollToBottom()
        
    def clear_pitches(self):
        """리스트 초기화"""
        self.pitches = []
        self.selected_number = None
        self.list_widget.clear()
        
    def select_pitch(self, number):
        """특정 투구 선택 (0이면 선택 해제)"""
        if number == 0:
            # 선택 해제
            self.list_widget.clearSelection()
            self.selected_number = None
        else:
            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                if i + 1 == number:
                    self.list_widget.setCurrentItem(item)
                    self.selected_number = number
                    break
                
    def _on_item_clicked(self, item):
        """아이템 클릭 이벤트 - 토글 기능"""
        row = self.list_widget.row(item)
        number = row + 1
        
        # 같은 아이템 다시 클릭하면 선택 해제
        if self.selected_number == number:
            self.list_widget.clearSelection()
            self.selected_number = None
            self.pitchSelected.emit(0)  # 0은 선택 해제 의미
        else:
            self.selected_number = number
            self.pitchSelected.emit(number)
        
    def get_pitch_info(self, number):
        """특정 투구 정보 반환"""
        if 0 < number <= len(self.pitches):
            return self.pitches[number - 1]
        return None
