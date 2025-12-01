# -*- coding: utf-8 -*-
"""
GUI 위젯 모듈
2D 기록지, 스코어보드, 9분할 뷰 등의 커스텀 위젯
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, 
    QPainterPath, QLinearGradient
)
import math

from gui_config import (
    record_config, scoreboard_config, 
    game_config, window_config
)


class RecordSheet2D(QWidget):
    """
    2D 기록지 위젯
    스트라이크 존을 위에서 본 시점으로 표시
    공의 위치를 마커로 표시 (스트라이크: 녹색, 볼: 빨강)
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.records = []  # [(x, z, is_strike, number, speed), ...]
        self.setMinimumSize(record_config.WIDTH, record_config.HEIGHT)
        self.setMaximumSize(record_config.WIDTH + 100, record_config.HEIGHT + 100)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        # 스트라이크 존 범위 (정규화용)
        self.zone_x_min = -0.25  # 미터
        self.zone_x_max = 0.25
        self.zone_z_min = 0.15
        self.zone_z_max = 0.75
        
        # 실제 스트라이크 존 경계
        self.strike_x_min = -0.15
        self.strike_x_max = 0.15
        self.strike_z_min = 0.25
        self.strike_z_max = 0.65
        
    def add_record(self, x, z, is_strike, speed=None):
        """기록 추가"""
        number = len(self.records) + 1
        self.records.append((x, z, is_strike, number, speed))
        
        # 최대 개수 초과시 오래된 것 제거
        if len(self.records) > record_config.MAX_DISPLAY_COUNT:
            self.records.pop(0)
            # 번호 재정렬
            for i, (x, z, is_s, _, spd) in enumerate(self.records):
                self.records[i] = (x, z, is_s, i + 1, spd)
        
        self.update()
        
    def clear_records(self):
        """기록 초기화"""
        self.records = []
        self.update()
        
    def _world_to_widget(self, x, z):
        """월드 좌표를 위젯 좌표로 변환"""
        margin = record_config.MARGIN
        w = self.width() - 2 * margin
        h = self.height() - 2 * margin
        
        # 정규화 (0~1)
        nx = (x - self.zone_x_min) / (self.zone_x_max - self.zone_x_min)
        nz = (z - self.zone_z_min) / (self.zone_z_max - self.zone_z_min)
        
        # 위젯 좌표 (Z는 위아래 반전)
        wx = margin + nx * w
        wy = margin + (1 - nz) * h
        
        return wx, wy
        
    def paintEvent(self, event):
        """그리기 이벤트"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        margin = record_config.MARGIN
        w = self.width() - 2 * margin
        h = self.height() - 2 * margin
        
        # 배경
        painter.fillRect(self.rect(), QColor(*record_config.COLOR_BACKGROUND))
        
        # 스트라이크 존 경계 좌표
        zone_left, zone_top = self._world_to_widget(self.strike_x_min, self.strike_z_max)
        zone_right, zone_bottom = self._world_to_widget(self.strike_x_max, self.strike_z_min)
        zone_w = zone_right - zone_left
        zone_h = zone_bottom - zone_top
        
        # 스트라이크 존 배경 (연한 녹색)
        zone_bg = QColor(200, 255, 200, 100)
        painter.fillRect(int(zone_left), int(zone_top), int(zone_w), int(zone_h), zone_bg)
        
        # 9분할 그리드
        grid_pen = QPen(QColor(*record_config.COLOR_GRID), 1, Qt.DashLine)
        painter.setPen(grid_pen)
        
        # 수직선 (3등분)
        for i in range(1, 3):
            x = zone_left + zone_w * i / 3
            painter.drawLine(int(x), int(zone_top), int(x), int(zone_bottom))
            
        # 수평선 (3등분)
        for i in range(1, 3):
            y = zone_top + zone_h * i / 3
            painter.drawLine(int(zone_left), int(y), int(zone_right), int(y))
        
        # 스트라이크 존 테두리
        zone_pen = QPen(QColor(*record_config.COLOR_ZONE_BORDER), 2)
        painter.setPen(zone_pen)
        painter.drawRect(int(zone_left), int(zone_top), int(zone_w), int(zone_h))
        
        # 구역 번호 표시
        font = QFont(window_config.FONT_FAMILY, 8)
        painter.setFont(font)
        painter.setPen(QColor(150, 150, 150))
        
        zone_positions = [
            (1, zone_left + zone_w/6, zone_top + zone_h/6),
            (2, zone_left + zone_w/2, zone_top + zone_h/6),
            (3, zone_left + 5*zone_w/6, zone_top + zone_h/6),
            (4, zone_left + zone_w/6, zone_top + zone_h/2),
            (5, zone_left + zone_w/2, zone_top + zone_h/2),
            (6, zone_left + 5*zone_w/6, zone_top + zone_h/2),
            (7, zone_left + zone_w/6, zone_top + 5*zone_h/6),
            (8, zone_left + zone_w/2, zone_top + 5*zone_h/6),
            (9, zone_left + 5*zone_w/6, zone_top + 5*zone_h/6),
        ]
        
        for num, x, y in zone_positions:
            painter.drawText(int(x - 5), int(y + 5), str(num))
        
        # 공 마커 그리기
        font = QFont(window_config.FONT_FAMILY, record_config.MARKER_FONT_SIZE)
        painter.setFont(font)
        
        for x, z, is_strike, number, speed in self.records:
            wx, wy = self._world_to_widget(x, z)
            
            # 마커 색상
            if is_strike:
                color = QColor(*record_config.COLOR_STRIKE)
            else:
                color = QColor(*record_config.COLOR_BALL)
            
            # 마커 그리기
            painter.setPen(QPen(color.darker(120), 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(
                int(wx - record_config.MARKER_RADIUS),
                int(wy - record_config.MARKER_RADIUS),
                record_config.MARKER_RADIUS * 2,
                record_config.MARKER_RADIUS * 2
            )
            
            # 번호 표시
            painter.setPen(Qt.white)
            text = str(number)
            painter.drawText(
                int(wx - 4), int(wy + 4),
                text
            )
        
        # 타이틀
        title_font = QFont(window_config.FONT_FAMILY, 12, QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(Qt.black)
        painter.drawText(10, 15, "투구 기록")


class Scoreboard(QFrame):
    """
    스코어보드 위젯
    B-S-O 카운트, 이닝, 점수 표시
    """
    
    countChanged = pyqtSignal(int, int, int)  # balls, strikes, outs
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(2)
        
        self.balls = 0
        self.strikes = 0
        self.outs = 0
        self.inning = 1
        self.is_top = True
        self.home_score = 0
        self.away_score = 0
        
        self._init_ui()
        
    def _init_ui(self):
        """UI 초기화"""
        layout = QHBoxLayout(self)
        layout.setSpacing(20)
        
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
        
        # 구분선
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # 이닝 섹션
        inning_frame = QFrame()
        inning_layout = QVBoxLayout(inning_frame)
        
        self.inning_label = QLabel(f"{'▲' if self.is_top else '▼'} {self.inning}회")
        self.inning_label.setFont(QFont(window_config.FONT_FAMILY, 18, QFont.Bold))
        self.inning_label.setAlignment(Qt.AlignCenter)
        inning_layout.addWidget(self.inning_label)
        
        layout.addWidget(inning_frame)
        
        # 구분선
        line2 = QFrame()
        line2.setFrameShape(QFrame.VLine)
        line2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line2)
        
        # 점수 섹션
        score_frame = QFrame()
        score_layout = QGridLayout(score_frame)
        
        font_team = QFont(window_config.FONT_FAMILY, 12)
        font_score = QFont(window_config.FONT_FAMILY, 24, QFont.Bold)
        
        lbl_away = QLabel("원정")
        lbl_away.setFont(font_team)
        score_layout.addWidget(lbl_away, 0, 0)
        
        self.away_score_label = QLabel(str(self.away_score))
        self.away_score_label.setFont(font_score)
        self.away_score_label.setAlignment(Qt.AlignCenter)
        score_layout.addWidget(self.away_score_label, 0, 1)
        
        lbl_vs = QLabel("-")
        lbl_vs.setFont(font_score)
        lbl_vs.setAlignment(Qt.AlignCenter)
        score_layout.addWidget(lbl_vs, 0, 2)
        
        self.home_score_label = QLabel(str(self.home_score))
        self.home_score_label.setFont(font_score)
        self.home_score_label.setAlignment(Qt.AlignCenter)
        score_layout.addWidget(self.home_score_label, 0, 3)
        
        lbl_home = QLabel("홈")
        lbl_home.setFont(font_team)
        score_layout.addWidget(lbl_home, 0, 4)
        
        layout.addWidget(score_frame)
        
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
            self._next_half_inning()
        self._update_display()
        
    def _next_half_inning(self):
        """다음 하프 이닝"""
        if self.is_top:
            self.is_top = False
        else:
            self.is_top = True
            self.inning += 1
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
        self.inning = 1
        self.is_top = True
        self.home_score = 0
        self.away_score = 0
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
            
        # Inning
        self.inning_label.setText(f"{'▲' if self.is_top else '▼'} {self.inning}회")
        
        # Scores
        self.away_score_label.setText(str(self.away_score))
        self.home_score_label.setText(str(self.home_score))
        
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
