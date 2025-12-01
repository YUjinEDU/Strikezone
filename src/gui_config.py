# -*- coding: utf-8 -*-
"""
GUI 설정 파일
시각화 옵션, 게임 모드 설정 등을 관리
"""

# ==================== 시각화 토글 설정 ====================
class VisualizationConfig:
    """시각화 옵션 설정 클래스"""
    
    # 스트라이크 존 시각화
    SHOW_STRIKE_ZONE = True          # 스트라이크 존 테두리
    SHOW_PLANE1 = True               # 평면1 (홈플레이트 앞)
    SHOW_PLANE2 = True               # 평면2 (홈플레이트 뒤)
    SHOW_9_GRID = True               # 9분할 그리드
    SHOW_BALL_TRAJECTORY = True      # 공 궤적
    SHOW_BALL_MARKER = True          # 공 마커 (현재 위치)
    SHOW_SPEED = True                # 구속 표시
    SHOW_ARUCO_MARKERS = True        # ArUco 마커 표시
    SHOW_COORDINATE_AXES = True      # 좌표축 표시
    
    # 색상 설정 (BGR)
    COLOR_STRIKE_ZONE = (0, 255, 0)      # 녹색
    COLOR_PLANE1 = (255, 255, 0)         # 시안
    COLOR_PLANE2 = (255, 0, 255)         # 마젠타
    COLOR_GRID = (128, 128, 128)         # 회색
    COLOR_TRAJECTORY = (0, 165, 255)     # 주황
    COLOR_BALL = (0, 0, 255)             # 빨강
    COLOR_STRIKE_TEXT = (0, 255, 0)      # 녹색
    COLOR_BALL_TEXT = (0, 0, 255)        # 빨강
    COLOR_SPEED = (0, 255, 255)          # 노랑
    
    # 투명도 (0.0 ~ 1.0)
    ZONE_ALPHA = 0.3
    PLANE_ALPHA = 0.2
    GRID_ALPHA = 0.5


# ==================== 게임 모드 설정 ====================
class GameModeConfig:
    """게임 모드 설정 클래스"""
    
    ENABLED = False                  # 게임 모드 활성화
    TARGET_ZONE = None               # 목표 구역 (1~9, None이면 랜덤)
    SCORE = 0                        # 현재 점수
    ATTEMPTS = 0                     # 시도 횟수
    MAX_ATTEMPTS = 10                # 최대 시도 횟수
    
    # 9분할 구역 점수
    ZONE_SCORES = {
        1: 10, 2: 10, 3: 10,         # 상단 (코너)
        4: 5,  5: 3,  6: 5,          # 중단 (가운데가 쉬움)
        7: 10, 8: 10, 9: 10          # 하단 (코너)
    }
    
    # 구역 이름
    ZONE_NAMES = {
        1: "좌상단", 2: "중상단", 3: "우상단",
        4: "좌중단", 5: "중앙",   6: "우중단",
        7: "좌하단", 8: "중하단", 9: "우하단"
    }


# ==================== 기록지 설정 ====================
class RecordSheetConfig:
    """2D 기록지 설정 클래스"""
    
    # 기록지 크기
    WIDTH = 300
    HEIGHT = 300
    MARGIN = 20
    
    # 마커 크기
    MARKER_RADIUS = 8
    MARKER_FONT_SIZE = 10
    
    # 최대 표시 개수
    MAX_DISPLAY_COUNT = 50
    
    # 색상 (RGB for Qt)
    COLOR_STRIKE = (0, 200, 0)       # 녹색
    COLOR_BALL = (200, 0, 0)         # 빨강
    COLOR_ZONE_BORDER = (0, 0, 0)    # 검정
    COLOR_GRID = (200, 200, 200)     # 밝은 회색
    COLOR_BACKGROUND = (255, 255, 255)  # 흰색


# ==================== 스코어보드 설정 ====================
class ScoreboardConfig:
    """스코어보드 설정 클래스"""
    
    # 기본 카운트
    BALLS = 0
    STRIKES = 0
    OUTS = 0
    
    # 이닝 정보
    INNING = 1
    IS_TOP = True                    # 초/말
    
    # 점수
    HOME_SCORE = 0
    AWAY_SCORE = 0
    
    # 카운트 리셋 조건
    STRIKES_FOR_OUT = 3
    BALLS_FOR_WALK = 4


# ==================== 윈도우 설정 ====================
class WindowConfig:
    """윈도우 설정 클래스"""
    
    # 메인 윈도우 크기
    WINDOW_WIDTH = 1400
    WINDOW_HEIGHT = 800
    
    # 비디오 디스플레이 크기
    VIDEO_WIDTH = 800
    VIDEO_HEIGHT = 600
    
    # 패널 크기
    RIGHT_PANEL_WIDTH = 350
    BOTTOM_PANEL_HEIGHT = 150
    
    # 폰트
    FONT_FAMILY = "맑은 고딕"
    FONT_SIZE_NORMAL = 12
    FONT_SIZE_LARGE = 18
    FONT_SIZE_TITLE = 24


# ==================== 전역 설정 인스턴스 ====================
vis_config = VisualizationConfig()
game_config = GameModeConfig()
record_config = RecordSheetConfig()
scoreboard_config = ScoreboardConfig()
window_config = WindowConfig()
