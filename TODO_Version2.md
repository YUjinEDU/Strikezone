# TODO: Robust color-based detection and tracking upgrades

Author: YUjinEDU  
Context: Strikezone (ARUCO/marker-based strike judgement with color-based ball detection)  
Goal: Make color/optics and detection robust across batting cages and gym interiors without ML. Keep current pipeline; add these upgrades later.

---

## 0) Scope and success criteria

- Scope: Color/optics, detection logic, marker stack, and camera timings. Do NOT change judgement logic (plane1→plane2, oriented distance) now.
- Success criteria
  - Detection precision/recall ≥ 95% for “ball-present frames” in both environments.
  - Speed error ≤ ±2.5 km/h vs. plane-cross timing ground-truth over ≥ 50 pitches.
  - Strike/ball decision matches operator labels ≥ 95% in test clips.
  - CPU stays under target (TBD) at 60 fps with undistortion enabled.

---

## 1) 색/광학 (나중에 적용)

Tasks
- [ ] 공 표시 개선: 형광 마젠타(무광) 테이프 또는 슬리브 적용
  - [ ] 반짝임(유광) 방지: 매트/무광 표면 선택
  - [ ] 적용 면적 가이드: 공의 한 면에 3~5cm 폭 띠 1~2개
- [ ] 노출/화이트밸런스/ISO “고정”
  - [ ] 카메라 드라이버/SDK에서 Auto Exposure/WB OFF
  - [ ] 초기 N(=30)프레임에서 평균 휘도/색 온도를 읽어 수동값을 결정한 뒤 고정
- [ ] 조명: 링라이트 또는 균일한 라이트 바 설치
  - [ ] 색온도 5000–5600K, 플리커 없는 상시광
  - [ ] 홈런/사각 음영을 줄이도록 카메라 근처 정면 배치
- [ ] (선택) 편광(Pol) + 크로스폴라라이저
  - [ ] 조명에 선형 편광 필름, 렌즈에 교차 방향 편광 필터
  - [ ] 반사 하이라이트를 억제하여 색상 마스크 안정화

Acceptance
- [ ] 카메라 미리보기에서 공의 형광 색 영역이 배경 대비 명확(HSV 히스토그램 분리도 확인)
- [ ] 시간대별/좌표별 조명 변화에도 HSV 마스크 유지율 ≥ 95%

---

## 2) 탐지 로직 (나중에 적용)

Additions to current BallDetector pipeline. 파일 경로 제안: `src/color_detector.py` (새 모듈), 기존 호출부는 `tracker_v1.py` → `BallDetector` 대체 가능.

Tasks
- [ ] 병렬 마스크 결합: HSV + 색상 불변량(비율/rg)
  - [ ] HSV 기본 마스크: H∈[hL,hU], S>SL, V>VL (초기 값: 형광 마젠타 기준)
  - [ ] rg-chromaticity: r=R/(R+G+B), g=G/(R+G+B), 임계 r>r0, g<g1 등
  - [ ] 색 비율: R/G > t1, R/B > t2 (환경 보정)
  - [ ] final_mask = (HSV_mask AND (rg_mask OR ratio_mask))
- [ ] 세션별 색 튜닝: 히스토그램 백프로젝션
  - [ ] 시작 시 공 ROI(수동 박스 or 첫 검출 결과)에서 HSV 히스토그램 생성
  - [ ] cv2.calcBackProject로 후보 강화 → 모폴로지 open/close
  - [ ] 매 세션 1회, 환경 바뀌면 재튜닝
- [ ] 원형도/면적/반지름 변화율 게이팅
  - [ ] circularity = 4πA/P² > 0.55(초기), 면적 30–8000 px²(환경별 튜닝)
  - [ ] 반지름 변화율 |Δr/Δt| < r_dot_max (칼만 예측과 Mahalanobis 게이트로 outlier 제외)
- [ ] 칼만 예측과 계측 게이팅 결합
  - [ ] Mahalanobis distance < χ²(3, 0.95)=7.81 (이미 사용 중 유지)
  - [ ] 예측 중심 근방에서만 contour 후보 채택
- [ ] 파라미터 자동화 유틸
  - [ ] `auto_tune_hsv_sv_lower()` → 초기 N프레임 평균/분산으로 SL/VL 낮춤-리밋
  - [ ] `auto_tune_ratio_thresholds()` → 작은 grid search로 F1 최대화

Code plan
- [ ] `src/color_detector.py`
  - [ ] `class ColorModel: hsv, rg, ratio thresholds + fit(backproj)/apply(frame)`
  - [ ] `def build_session_model(frames[:N], seed_roi): -> ColorModel`
  - [ ] `def apply_masks(frame, model) -> mask`
- [ ] `src/tracker_v1.py` or new `src/ball_detector_v2.py`
  - [ ] BallDetectorV2.detect(): 병렬 마스크 결합, 규칙 필터링, 반환(center, radius, mask, debug)
  - [ ] 플래그로 기존/신규 스위칭

Acceptance
- [ ] 두 환경(배팅장/체육관) 테스트 클립에서 FP/ FN 각각 ≤ 5%
- [ ] 강한 그림자/반짝임에서 연속 프레임 유실 ≤ 1프레임(60fps 기준)

참고 초기 파라미터(형광 마젠타 예)
- HSV: H∈[140,179], S>120, V>120
- rg: r>0.45, g<0.35
- 비율: R/G>1.6, R/B>1.8  
(현장 캘리브레이션으로 재조정)

---

## 3) 마커 (나중에 적용)

AprilTag + 보드(2~4개) + KLT 추적 + solvePnP + Health Check

Tasks
- [ ] AprilTag 라이브러리 적용 (OpenCV or apriltag-py)
  - [ ] 태그 패밀리: tag36h11 권장
  - [ ] 보드 레이아웃: 2×2 또는 1×3 (마커 간격 및 크기 문서화)
- [ ] 초기 탐지 후 코너 추적(KLT)
  - [ ] `cv2.calcOpticalFlowPyrLK`로 각 코너 추적
  - [ ] 누적 드리프트 방지: N프레임마다 태그 재검출
- [ ] solvePnP로 매 프레임 extrinsic 업데이트
  - [ ] 입력: 3D 보드 포인트 ↔ 2D 추적 코너
  - [ ] R,t 산출 → projectPoints 검증
- [ ] Health Check(재투영 오차)
  - [ ] reproj_err_mean > 1.5 px (초기값)일 때 강제 재검출
  - [ ] 가림/유실 시 fallback
- [ ] (고정 설치 시) 빈도 낮추기
  - [ ] 완전 고정이면 1초 1회 재검출 + 나머지는 KLT+solvePnP
  - [ ] extrinsic persist: 시작 시 저장/로드 옵션

Code plan
- [ ] `src/marker_tracker.py`
  - [ ] `class BoardPoseTracker: detect_init(), track_klt(), solve_pnp(), health_check()`
  - [ ] `get_pose(frame) -> (rvec, tvec, quality)`
- [ ] `src/main.py`
  - [ ] 기존 ArUco 사용부를 인터페이스로 추상화하여 교체 가능하게 설계

Acceptance
- [ ] 급격한 움직임/부분 가림에도 pose quality ≥ threshold, 재투영 오차 평균 ≤ 1.5px 유지
- [ ] 타임라인에서 pose drop ≤ 1% 프레임

---

## 4) 카메라 (나중에 적용)

Tasks
- [ ] 셔터/게인/ISO 고정: 모션블러 최소화(셔터 1/200s~1/500s 권장)
- [ ] 60fps 유지(프레임 드랍 감시)
  - [ ] 매 프레임 `time.perf_counter()`로 타임스탬프 로깅
  - [ ] 드랍/지터 알림(로그 WARN)
- [ ] 왜곡 보정(이미 remap 도입): 해상도/캘리브레이션 일치 확인
- [ ] 타임라인 정확화
  - [ ] 보간 기반 속도 측정 루틴 유지: `t_cross = t_prev + α Δt` (α=prev_d/(prev_d - curr_d))

Acceptance
- [ ] 프레임 간 Δt 표준편차 ≤ 1ms (카드/드라이버 환경에 따라 조정)
- [ ] 속도 계산 반복성(동일 투구 반복) 표준편차 ≤ 1.5 km/h

---

## 5) 통합 로드맵 (Phase-by-Phase)

- Phase 0: 계측/로그만 먼저
  - [ ] 속도/판정 이벤트 로그 CSV 추가(속도, d1/d2, t1/t2, in_poly1/2, reproj_err)
  - [ ] 대시보드에 품질 메트릭 카드(평균 속도 오차, 탐지 유실률 등)
- Phase 1: 색상 모델(백프로젝션 포함) 도입 + 파라미터 위자드
  - [ ] UI 키/옵션으로 “세션 튜닝 모드” 진입→ 샘플링→ 자동 임계 저장
- Phase 2: 마커 스택을 AprilTag+보드+KLT로 교체(인터페이스 호환)
- Phase 3: 광학 업그레이드(형광 슬리브/조명/편광)
- Phase 4: IR/UV 대역 분리(선택)

각 Phase 완료 시 회귀 테스트
- [ ] 배팅장/체육관 클립 세트로 자동 스크립트 평가
- [ ] 메트릭 비교(정밀도/재현율/속도 오차/CPU/FPS)

---

## 6) 메트릭/텔레메트리

- Detection
  - [ ] Precision/Recall/F1 (공 존재 프레임 기준)
  - [ ] FP/FN 타임스탬프 목록
- Speed
  - [ ] v_depth km/h, 95% 구간, 표준편차
- Pose
  - [ ] Reprojection error 평균/최대, pose drop 비율
- Performance
  - [ ] FPS, 프레임 지터 std, CPU/GPU

파일
- [ ] `log/detection_metrics.csv`: t, center_px, area, circ, accepted(bool), reason(outlier?)
- [ ] `log/speed_metrics.csv`: pitch_id, t1, t2, dt, v_mps, v_kmh
- [ ] `log/pose_metrics.csv`: frame_id, reproj_err_mean, num_tags

---

## 7) 위험요소 / 대응

- 동일/유사 색 배경 → 형광색 + 불변량 + 백프로젝션으로 회피
- 강한 반사/하이라이트 → 무광/편광, 조명 균일화
- 모션블러 → 셔터 짧게, ISO 보정, 더 밝은 조명
- 마커 가림/각도 → 보드/멀티태그 + KLT + health check
- 시간 경과 파라미터 드리프트 → 세션 재튜닝/재검출 루틴

---

## 8) 작업 체크리스트(요약)

- [ ] 색/광학: 형광 마젠타, 무광, 노출/WB/ISO 고정, 균일 조명, (선택) 편광
- [ ] 탐지: HSV + rg/ratio 병렬, 원형도/면적/반지름 변화율, 칼만 게이팅, 백프로젝션 튜닝
- [ ] 마커: AprilTag 보드 + KLT + solvePnP + reproj health check
- [ ] 카메라: 셔터/ISO 고정, 60fps 타임스탬프(perf_counter), 왜곡 보정 확인
- [ ] 메트릭: 로그/대시보드 품질 카드 추가
- [ ] 단계별 릴리즈/회귀 테스트

---

## 9) 참고 파라미터 치트시트 (초기값)

- 형광 마젠타(예)
  - HSV: H∈[140,179], S>120, V>120
  - rg: r>0.45, g<0.35
  - 비율: R/G>1.6, R/B>1.8
- 모폴로지: open (3×3) → dilate×2
- 원형도: > 0.55, 면적: 30–8000 px² (해상도 의존)
- Mahalanobis gate: < 7.81
- 반지름 변화율: |Δr/Δt| < 0.6 px/frame (초기치, 환경 보정)
- 재투영 오차 경계: mean < 1.5 px (보드 트래킹 중)

---

## 10) 코드 변경 포인터(추후)

- [ ] `src/color_detector.py` (신규): ColorModel + session fit/apply + backprojection
- [ ] `src/ball_detector_v2.py` (신규): BallDetectorV2.detect() with combined masks
- [ ] `src/marker_tracker.py` (신규): AprilTag 보드 + KLT + solvePnP + health check
- [ ] `src/main.py`:
  - [ ] 컬러 위자드/세션 튜닝 플래그 추가
  - [ ] perf_counter 기반 타임스탬프 로깅 강제
- [ ] `src/dashboard.py`:
  - [ ] 품질 메트릭 카드(Precision, Recall, v_error, reproj_err)

Notes
- 적용 순서는 Phase 로드맵 참조. 배포 전/후 테스트 스크립트로 자동 평가 권장.