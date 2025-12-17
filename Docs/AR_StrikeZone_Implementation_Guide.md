# 🎯 AR Strike Zone 딥러닝 업그레이드 - 완전 구현 가이드

> **목표**: 실제 흰색 야구공 | 60fps 1080p | 휴대폰 온디바이스 추론 + 실시간 음성 | 서버 전송 + 웹 시각화

---

## 📋 목차

1. [시스템 아키텍처 개요](#1-시스템-아키텍처-개요)
2. [Phase 0: 데이터 수집 및 벤치마크](#2-phase-0-데이터-수집-및-벤치마크-1-2주)
3. [Phase 1: 딥러닝 모델 개발](#3-phase-1-딥러닝-모델-개발-2-3주)
4. [Phase 2: 안드로이드 앱 개발](#4-phase-2-안드로이드-앱-개발-3-4주)
5. [Phase 3: 서버 및 웹 대시보드](#5-phase-3-서버-및-웹-대시보드-3-4주)
6. [Phase 4: 물리 기반 폐루프 추적](#6-phase-4-물리-기반-폐루프-추적-2-3주)
7. [Phase 5: 고급 데이터 증강](#7-phase-5-고급-데이터-증강-2주)
8. [필수 참고 논문 목록](#8-필수-참고-논문-목록)
9. [기술 스택 상세](#9-기술-스택-상세)
10. [실험 설계 및 평가](#10-실험-설계-및-평가)

---

## 1. 시스템 아키텍처 개요

### 1.1 전체 시스템 흐름

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        📱 Android App (On-Device)                        │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ CameraX  │───▶│  YOLOv8n │───▶│ 3D 좌표  │───▶│  판정    │          │
│  │ 60fps    │    │ TFLite   │    │ 변환     │    │ 로직    │          │
│  │ 1080p    │    │ INT8     │    │ (핀홀+   │    │ Strike/ │          │
│  └──────────┘    └──────────┘    │ ArUco)   │    │ Ball    │          │
│       │               │          └──────────┘    └────┬─────┘          │
│       │               │                               │                │
│       ▼               ▼                               ▼                │
│  ┌──────────┐    ┌──────────┐                   ┌──────────┐          │
│  │ AR 오버  │    │ 칼만     │                   │ TTS      │          │
│  │ 레이     │    │ 필터     │                   │ 음성출력 │          │
│  └──────────┘    └──────────┘                   └──────────┘          │
│                                                       │                │
└───────────────────────────────────────────────────────┼────────────────┘
                                                        │
                            WebSocket (JSON)            │
                                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          🖥️ Backend Server                              │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                          │
│  │ FastAPI  │───▶│ Postgres │───▶│ Redis    │                          │
│  │ WebSocket│    │ Timescale│    │ Cache    │                          │
│  └──────────┘    └──────────┘    └──────────┘                          │
└───────────────────────────────────────────────────────┬────────────────┘
                                                        │
                            WebSocket (Real-time)       │
                                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        🌐 Web Dashboard (React)                         │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ 3D 궤적  │    │ 히트맵   │    │ 구속     │    │ 계정별   │          │
│  │ Three.js │    │ Plotly   │    │ 통계     │    │ 기록     │          │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 핵심 설계 원칙

| 원칙 | 설명 | 구현 방향 |
|------|------|----------|
| **온디바이스 우선** | 모든 추론은 휴대폰에서 실행 | TFLite INT8 + GPU delegate |
| **기존 파이프라인 재사용** | ArUco + 핀홀 깊이 추정 유지 | 검출부만 딥러닝으로 교체 |
| **실시간 피드백** | 현장에서 즉각적인 판정 | 오프라인 TTS + 저지연 추론 |
| **분석은 서버에서** | 상세 시각화/통계는 웹 | 결과 JSON만 전송 |

### 1.3 지연 시간 목표 (End-to-End)

```
캡처 → 전처리 → 추론 → 후처리 → 판정 → TTS 출력
 5ms     10ms     30ms    5ms      1ms     50ms
                                          ─────────
                                    총 목표: < 150ms
```

---

## 2. Phase 0: 데이터 수집 및 벤치마크 (1-2주)

### 2.1 데이터 수집 요구사항

#### 2.1.1 촬영 환경 체크리스트

```markdown
## 필수 촬영 환경 (각 환경에서 최소 30구 이상)

### 조명 조건
- [ ] 맑은 낮 (직사광선)
- [ ] 흐린 낮 (확산광)
- [ ] 역광 (태양이 카메라 뒤)
- [ ] 야간 (조명등 아래)
- [ ] 실내 (형광등/LED)

### 배경 조건
- [ ] 깨끗한 녹색 잔디 배경
- [ ] 흰색 유니폼 착용 투수
- [ ] 관중석/광고판 포함
- [ ] 포수 장비 포함
- [ ] 빈 프레임 (공 없음) - 최소 100프레임

### 투구 종류
- [ ] 직구 (120-150 km/h)
- [ ] 변화구 (커브, 슬라이더)
- [ ] 체인지업 (느린 공)
```

#### 2.1.2 카메라 설정 스펙

```yaml
camera_settings:
  resolution: 1920x1080
  frame_rate: 60fps
  codec: H.264 또는 H.265
  bitrate: 50Mbps 이상
  
positioning:
  distance_from_plate: 18.44m (마운드 거리) 또는 그 뒤
  height: 1.2m - 1.5m (포수 시점 근사)
  angle: 정면 또는 약간 측면 (±15°)
  
aruco_marker:
  size: 20cm x 20cm
  dictionary: DICT_6X6_250
  placement: 홈플레이트 옆 또는 앞
```

### 2.2 라벨링 규칙 상세

#### 2.2.1 YOLO 형식 라벨링

```
# 라벨 파일 형식: frame_000001.txt
# class_id  x_center  y_center  width  height
# (모두 0-1로 정규화)

0 0.5234 0.4521 0.0156 0.0278
```

#### 2.2.2 라벨링 도구 추천

| 도구 | 특징 | 추천 용도 |
|------|------|----------|
| **CVAT** | 웹 기반, 비디오 지원, 인터폴레이션 | 대량 비디오 라벨링 |
| **LabelImg** | 가볍고 빠름, YOLO 형식 직접 지원 | 빠른 이미지 라벨링 |
| **Roboflow** | 자동 증강, 데이터셋 관리 | 팀 협업, 버전 관리 |

#### 2.2.3 라벨링 가이드라인

```markdown
## 공 라벨링 규칙

1. **바운딩 박스 크기**
   - 공 외곽을 꽉 채우는 정사각형에 가깝게
   - 모션 블러가 있어도 공의 "핵심 영역"만 포함
   - 블러 꼬리는 제외

2. **모호한 상황 처리**
   - 공이 완전히 보이지 않으면 → 라벨링 안 함
   - 50% 이상 가려지면 → 라벨링 안 함
   - 모션 블러로 인해 원형이 아니면 → 블러 중심부만

3. **하드 네거티브 (공 없는 프레임)**
   - 빈 라벨 파일 생성 (0바이트 .txt)
   - 흰색 유사 객체 포함 프레임 필수
```

### 2.3 벤치마크 지표 정의

#### 2.3.1 검출 성능 지표

```python
# 평가 지표 계산 예시
from collections import defaultdict

class DetectionMetrics:
    """
    소형 객체(야구공) 검출 성능 평가 지표
    """
    
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.results = defaultdict(list)
    
    def calculate_iou(self, box1, box2):
        """IoU (Intersection over Union) 계산"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def evaluate(self, predictions, ground_truths):
        """
        Returns:
            precision: TP / (TP + FP)
            recall: TP / (TP + FN)
            mAP: mean Average Precision
        """
        # 구현 상세...
        pass

# 목표 성능
TARGET_METRICS = {
    'mAP@0.5': 0.85,           # 85% 이상
    'Recall': 0.90,            # 90% 이상 (놓치면 안 됨)
    'Precision': 0.80,         # 80% 이상
    'mAP_small': 0.75,         # 소형 객체(32x32 미만) 75% 이상
}
```

#### 2.3.2 추적 성능 지표

```python
class TrackingMetrics:
    """
    궤적 추적 성능 평가 지표
    """
    
    def calculate_ade(self, pred_trajectory, gt_trajectory):
        """
        ADE (Average Displacement Error)
        - 예측 궤적과 실제 궤적 사이의 평균 거리
        """
        errors = []
        for pred, gt in zip(pred_trajectory, gt_trajectory):
            dist = np.sqrt((pred[0]-gt[0])**2 + (pred[1]-gt[1])**2 + (pred[2]-gt[2])**2)
            errors.append(dist)
        return np.mean(errors)
    
    def calculate_track_retention(self, detections, total_frames):
        """
        트랙 유지율: 투구 구간에서 공을 검출한 프레임 비율
        """
        return len(detections) / total_frames

# 목표 성능
TARGET_TRACKING = {
    'ADE': 5.0,                 # 5cm 이하
    'Track_Retention': 0.85,   # 85% 이상
    'ID_Switches': 0,          # 단일 객체이므로 0
}
```

#### 2.3.3 시스템 지연 지표

```python
class LatencyMetrics:
    """
    시스템 지연 시간 측정
    """
    
    def measure_pipeline_latency(self):
        """
        각 단계별 지연 시간 측정 (ms)
        """
        return {
            'capture': 0,           # 프레임 캡처
            'preprocess': 0,        # 리사이즈, 정규화
            'inference': 0,         # 모델 추론
            'postprocess': 0,       # NMS, 좌표 변환
            'judgment': 0,          # 판정 로직
            'tts': 0,               # 음성 출력
            'total': 0              # 전체
        }

# 목표 지연
TARGET_LATENCY = {
    'inference': 35,        # 35ms 이하 (≈28fps)
    'total': 150,           # 150ms 이하 (체감 실시간)
    'tts_queue': 100,       # TTS 큐잉 100ms 이하
}
```

### 2.4 데이터셋 디렉토리 구조

```
dataset/
├── raw/                          # 원본 영상
│   ├── day_sunny/
│   ├── day_cloudy/
│   ├── night/
│   └── indoor/
│
├── frames/                       # 추출된 프레임
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│
├── augmented/                    # 증강된 데이터
│   ├── motion_blur/
│   ├── exposure/
│   └── background_swap/
│
├── metadata/
│   ├── pitch_info.json          # 투구 정보 (구속, 구종 등)
│   └── camera_calibration.json  # 카메라 캘리브레이션
│
└── benchmarks/
    ├── detection_results.json
    ├── tracking_results.json
    └── latency_results.json
```

---

## 3. Phase 1: 딥러닝 모델 개발 (2-3주)

### 3.1 모델 선택 근거

#### 3.1.1 모델 비교표

| 모델 | 크기 (MB) | mAP@0.5 | 모바일 FPS | 소형 객체 성능 |
|------|----------|---------|-----------|---------------|
| **YOLOv8n** | 6.2 | 37.3 | 25-35 | ⭐⭐⭐⭐ |
| YOLOv8s | 22.5 | 44.9 | 15-20 | ⭐⭐⭐⭐⭐ |
| MobileNet-SSD | 5.8 | 20-25 | 40-50 | ⭐⭐ |
| EfficientDet-Lite0 | 4.4 | 25.7 | 20-30 | ⭐⭐⭐ |
| NanoDet-Plus | 4.7 | 30.4 | 30-40 | ⭐⭐⭐ |

> **선택: YOLOv8n** - 소형 객체 성능과 속도의 균형이 가장 좋음

#### 3.1.2 YOLOv8n 아키텍처 특징

```
YOLOv8n 구조:
├── Backbone: CSPDarknet (경량화)
│   └── P3, P4, P5 피처 추출
├── Neck: PAFPN (양방향 피처 융합)
│   └── 멀티스케일 피처 결합
└── Head: Decoupled Head
    └── 분류/회귀 분리 → 소형 객체에 유리

핵심 장점:
1. Anchor-free 설계 → 작은 객체 검출 유연성
2. PAFPN → 고해상도 피처 보존
3. 6MB 경량 → 모바일 최적화
```

### 3.2 학습 환경 설정

#### 3.2.1 환경 설정 코드

```python
# train_config.py
from ultralytics import YOLO

# 데이터셋 설정 (data.yaml)
DATA_YAML = """
path: ./dataset
train: frames/images/train
val: frames/images/val
test: frames/images/test

nc: 1  # 클래스 수 (baseball만)
names: ['baseball']
"""

# 학습 하이퍼파라미터
TRAIN_CONFIG = {
    'model': 'yolov8n.pt',          # 사전학습 가중치
    'data': 'data.yaml',
    'epochs': 300,
    'imgsz': 416,                    # 입력 크기
    'batch': 16,
    'device': 0,                     # GPU
    
    # 소형 객체 최적화
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    
    # 증강 (초기에는 보수적으로)
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 0,                    # 회전 안 함 (공은 구형)
    'translate': 0.1,
    'scale': 0.5,
    'flipud': 0.0,                   # 상하반전 안 함
    'fliplr': 0.5,                   # 좌우반전 허용
    'mosaic': 1.0,
    'mixup': 0.1,
}

def train():
    model = YOLO('yolov8n.pt')
    results = model.train(**TRAIN_CONFIG)
    return results
```

#### 3.2.2 소형 객체 특화 수정 (선택적)

```python
# 고해상도 피처 헤드 추가 (P2 레벨)
# ultralytics/nn/modules/head.py 수정

"""
기본 YOLOv8: P3(80x80), P4(40x40), P5(20x20)
수정 버전: P2(160x160) 추가 → 작은 공 검출 향상

효과: 10-15px 크기 객체 검출률 약 15% 향상
비용: 추론 시간 약 20% 증가
"""

# 또는 입력 해상도 증가로 대체 (더 간단)
# imgsz: 416 → 640 (추론 시간 2배 증가 주의)
```

### 3.3 TFLite 변환 및 최적화

#### 3.3.1 변환 파이프라인

```python
# export_tflite.py
from ultralytics import YOLO
import tensorflow as tf

def export_to_tflite(model_path, output_path, quantize=True):
    """
    YOLOv8 → ONNX → TFLite 변환
    """
    # 1. YOLO 모델 로드
    model = YOLO(model_path)
    
    # 2. TFLite로 직접 내보내기 (Ultralytics 지원)
    model.export(
        format='tflite',
        imgsz=416,
        half=False,           # FP16 비활성화 (INT8 사용 시)
        int8=quantize,        # INT8 양자화
        data='data.yaml',     # 캘리브레이션용 데이터
    )
    
    print(f"Exported to: {output_path}")

def create_representative_dataset():
    """
    INT8 양자화를 위한 대표 데이터셋 생성
    - 최소 100개 이상의 대표 이미지 필요
    """
    import glob
    import cv2
    import numpy as np
    
    image_paths = glob.glob('dataset/frames/images/train/*.jpg')[:200]
    
    def representative_data_gen():
        for path in image_paths:
            img = cv2.imread(path)
            img = cv2.resize(img, (416, 416))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            yield [img]
    
    return representative_data_gen

# 변환 실행
export_to_tflite('runs/detect/train/weights/best.pt', 'baseball_detector.tflite')
```

#### 3.3.2 모델 크기 및 성능 비교

```
변환 결과 비교:

| 형식           | 크기    | 추론 시간 (Pixel 6) | mAP 변화 |
|---------------|---------|-------------------|----------|
| PyTorch (FP32)| 6.2 MB  | -                 | 기준     |
| ONNX (FP32)   | 12.4 MB | 45ms              | 0%       |
| TFLite (FP32) | 12.4 MB | 40ms              | 0%       |
| TFLite (FP16) | 6.2 MB  | 32ms              | -0.1%    |
| TFLite (INT8) | 3.2 MB  | 25ms              | -1~2%    |

→ INT8 권장: 크기 절반, 속도 1.6배, 정확도 손실 미미
```

### 3.4 검증 및 테스트

#### 3.4.1 TFLite 모델 검증 코드

```python
# validate_tflite.py
import numpy as np
import tensorflow as tf
import cv2
import time

class TFLiteDetector:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details[0]['shape'][1:3]
    
    def preprocess(self, image):
        """입력 이미지 전처리"""
        img = cv2.resize(image, tuple(self.input_shape))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    
    def detect(self, image, conf_threshold=0.5):
        """객체 검출 수행"""
        # 전처리
        input_data = self.preprocess(image)
        
        # 추론
        start_time = time.time()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000
        
        # 출력 파싱 (YOLOv8 출력 형식)
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # NMS 및 후처리
        detections = self.postprocess(output, conf_threshold)
        
        return detections, inference_time
    
    def postprocess(self, output, conf_threshold):
        """YOLOv8 출력 후처리"""
        # output shape: [1, 84, 8400] for YOLOv8n
        # 84 = 4 (bbox) + 80 (classes) → 우리는 1 class만 사용
        
        predictions = output[0].T  # [8400, 84]
        
        # 신뢰도 필터링
        scores = predictions[:, 4]  # objectness score
        mask = scores > conf_threshold
        
        boxes = predictions[mask, :4]
        scores = scores[mask]
        
        # NMS 적용
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(),
            conf_threshold, 0.45
        )
        
        return [(boxes[i], scores[i]) for i in indices.flatten()]

# 벤치마크 실행
def benchmark_model(model_path, test_images, num_runs=100):
    detector = TFLiteDetector(model_path)
    
    latencies = []
    for img_path in test_images[:num_runs]:
        img = cv2.imread(img_path)
        _, latency = detector.detect(img)
        latencies.append(latency)
    
    print(f"Average Inference Time: {np.mean(latencies):.2f}ms")
    print(f"Std: {np.std(latencies):.2f}ms")
    print(f"Min: {np.min(latencies):.2f}ms")
    print(f"Max: {np.max(latencies):.2f}ms")
    print(f"FPS: {1000/np.mean(latencies):.1f}")
```

---

## 4. Phase 2: 안드로이드 앱 개발 (3-4주)

### 4.1 프로젝트 구조

```
app/
├── src/main/
│   ├── java/com/strikezone/
│   │   ├── MainActivity.kt
│   │   ├── camera/
│   │   │   ├── CameraManager.kt
│   │   │   └── FrameAnalyzer.kt
│   │   ├── detection/
│   │   │   ├── BallDetector.kt
│   │   │   └── TFLiteWrapper.kt
│   │   ├── tracking/
│   │   │   ├── KalmanTracker.kt
│   │   │   └── PhysicsModel.kt
│   │   ├── judgment/
│   │   │   ├── StrikeZone.kt
│   │   │   └── PitchJudgment.kt
│   │   ├── ar/
│   │   │   ├── ArUcoDetector.kt
│   │   │   └── AROverlay.kt
│   │   ├── audio/
│   │   │   └── TTSManager.kt
│   │   └── network/
│   │       └── WebSocketClient.kt
│   ├── assets/
│   │   └── baseball_detector.tflite
│   └── res/
│       └── ...
└── build.gradle.kts
```

### 4.2 핵심 컴포넌트 구현

#### 4.2.1 CameraX 설정 (60fps 1080p)

```kotlin
// CameraManager.kt
class CameraManager(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val analyzer: ImageAnalysis.Analyzer
) {
    private lateinit var cameraProvider: ProcessCameraProvider
    
    fun startCamera(previewView: PreviewView) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            
            // Preview 설정
            val preview = Preview.Builder()
                .setTargetResolution(Size(1920, 1080))
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }
            
            // ImageAnalysis 설정 (60fps 목표)
            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(1920, 1080))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(ContextCompat.getMainExecutor(context), analyzer) }
            
            // 카메라 바인딩
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            
            try {
                cameraProvider.unbindAll()
                val camera = cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    preview,
                    imageAnalysis
                )
                
                // 수동 노출/포커스 설정 (선택적)
                setupCameraControls(camera)
                
            } catch (e: Exception) {
                Log.e("CameraManager", "Camera binding failed", e)
            }
        }, ContextCompat.getMainExecutor(context))
    }
    
    private fun setupCameraControls(camera: Camera) {
        // 고속 셔터 설정 (모션 블러 감소)
        camera.cameraControl.setExposureCompensationIndex(-1)
    }
}
```

#### 4.2.2 TFLite 추론 래퍼

```kotlin
// TFLiteWrapper.kt
class TFLiteWrapper(context: Context, modelPath: String) {
    
    private val interpreter: Interpreter
    private val inputShape: IntArray
    private val outputShape: IntArray
    
    // GPU Delegate 사용
    private val gpuDelegate: GpuDelegate?
    
    init {
        // GPU Delegate 초기화
        gpuDelegate = try {
            GpuDelegate(GpuDelegate.Options().apply {
                setPrecisionLossAllowed(true)  // 성능 향상
                setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER)
            })
        } catch (e: Exception) {
            Log.w("TFLite", "GPU Delegate not available, falling back to CPU")
            null
        }
        
        // Interpreter 옵션
        val options = Interpreter.Options().apply {
            setNumThreads(4)
            gpuDelegate?.let { addDelegate(it) }
        }
        
        // 모델 로드
        val modelBuffer = loadModelFile(context, modelPath)
        interpreter = Interpreter(modelBuffer, options)
        
        // 입출력 형태 저장
        inputShape = interpreter.getInputTensor(0).shape()
        outputShape = interpreter.getOutputTensor(0).shape()
        
        Log.d("TFLite", "Model loaded: input=$inputShape, output=$outputShape")
    }
    
    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(modelPath)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            assetFileDescriptor.startOffset,
            assetFileDescriptor.declaredLength
        )
    }
    
    fun detect(bitmap: Bitmap, confThreshold: Float = 0.5f): List<Detection> {
        // 전처리
        val inputBuffer = preprocessBitmap(bitmap)
        
        // 출력 버퍼 준비
        val outputBuffer = Array(1) { Array(outputShape[1]) { FloatArray(outputShape[2]) } }
        
        // 추론
        val startTime = SystemClock.elapsedRealtimeNanos()
        interpreter.run(inputBuffer, outputBuffer)
        val inferenceTime = (SystemClock.elapsedRealtimeNanos() - startTime) / 1_000_000f
        
        Log.d("TFLite", "Inference time: ${inferenceTime}ms")
        
        // 후처리
        return postprocess(outputBuffer[0], confThreshold, bitmap.width, bitmap.height)
    }
    
    private fun preprocessBitmap(bitmap: Bitmap): ByteBuffer {
        val inputSize = inputShape[1]  // 416
        
        // 리사이즈
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        
        // ByteBuffer 생성 (FLOAT32)
        val buffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
        buffer.order(ByteOrder.nativeOrder())
        
        val pixels = IntArray(inputSize * inputSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        
        for (pixel in pixels) {
            // RGB 추출 및 정규화 (0-1)
            buffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)  // R
            buffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)   // G
            buffer.putFloat((pixel and 0xFF) / 255.0f)           // B
        }
        
        buffer.rewind()
        return buffer
    }
    
    private fun postprocess(
        output: Array<FloatArray>,
        confThreshold: Float,
        origWidth: Int,
        origHeight: Int
    ): List<Detection> {
        val detections = mutableListOf<Detection>()
        
        // YOLOv8 출력: [84, 8400] → transpose
        // 84 = 4 (xywh) + 80 (classes) → 우리는 1 class만
        
        for (i in 0 until output[0].size) {
            val confidence = output[4][i]  // class 0 confidence
            
            if (confidence > confThreshold) {
                // 좌표 복원 (0-1 → 원본 크기)
                val cx = output[0][i] * origWidth
                val cy = output[1][i] * origHeight
                val w = output[2][i] * origWidth
                val h = output[3][i] * origHeight
                
                detections.add(Detection(
                    centerX = cx,
                    centerY = cy,
                    width = w,
                    height = h,
                    confidence = confidence
                ))
            }
        }
        
        // NMS 적용
        return applyNMS(detections, 0.45f)
    }
    
    private fun applyNMS(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        if (detections.isEmpty()) return emptyList()
        
        val sorted = detections.sortedByDescending { it.confidence }
        val selected = mutableListOf<Detection>()
        val active = BooleanArray(sorted.size) { true }
        
        for (i in sorted.indices) {
            if (!active[i]) continue
            selected.add(sorted[i])
            
            for (j in i + 1 until sorted.size) {
                if (active[j] && calculateIoU(sorted[i], sorted[j]) > iouThreshold) {
                    active[j] = false
                }
            }
        }
        
        return selected
    }
    
    fun close() {
        interpreter.close()
        gpuDelegate?.close()
    }
}

data class Detection(
    val centerX: Float,
    val centerY: Float,
    val width: Float,
    val height: Float,
    val confidence: Float
) {
    val radius: Float get() = (width + height) / 4  // 평균 반지름
}
```

#### 4.2.3 3D 좌표 변환 (핀홀 모델)

```kotlin
// CoordinateTransformer.kt
class CoordinateTransformer(
    private val cameraMatrix: FloatArray,  // 3x3 내부 파라미터
    private val distCoeffs: FloatArray,    // 왜곡 계수
    private val ballRadiusReal: Float = 0.0365f  // 야구공 반지름 3.65cm
) {
    
    /**
     * 2D 검출 → 3D 카메라 좌표 변환
     * 
     * @param detection 검출 결과 (픽셀 좌표)
     * @return 3D 위치 (카메라 좌표계, 미터)
     */
    fun estimateDepth(detection: Detection): FloatArray {
        // 핀홀 모델: Z = (f * R_real) / r_pixel
        val focalLength = cameraMatrix[0]  // fx (픽셀 단위)
        
        // 깊이 추정
        val z = (focalLength * ballRadiusReal) / detection.radius
        
        // 2D → 3D 역투영
        val cx = cameraMatrix[2]  // 주점 x
        val cy = cameraMatrix[5]  // 주점 y
        
        val x = (detection.centerX - cx) * z / focalLength
        val y = (detection.centerY - cy) * z / focalLength
        
        return floatArrayOf(x, y, z)
    }
    
    /**
     * 카메라 좌표 → ArUco 마커 좌표 변환
     * 
     * @param point3D 카메라 좌표계 3D 점
     * @param rvec ArUco 회전 벡터
     * @param tvec ArUco 이동 벡터
     * @return 마커 좌표계 3D 점
     */
    fun transformToMarkerCoord(
        point3D: FloatArray,
        rvec: FloatArray,
        tvec: FloatArray
    ): FloatArray {
        // 회전 행렬 계산
        val rotMat = rodrigues(rvec)
        
        // 변환: P_marker = R^T * (P_cam - t)
        val translated = floatArrayOf(
            point3D[0] - tvec[0],
            point3D[1] - tvec[1],
            point3D[2] - tvec[2]
        )
        
        // R^T * translated
        val result = FloatArray(3)
        for (i in 0..2) {
            result[i] = rotMat[i] * translated[0] + 
                        rotMat[3 + i] * translated[1] + 
                        rotMat[6 + i] * translated[2]
        }
        
        return result
    }
    
    private fun rodrigues(rvec: FloatArray): FloatArray {
        // OpenCV의 Rodrigues 공식 구현
        val theta = sqrt(rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2])
        
        if (theta < 1e-6) {
            return floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
        }
        
        val k = floatArrayOf(rvec[0]/theta, rvec[1]/theta, rvec[2]/theta)
        val c = cos(theta)
        val s = sin(theta)
        
        // 회전 행렬
        return floatArrayOf(
            c + k[0]*k[0]*(1-c),     k[0]*k[1]*(1-c) - k[2]*s,  k[0]*k[2]*(1-c) + k[1]*s,
            k[1]*k[0]*(1-c) + k[2]*s, c + k[1]*k[1]*(1-c),       k[1]*k[2]*(1-c) - k[0]*s,
            k[2]*k[0]*(1-c) - k[1]*s, k[2]*k[1]*(1-c) + k[0]*s,  c + k[2]*k[2]*(1-c)
        )
    }
}
```

#### 4.2.4 TTS 음성 출력

```kotlin
// TTSManager.kt
class TTSManager(private val context: Context) : TextToSpeech.OnInitListener {
    
    private var tts: TextToSpeech? = null
    private var isInitialized = false
    
    // 음성 출력 큐 (중복 방지)
    private var lastSpokenTime = 0L
    private val minSpeakInterval = 500L  // 최소 0.5초 간격
    
    init {
        tts = TextToSpeech(context, this)
    }
    
    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            // 한국어 설정
            val result = tts?.setLanguage(Locale.KOREAN)
            
            if (result == TextToSpeech.LANG_MISSING_DATA || 
                result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e("TTS", "Korean language not supported")
                // 영어로 폴백
                tts?.setLanguage(Locale.US)
            }
            
            // 음성 속도 설정 (빠르게)
            tts?.setSpeechRate(1.2f)
            tts?.setPitch(1.0f)
            
            isInitialized = true
        }
    }
    
    fun speak(text: String, priority: Int = TextToSpeech.QUEUE_FLUSH) {
        if (!isInitialized) return
        
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastSpokenTime < minSpeakInterval) {
            return  // 너무 빠른 연속 출력 방지
        }
        
        lastSpokenTime = currentTime
        tts?.speak(text, priority, null, "pitch_result_${currentTime}")
    }
    
    fun speakJudgment(judgment: PitchJudgment) {
        val text = when (judgment) {
            PitchJudgment.STRIKE -> "스트라이크"
            PitchJudgment.BALL -> "볼"
            PitchJudgment.STRIKE_OUT -> "스트라이크 아웃"
            PitchJudgment.WALK -> "볼넷"
        }
        speak(text)
    }
    
    fun speakSpeed(speedKmh: Float) {
        speak("${speedKmh.toInt()} 킬로")
    }
    
    fun shutdown() {
        tts?.stop()
        tts?.shutdown()
    }
}

enum class PitchJudgment {
    STRIKE, BALL, STRIKE_OUT, WALK
}
```

#### 4.2.5 WebSocket 클라이언트

```kotlin
// WebSocketClient.kt
class WebSocketClient(
    private val serverUrl: String,
    private val onMessage: (String) -> Unit,
    private val onError: (Exception) -> Unit
) {
    private var webSocket: WebSocket? = null
    private val client = OkHttpClient.Builder()
        .pingInterval(30, TimeUnit.SECONDS)
        .build()
    
    // 오프라인 버퍼
    private val offlineBuffer = mutableListOf<PitchData>()
    private var isConnected = false
    
    fun connect() {
        val request = Request.Builder()
            .url(serverUrl)
            .build()
        
        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                isConnected = true
                Log.d("WebSocket", "Connected to server")
                
                // 오프라인 버퍼 전송
                flushOfflineBuffer()
            }
            
            override fun onMessage(webSocket: WebSocket, text: String) {
                onMessage(text)
            }
            
            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                isConnected = false
                onError(t as Exception)
            }
            
            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                isConnected = false
            }
        })
    }
    
    fun sendPitchData(data: PitchData) {
        val json = Gson().toJson(data)
        
        if (isConnected) {
            webSocket?.send(json)
        } else {
            // 오프라인 버퍼에 저장
            offlineBuffer.add(data)
            if (offlineBuffer.size > 1000) {
                offlineBuffer.removeAt(0)  // 오래된 데이터 제거
            }
        }
    }
    
    private fun flushOfflineBuffer() {
        offlineBuffer.forEach { data ->
            val json = Gson().toJson(data)
            webSocket?.send(json)
        }
        offlineBuffer.clear()
    }
    
    fun disconnect() {
        webSocket?.close(1000, "Client closing")
    }
}

data class PitchData(
    val timestamp: Long,
    val userId: String,
    val trajectory: List<Point3D>,      // 3D 궤적
    val speed: Float,                    // km/h
    val judgment: String,                // STRIKE/BALL
    val crossingPoint: Point3D?,         // 존 통과 위치
    val pitchType: String? = null        // 구종 (선택)
)

data class Point3D(
    val x: Float,
    val y: Float,
    val z: Float,
    val timestamp: Long
)
```

### 4.3 앱 권한 및 설정

```xml
<!-- AndroidManifest.xml -->
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    
    <!-- 권한 -->
    <uses-permission android:name="android.permission.CAMERA" />
    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
    
    <!-- 하드웨어 요구사항 -->
    <uses-feature android:name="android.hardware.camera" android:required="true" />
    <uses-feature android:name="android.hardware.camera.autofocus" />
    
    <application
        android:name=".StrikeZoneApp"
        android:hardwareAccelerated="true"
        android:largeHeap="true">
        
        <!-- ... -->
        
    </application>
</manifest>
```

```groovy
// build.gradle (app)
dependencies {
    // CameraX
    def camerax_version = "1.3.1"
    implementation "androidx.camera:camera-core:$camerax_version"
    implementation "androidx.camera:camera-camera2:$camerax_version"
    implementation "androidx.camera:camera-lifecycle:$camerax_version"
    implementation "androidx.camera:camera-view:$camerax_version"
    
    // TensorFlow Lite
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    
    // OpenCV (ArUco용)
    implementation 'org.opencv:opencv:4.8.0'
    
    // 네트워크
    implementation 'com.squareup.okhttp3:okhttp:4.12.0'
    implementation 'com.google.code.gson:gson:2.10.1'
    
    // Coroutines
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
}
```

---

**(계속: Phase 3~10은 다음 파일에...)**

