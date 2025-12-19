# Android App (On-device)

이 폴더는 **온디바이스 추론 + TTS 피드백 + 서버 전송**을 담당하는 안드로이드 앱 공간입니다.

## 목표
- CameraX로 60fps 1080p 캡처
- TFLite(GPU/NNAPI)로 공 검출/추적
- ArUco + 핀홀 모델 기반 3D 위치 추정(추후)
- 판정 결과를 오프라인 TTS로 즉시 안내
- 결과(JSON)만 서버로 WebSocket 전송

가이드는 `Docs/AR_StrikeZone_Implementation_Guide.md` 참고.


