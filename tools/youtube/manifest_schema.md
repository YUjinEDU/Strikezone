# YouTube Dataset Manifest Schema

이 문서는 **유튜브 영상 링크 + 투구(클립) 구간**을 체계적으로 관리하기 위한 JSON 스키마를 정의합니다.

## 목표
- 유튜브 원본 영상에서 **여러 개 투구 구간**(투구 시작~포수 미트 도착 등)을 **초 단위**로 기록
- JSON만으로 **구간 클립을 재현 가능**하게 다운로드/네이밍

## 시간 단위 규칙
- 모든 구간은 **초 단위 float**로 기록합니다.
  - `start_sec`: 구간 시작 (예: `123.45`)
  - `end_sec`: 구간 끝 (예: `124.12`)
- 내부적으로는 파일명에 **밀리초 정수(ms)** 로 변환하여 저장합니다.

## Top-level
```json
{
  "dataset_id": "catcher_view_v1",
  "created_at": "2025-12-17",
  "default": {
    "view_type": "catcher"
  },
  "videos": []
}
```

### 필드
- `dataset_id` (string, required): 데이터셋 식별자
- `created_at` (string, optional): 생성일(자유 형식)
- `default` (object, optional): 비디오/클립 기본값
  - `view_type` (string): 기본 시점
- `videos` (array, required): 유튜브 영상 목록

## videos[]
```json
{
  "video_id": "dQw4w9WgXcQ",
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "source": "youtube",
  "view_type": "catcher",
  "meta": {
    "title": "Example",
    "channel": "ExampleChannel",
    "event": "practice",
    "resolution_hint": "1080p60"
  },
  "clips": []
}
```

### 필드
- `video_id` (string, optional): 유튜브 video id (없으면 URL에서 자동 추출 가능)
- `url` (string, required): 유튜브 URL
- `source` (string, optional): 기본 `youtube`
- `view_type` (string, optional): 시점 유형 (없으면 `default.view_type` 사용)
  - 권장 값: `catcher`, `umpire`, `side_1b`, `side_3b`, `diagonal`, `broadcast`
- `meta` (object, optional): 검색/정리를 위한 메타데이터
- `clips` (array, required): 이 영상에서 추출할 투구 구간들

## clips[]
```json
{
  "clip_id": "p001",
  "start_sec": 123.45,
  "end_sec": 124.12,
  "tags": ["fastball", "day"],
  "notes": "투구 시작~미트 도착"
}
```

### 필드
- `clip_id` (string, required): 영상 내에서 유니크한 클립 ID (예: `p001`)
- `start_sec` (number, required): 시작 시간(초)
- `end_sec` (number, required): 종료 시간(초)
- `tags` (array[string], optional): 검색/필터용 태그
- `notes` (string, optional): 자유 메모

## 파일 네이밍 규칙 (클립 mp4)
기본 규칙:
```
{dataset_id}__{view_type}__{videoId}__{clip_id}__s{start_ms}__e{end_ms}.mp4
```

예시:
```
catcher_view_v1__catcher__dQw4w9WgXcQ__p003__s123450__e124120.mp4
```

## 권장 운영 방식
- `Docs/Dataset 정리.md`에는 **검색어/링크 모음**을 유지
- 실제 수집 대상(다운로드 재현 가능한 목록)은 `data/youtube/manifest/*.json`에 저장


