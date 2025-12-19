# YouTube Dataset Tools

## 1) 매니페스트 작성
- 스키마 문서: `tools/youtube/manifest_schema.md`
- 예시 파일: `data/youtube/manifest/example.json`

매니페스트는 `data/youtube/manifest/*.json`에 두고 관리합니다.

## 2) 매니페스트 검증
```bash
python tools/youtube/0_manifest_validate.py --manifest data/youtube/manifest/example.json
```

WARN도 에러로 취급하고 싶으면:
```bash
python tools/youtube/0_manifest_validate.py --manifest data/youtube/manifest/example.json --strict
```

## 3) 클립 다운로드(구간만)
`yt-dlp`와 `ffmpeg`가 필요합니다.

```bash
pip install yt-dlp
```

ffmpeg 설치 후 PATH에 추가(Windows):
- `choco install ffmpeg` 또는
- ffmpeg zip 다운로드 후 `bin/`을 PATH에 추가

다운로드 실행:
```bash
python tools/youtube/1_download_clips.py ^
  --manifest data/youtube/manifest/example.json ^
  --out_dir data/youtube/clips ^
  --log_dir data/youtube/logs ^
  --skip_existing ^
  --force_mp4
```

### 최고 화질/최고 FPS로 받기(권장)
기본값이 이미 `--format auto`로 되어 있어서 **가능한 한 높은 FPS/해상도**를 우선 선택하고, 없으면 자동으로 fallback합니다.\n
명시적으로 조절하고 싶으면:

```bash
python tools/youtube/1_download_clips.py ^
  --manifest data/youtube/manifest/example.json ^
  --min_fps 60 ^
  --min_height 1080 ^
  --format_sort "fps,res,br" ^
  --skip_existing ^
  --force_mp4
```

- `--min_fps 120`처럼 더 높게도 가능(영상이 지원할 때)\n
- 조건을 끄려면 `--min_fps 0` 또는 `--min_height 0`\n

## 4) 프레임 추출 (라벨링/학습용)
다운받은 클립(mp4)에서 프레임(jpg)을 추출합니다.

```bash
python tools/youtube/2_extract_frames.py ^
  --clips_dir data/youtube/clips ^
  --out_dir data/youtube/frames ^
  --every_n 1 ^
  --skip_existing
```

용량을 아끼려면 클립(mp4)을 추출 후 삭제할 수 있습니다:

```bash
python tools/youtube/2_extract_frames.py ^
  --clips_dir data/youtube/clips ^
  --out_dir data/youtube/frames ^
  --every_n 1 ^
  --skip_existing ^
  --delete_clips
```

출력 예:
- `data/youtube/frames/<clip_stem>/frame_000120.jpg`
- `data/youtube/frames/<clip_stem>/index.json` (프레임 인덱스/시간 포함)

## 5) (선택) YOLO 학습용 dataset 스켈레톤 생성
```bash
python tools/dataset/3_init_yolo_dataset.py
```

## 6) (추천) Ultralytics 학습용 data.yaml 자동 생성
```bash
python tools/dataset/5_generate_data_yaml.py
```

## 출력 파일 네이밍
```
{dataset_id}__{view_type}__{videoId}__{clip_id}__s{start_ms}__e{end_ms}.mp4
```

예:
```
catcher_view_v1__catcher__dQw4w9WgXcQ__p003__s123450__e124120.mp4
```


