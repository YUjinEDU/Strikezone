# Dataset Tools

## 1) 유튜브 클립 수집 → 프레임 추출
1. `tools/youtube/`로 클립 다운로드
2. `tools/youtube/extract_frames.py`로 프레임(jpg) 추출

## 2) YOLO 학습용 폴더 스켈레톤 생성
```bash
python tools/dataset/3_init_yolo_dataset.py
```

생성되는 구조(기본):
- `dataset/frames/images/{train,val,test}`
- `dataset/frames/labels/{train,val,test}`
- `dataset/metadata/`
- `dataset/benchmarks/`

## 다음 단계(라벨링)
프레임들을 `dataset/frames/images/*`로 배치한 뒤, CVAT/LabelImg로 라벨링해서\n
같은 파일명으로 `dataset/frames/labels/*`에 YOLO 텍스트 라벨을 생성하면 됩니다.

## 3) 라벨링 완료 후 train/val/test split
라벨이 생성된 뒤(이미지+txt가 짝이 맞는 상태) 아래로 분배하세요.

```bash
python tools/dataset/4_split_dataset.py --frames_root data/youtube/frames --dataset_dir dataset/frames --by clip --val_ratio 0.1 --test_ratio 0.1 --seed 42
```

## 4) Ultralytics 학습용 data.yaml 자동 생성
```bash
python tools/dataset/5_generate_data_yaml.py
```


