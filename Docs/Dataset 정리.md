- 심판 시점 카메라[MLB | Skilled Catcher](https://www.youtube.com/results?search_query=UMP+CAM)
- 포수 시점 카메라: [(169) POV BASEBALL - YouTube](https://www.youtube.com/results?search_query=POV+BASEBALL)

---

## 운영 방식(중요)
이 파일은 **검색어/링크 모음**으로 유지합니다.\n
실제로 “다운로드할 유튜브 영상 + 투구 구간(start~end)”은 아래 경로의 **JSON 매니페스트**로 관리합니다:\n
- `data/youtube/manifest/*.json`\n
매니페스트 스키마/예시는 다음을 참고하세요:\n
- `tools/youtube/manifest_schema.md`\n
- `data/youtube/manifest/example.json`\n

다운로드 실행(구간만):\n
```powershell
python tools/youtube/manifest_validate.py --manifest data/youtube/manifest/example.json
python tools/youtube/download_clips.py --manifest data/youtube/manifest/example.json --skip_existing --force_mp4
```


**https://baseballcv.com/background-theory**


예시 검색어:

“catcher POV baseball bullpen”

“GoPro catcher view pitching”

“baseball catcher first person”

“behind the plate view fastball”