# data/

이 폴더의 내용물은 **git에 올라가지 않는다** (`.gitignore` 참조).

## 받아야 할 파일

`LSWMD.pkl` (약 214MB) — WM-811K 웨이퍼맵 데이터셋

## 출처

- Kaggle: `qingyi/wm811k-wafer-map`
- MIR Lab (원본): http://mirlab.org/dataSet/public/

## 배치

    data/LSWMD.pkl

받은 뒤 `python src/load_data.py`를 실행하면 클래스별 `.npz`가
`data/processed/` 아래에 생성된다.
