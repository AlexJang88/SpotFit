from __future__ import annotations

import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = (ROOT / "data" / "raw").resolve()
TARGET = (RAW / "안전" / "생활안전").resolve()

if RAW not in TARGET.parents or TARGET == RAW:
    raise RuntimeError(f"안전하지 않은 대상 경로: {TARGET}")

FOLDERS = {
    "01": "01_인구만명당_생활안전관련_119구급건수",
    "02": "02_인구만명당_제조업종사자수",
    "03": "03_인구만명당_재난약자수",
    "04": "04_주상공관리지역면적당_및_인구만명당_AED설치대수",
    "05": "05_공공질서및안전예산지출액비율",
    "06": "06_인구만명당_생활안전관련_안전신문고신고건수",
    "07": "07_기준연도대비_응급처치교육인원수",
}

# 기존 파일을 공식 지표와의 관계에 따라 명시적으로만 이동한다.
# 이름이 비슷하다는 이유만으로 공식 지표 자료로 간주하지 않는다.
MOVES = [
    (TARGET / "01_인구만명당_생활안전관련_119구급건수" / "생활안전_구급자료.xlsx", "01", "참고", "전체 구급자료로 공식 17개 생활안전 사고유형 자료와 다름"),
    (TARGET / "01_인구만명당_생활안전관련_119구급건수" / "안전한동네_구급.xlsx", "01", "참고", "전체 구급자료; 공식 지표 산출 전 필터 필요"),
    (TARGET / "04_주상공관리지역면적당_및_인구만명당_AED설치대수" / "안전한동네_AED.xlsx", "04", "부분", "AED 자료이나 연도별·용도지역 면적 보정 확인 필요"),
    (TARGET / "04_주상공관리지역면적당_및_인구만명당_AED설치대수" / "자동심장충격기.csv", "04", "부분", "AED 시설자료이나 공식 연도별 지표는 아님"),
    (TARGET / "03_인구만명당_재난약자수" / "elderly_population_page_1.json", "03", "정확_부분", "재난약자 구성요소 중 65세 이상 인구 원본"),
    (TARGET / "03_인구만명당_재난약자수" / "elderly_population_page_2.json", "03", "정확_부분", "재난약자 구성요소 중 65세 이상 인구 원본"),
    (TARGET / "03_인구만명당_재난약자수" / "elderly_population_page_3.json", "03", "정확_부분", "재난약자 구성요소 중 65세 이상 인구 원본"),
    (TARGET / "03_인구만명당_재난약자수" / "서울시_고령자현황_전체.csv", "03", "정확_부분", "재난약자 구성요소 중 65세 이상 인구 추출본"),
]


def move_without_overwrite(src: Path, dst_dir: Path) -> Path | None:
    if not src.exists():
        return None
    dst = dst_dir / src.name
    if dst.exists():
        if src.resolve() == dst.resolve():
            return dst
        raise FileExistsError(f"덮어쓰기 방지를 위해 중단: {dst}")
    shutil.move(str(src), str(dst))
    return dst


for folder in FOLDERS.values():
    (TARGET / folder).mkdir(parents=True, exist_ok=True)

rows = []
for src, code, status, note in MOVES:
    dst = move_without_overwrite(src, TARGET / FOLDERS[code])
    rows.append({
        "파일명": src.name,
        "공식지표폴더": FOLDERS[code],
        "분류상태": status,
        "설명": note,
        "현재경로": str(dst.relative_to(ROOT)) if dst else "원본 없음",
    })

manifest = TARGET / "생활안전_raw_파일분류표.csv"
with manifest.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["파일명", "공식지표폴더", "분류상태", "설명", "현재경로"])
    writer.writeheader()
    writer.writerows(rows)

print(f"정리 대상: {TARGET}")
print(f"분류표: {manifest}")
print(f"이동/확인 파일: {sum(r['현재경로'] != '원본 없음' for r in rows)}")
