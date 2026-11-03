"""Locate and extract Seoul district workplace rows from the 2024 NHIS yearbook."""
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".tmp_pdf_extract"))
from pypdf import PdfReader  # noqa: E402


pdf = next((ROOT / "data" / "raw").rglob("2024_건강보험통계연보.pdf"))
output = ROOT / "data" / "interim" / "안전" / "자치구별_의료보장사업장수_2024.csv"
reader = PdfReader(pdf)
text = reader.pages[101].extract_text() or ""  # printed page 20, Table I-5
seoul_block = text.split("서 울", 1)[1].split("부 산", 1)[0]
districts = {
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구",
    "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구",
    "구로구", "금천구", "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구",
}
pattern = re.compile(
    r"^\s*([^\d\n]+?)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+"
    r"([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+[A-Za-z-]",
    re.MULTILINE,
)
rows = []
for match in pattern.finditer(seoul_block):
    gu = re.sub(r"\s+", "", match.group(1))
    if gu not in districts:
        continue
    nums = [int(match.group(i).replace(",", "")) for i in range(2, 9)]
    rows.append({
        "연도": 2024, "자치구": gu, "건강보험적용인구": nums[0],
        "의료보장사업장수": nums[1], "직장적용인구_계": nums[2],
        "직장가입자": nums[3], "직장피부양자": nums[4], "지역가입자": nums[5],
        "지역세대수": nums[6], "기준": "2024년 말, 실거주지 기준",
        "출처": "국민건강보험공단·건강보험심사평가원 2024 건강보험통계연보 제Ⅰ-5표",
    })
data = pd.DataFrame(rows).sort_values("자치구")
if len(data) != 25 or set(data["자치구"]) != districts:
    raise ValueError(f"district parse failed: rows={len(data)}, missing={districts-set(data['자치구'])}")
output.parent.mkdir(parents=True, exist_ok=True)
data.to_csv(output, index=False, encoding="utf-8-sig")
print(f"saved={output}")
print(f"rows={len(data)}, workplace_sum={data['의료보장사업장수'].sum()}")
