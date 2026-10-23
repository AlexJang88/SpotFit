"""선택 강조에 사용하는 구/동 경계와 동일한 서울 외곽을 생성한다.

사용: python scripts/build_seoul_outline.py
의존성: shapely (오프라인 경계 처리용, 프런트엔드 실행에는 불필요).
원본 좌표를 단순화하거나 반올림하지 않아 선택 경계와의 불일치를 방지한다.
"""
import json
from pathlib import Path

from shapely.geometry import mapping, shape
from shapely.ops import unary_union


def main():
    # scripts/ 아래에 배치하면 실행 위치에 관계없이 프로젝트 public 폴더를 찾는다.
    public = Path(__file__).resolve().parents[1] / "frontend" / "public"
    boundaries = {}
    for level, expected in [("dong", 427), ("gu", 25)]:
        collection = json.loads((public / f"seoul-{level}.geojson").read_text(encoding="utf-8-sig"))
        geometries = [shape(feature["geometry"]) for feature in collection["features"]]
        if len(geometries) != expected or any(not geom.is_valid for geom in geometries):
            raise ValueError(f"{level}: 경계 개수 또는 도형 유효성을 확인해주세요.")
        # dissolve: 맞닿은 내부 경계는 제거하고 전체 합집합의 외곽/구멍은 보존한다.
        boundaries[level] = unary_union(geometries)
    outline = boundaries["dong"]
    if not outline.is_valid or outline.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError("서울 외곽 도형이 유효하지 않습니다.")
    # 위경도 평면상의 극소 연산 오차만 허용한다. 실제 면적(km²) 검사는 아니다.
    mismatch = outline.symmetric_difference(boundaries["gu"]).area
    if mismatch > 1e-12:
        raise ValueError(f"구/동 외곽 불일치: {mismatch}")
    feature = {
        "type": "Feature",
        "properties": {
            "name": "서울특별시", "boundary_date": "2026-10-01",
            "source": "SGIS / vuski/admdongkor",
            "license": "KOGL Type 1 / CC BY 4.0",
            "processing": "Union of seoul-dong.geojson; no simplification",
        },
        "geometry": mapping(outline),
    }
    output = public / "seoul-boundary-20261001.geojson"
    output.write_text(json.dumps(feature, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    # 저장 후 다시 읽어 좌표 직렬화로 도형이 달라지지 않았는지도 확인한다.
    restored = shape(json.loads(output.read_text(encoding="utf-8"))["geometry"])
    assert restored.equals(outline)
    print(f"서울 외곽 저장 완료: {output.name}, 구/동 대칭차 면적={mismatch}")


if __name__ == "__main__":
    main()
