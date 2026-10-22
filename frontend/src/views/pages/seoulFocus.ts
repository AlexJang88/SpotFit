import L from "leaflet";
import type { Feature, Polygon, MultiPolygon } from "geojson";

/**
 * 서울 바깥에 어두운 마스크를 만들고 서울 외곽선을 표시한다.
 * 선택 강조와 동일한 2026-10-01 행정동의 합집합으로 외곽을 만들어 경계 어긋남을 방지한다.
 * scripts/build_seoul_outline.py에서 생성하며 행정동 집계/주소 판정용은 아니다.
 * 반환값은 HomePage effect의 cleanup에서 호출할 정리 함수다.
 * 구·동 선택 강조는 administrativeBoundaries.ts에서 별도로 관리한다.
 */
export function addSeoulFocus(map: L.Map): () => void {
  const controller = new AbortController();
  const group = L.layerGroup().addTo(map);
  // 기본 타일(200) 위, 지역 polygon(400) 아래. 클릭/드래그 이벤트는 지도에 전달한다.
  const pane = map.createPane("seoul-focus");
  pane.style.zIndex = "350";
  pane.style.pointerEvents = "none";
  const credit = '서울 외곽 2026-10-01 (행정동 합집합): <a href="https://sgis.kostat.go.kr">SGIS</a> (공공누리 1유형), <a href="https://github.com/vuski/admdongkor">vuski/admdongkor</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>';
  let active = true;
  // BASE_URL은 하위 경로 배포에서도 public 파일을 찾기 위한 Vite 기준 경로다.
  // active는 응답을 받은 뒤 이미 지도에서 제거된 작업인지 확인하는 수명 표시다.
  // 날짜별 파일명으로 기존 2018년 파일의 브라우저 캐시와 구분한다.
  void fetch(`${import.meta.env.BASE_URL}seoul-boundary-20261001.geojson`, { signal: controller.signal })
    .then(response => {
      if (!response.ok) throw new Error("서울 경계 파일을 불러오지 못했습니다.");
      return response.json() as Promise<Feature<Polygon | MultiPolygon>>;
    })
    .then(feature => {
      if (!active) return;
      const polygons = feature.geometry.type === "Polygon" ? [feature.geometry.coordinates] : feature.geometry.coordinates;
      // Polygon/MultiPolygon을 같은 배열 구조로 맞춰 반복 처리한다.
      // 각 polygon[0]은 외곽 ring, 이후 ring은 그 영역 안의 구멍이다.
      // GeoJSON은 경도/위도 순서, Leaflet은 위도/경도 순서이므로 명시적으로 변환한다.
      const convert = (ring: number[][]): L.LatLngTuple[] => ring.map(([lng, lat]) => [lat, lng]);
      const world: L.LatLngTuple[] = [[85, -180], [85, 180], [-85, 180], [-85, -180]];
      // Web Mercator에서 극점을 피하기 위해 위도 ±85까지만 마스크를 만든다.
      // evenodd 채우기 규칙은 외곽 사각형 안의 서울 ring을 투명한 구멍으로 처리한다.
      // 외부 사각형에서 서울 외곽 ring을 구멍으로 뚫어 지도 이동/줌에도 지리적 음영을 유지한다.
      L.polygon([world, ...polygons.map(p => convert(p[0]))], {
        pane: "seoul-focus", stroke: false, fillColor: "#24382f", fillOpacity: 0.48,
        fillRule: "evenodd", interactive: false,
      }).addTo(group);
      polygons.forEach(polygon => {
        // 서울 polygon 내부의 구멍은 서울 영역이 아니므로 다시 어둡게 채운다.
        polygon.slice(1).forEach(hole => L.polygon(convert(hole), {
          pane: "seoul-focus", stroke: false, fillColor: "#24382f", fillOpacity: .48, interactive: false,
        }).addTo(group));
      });
      L.geoJSON(feature, { pane: "seoul-focus", interactive: false,
        style: { color: "#1d7055", weight: 3, opacity: .9, fill: false },
      }).addTo(group);
      map.attributionControl.addAttribution(credit);
    })
    .catch(error => {
      if (active && error instanceof Error && error.name !== "AbortError") {
        map.attributionControl.addAttribution("서울 강조 경계 로딩 실패");
      }
    });
  return () => {
    // abort는 진행 중 네트워크를 취소하고 active=false는 이미 도착한 응답의 후속 처리를 막는다.
    // 지도 pane 자체는 HomePage의 instance.remove()가 최종 정리한다.
    active = false;
    controller.abort();
    group.remove();
    map.attributionControl.removeAttribution(credit);
  };
}
