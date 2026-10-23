import L from "leaflet";
import type { FeatureCollection } from "geojson";

/**
 * 가격 버튼으로 선택한 구/행정동의 실제 경계 하나만 녹색으로 강조한다.
 * map: 이미 생성된 Leaflet 지도, level: 현재 단위, selected: 카드 선택 지역.
 * 반환 함수는 이전 강조와 출처 표시를 제거한다. 가격·환경 점수와 음영 농도는 무관하다.
 */
export function addAdministrativeBoundaries(map: L.Map, level: "시" | "구" | "행정동", selected: { name: string; parent: string } | null): () => void {
  if (level === "시" || !selected) return () => {};
  // 시 외곽 강조는 seoulFocus가 담당한다. 선택이 없으면 아무 레이어도 만들지 않는다.
  const controller = new AbortController();
  const layer = L.layerGroup().addTo(map);
  let active = true;
  const credit = '경계 2026-10-01: <a href="https://sgis.kostat.go.kr">SGIS</a> (공공누리 1유형), <a href="https://github.com/vuski/admdongkor">vuski/admdongkor</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>';
  const failure = "행정 경계 로딩 실패 · 새로고침해주세요";
  // Abort와 active 검사를 함께 사용해 빠르게 단위를 바꿔도 이전 응답이 남지 않게 한다.
  void fetch(`${import.meta.env.BASE_URL}seoul-${level === "구" ? "gu" : "dong"}.geojson`, { signal: controller.signal })
    .then(response => {
      // fetch는 HTTP 404/500만으로 reject되지 않으므로 상태를 직접 확인한다.
      // 타입 단언은 런타임 데이터 검증이 아니다. 현재는 검토된 로컬 GeoJSON을 전제로 한다.
      if (!response.ok) throw new Error("경계 파일 로딩 실패");
      return response.json() as Promise<FeatureCollection>;
    })
    .then(data => {
      if (!active) return;
      // 동명 중복을 피하기 위해 서울·상위 자치구·동 이름 전체가 일치하는 경계만 강조한다.
      const matches = data.features.filter(feature => level === "구"
        ? feature.properties?.name === selected.name
        : feature.properties?.adm_nm === `서울특별시 ${selected.parent} ${selected.name}`);
      if (matches.length !== 1) throw new Error("선택 지역 경계 매핑 실패");
      // 구 파일은 name, 동 파일은 전체 주소형 adm_nm을 사용한다.
      // 이름 변경/중복으로 0개 또는 여러 개가 잡히면 임의의 첫 영역을 선택하지 않는다.
      // 녹색은 가격이 아닌 선택 상태다. 자동 줌은 하지 않아 지역 단계 변경을 방지한다.
      const selectedBoundary: FeatureCollection = { type: "FeatureCollection", features: matches };
      L.geoJSON(selectedBoundary, {
        style: { renderer: L.canvas(), color: "#18765a", weight: 3, opacity: 1, fill: true, fillColor: "#38a77e", fillOpacity: .3 },
        onEachFeature: (feature, boundary) => {
          // 마우스가 경계 위에 있을 때 지역명을 따라다니는 툴팁을 표시한다.
          const name = document.createElement("span");
          name.textContent = String(feature.properties?.name ?? feature.properties?.adm_nm ?? "");
          boundary.bindTooltip(name, { sticky: true });
        },
      }).addTo(layer);
      map.attributionControl.addAttribution(credit);
    })
    .catch(() => { if (active) map.attributionControl.addAttribution(failure); });
  return () => {
    // 빠른 버튼 전환 시 이전 fetch가 늦게 끝나도 이전 지역이 다시 그려지지 않게 한다.
    active = false;
    controller.abort();
    layer.remove();
    map.attributionControl.removeAttribution(credit);
    map.attributionControl.removeAttribution(failure);
  };
}
