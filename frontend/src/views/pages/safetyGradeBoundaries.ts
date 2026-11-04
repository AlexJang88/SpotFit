import L from "leaflet";
import type { FeatureCollection } from "geojson";
import type { CrimeSafetyForecast, SafetyGrade } from "./crimeSafetyApi";

const GRADE_COLORS: Record<SafetyGrade, string> = {
  A: "#168b67",
  B: "#65b97f",
  C: "#e5bd45",
  D: "#e8893f",
  E: "#d95555",
};

/** 지역점수 예상등급에 따라 서울 자치구 경계를 반투명하게 채운다. */
export function addSafetyGradeBoundaries(
  map: L.Map,
  forecasts: CrimeSafetyForecast[],
): () => void {
  const controller = new AbortController();
  const layer = L.layerGroup().addTo(map);
  const byDistrict = new Map(forecasts.map(item => [item.district, item]));
  let active = true;

  void fetch(`${import.meta.env.BASE_URL}seoul-gu.geojson`, { signal: controller.signal })
    .then(response => {
      if (!response.ok) throw new Error("자치구 경계 파일 로딩 실패");
      return response.json() as Promise<FeatureCollection>;
    })
    .then(data => {
      if (!active) return;
      L.geoJSON(data, {
        style: feature => {
          const district = String(feature?.properties?.name ?? "");
          const forecast = byDistrict.get(district);
          const color = forecast ? GRADE_COLORS[forecast.predicted_grade] : "#8c9891";
          return {
            renderer: L.canvas(), color, weight: 1.5, opacity: 0.9,
            fill: true, fillColor: color, fillOpacity: forecast ? 0.48 : 0.14,
            interactive: false,
          };
        },
      }).addTo(layer);
    })
    .catch(error => {
      if (error instanceof DOMException && error.name === "AbortError") return;
    });

  return () => {
    active = false;
    controller.abort();
    layer.remove();
  };
}
