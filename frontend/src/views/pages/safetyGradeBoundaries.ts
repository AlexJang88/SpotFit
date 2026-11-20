import L from "leaflet";
import type { FeatureCollection } from "geojson";
import type { CrimeSafetyForecast, SafetyGrade } from "./crimeSafetyApi";

const GRADE_OPACITY: Record<SafetyGrade, number> = { A: 0.75, B: 0.58, C: 0.42, D: 0.26, E: 0.12 };

/** 지역점수 예상등급에 따라 서울 자치구 경계를 반투명하게 채운다. */
export function addSafetyGradeBoundaries(
  map: L.Map,
  forecasts: Pick<CrimeSafetyForecast, "district" | "predicted_grade">[],
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
          const color = "#526e68";
          return {
            renderer: L.canvas(), color, weight: 1.5, opacity: 0.9,
            fill: true, fillColor: color, fillOpacity: forecast ? GRADE_OPACITY[forecast.predicted_grade] : 0.04,
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
