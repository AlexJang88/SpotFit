import L from "leaflet";
import type { Feature, FeatureCollection, Geometry } from "geojson";

type Level = "시" | "구" | "행정동";
type MapRegion = { name: string; parent: string; level: Level };

/** Render every boundary at the current level. Labels exist only as hover tooltips. */
export function addAdministrativeBoundaries(
  map: L.Map,
  level: Level,
  selected: MapRegion | null,
  regions: MapRegion[],
  labelFor: (region: MapRegion) => string,
  onSelect: (region: MapRegion) => void,
): () => void {
  const controller = new AbortController();
  const layer = L.layerGroup().addTo(map);
  let active = true;
  const filename = level === "시"
    ? "seoul-boundary-20261001.geojson"
    : `seoul-${level === "구" ? "gu" : "dong"}.geojson`;
  const credit = '행정경계: <a href="https://sgis.kostat.go.kr">SGIS</a>, <a href="https://github.com/vuski/admdongkor">vuski/admdongkor</a>';

  const regionFor = (feature: Feature<Geometry>): MapRegion | undefined => {
    if (level === "시") return regions.find(region => region.level === "시");
    if (level === "구") {
      const name = String(feature.properties?.name ?? feature.properties?.sggnm ?? "");
      return regions.find(region => region.level === "구" && region.name === name);
    }
    const fullName = String(feature.properties?.adm_nm ?? "");
    const district = String(feature.properties?.sggnm ?? fullName.split(" ")[1] ?? "");
    const name = fullName.split(" ").at(-1) ?? "";
    return regions.find(region => region.level === "행정동" && region.parent === district && region.name === name);
  };

  void fetch(`${import.meta.env.BASE_URL}${filename}`, { signal: controller.signal })
    .then(response => {
      if (!response.ok) throw new Error("행정 경계 로딩 실패");
      return response.json() as Promise<FeatureCollection | Feature<Geometry>>;
    })
    .then(data => {
      if (!active) return;
      L.geoJSON(data, {
        style: feature => {
          const region = feature ? regionFor(feature as Feature<Geometry>) : undefined;
          const isSelected = Boolean(region && selected && region.level === selected.level
            && region.name === selected.name && region.parent === selected.parent);
          return {
            color: isSelected ? "#18765a" : "#789087",
            weight: isSelected ? 3 : 1,
            opacity: isSelected ? 1 : 0.45,
            fill: true,
            fillColor: isSelected ? "#38a77e" : "#ffffff",
            fillOpacity: isSelected ? 0.3 : 0.01,
          };
        },
        onEachFeature: (feature, boundary) => {
          const region = regionFor(feature as Feature<Geometry>);
          if (!region) return;
          const tooltip = document.createElement("span");
          tooltip.textContent = `${region.name} · ${labelFor(region)}`;
          boundary.bindTooltip(tooltip, { sticky: true, direction: "top", opacity: 0.96 });
          boundary.on({
            mouseover: event => {
              const target = event.target as L.Path;
              target.setStyle({ weight: 3, color: "#18765a", fillColor: "#38a77e", fillOpacity: 0.18 });
              target.bringToFront();
            },
            mouseout: event => {
              const target = event.target as L.Path;
              const isSelected = selected && region.level === selected.level
                && region.name === selected.name && region.parent === selected.parent;
              target.setStyle({
                color: isSelected ? "#18765a" : "#789087",
                weight: isSelected ? 3 : 1,
                fillColor: isSelected ? "#38a77e" : "#ffffff",
                fillOpacity: isSelected ? 0.3 : 0.01,
              });
            },
            click: () => onSelect(region),
          });
        },
      }).addTo(layer);
      map.attributionControl.addAttribution(credit);
    })
    .catch(error => {
      if (active && error instanceof Error && error.name !== "AbortError") {
        map.attributionControl.addAttribution("행정 경계 로딩 실패");
      }
    });

  return () => {
    active = false;
    controller.abort();
    layer.remove();
    map.attributionControl.removeAttribution(credit);
    map.attributionControl.removeAttribution("행정 경계 로딩 실패");
  };
}
