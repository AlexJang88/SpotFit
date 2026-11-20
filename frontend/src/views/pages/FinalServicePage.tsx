import { useEffect, useMemo, useRef, useState } from "react";
import L from "leaflet";
import type { FeatureCollection } from "geojson";
import { addSeoulFocus } from "./seoulFocus";
import { addAdministrativeBoundaries } from "./administrativeBoundaries";
import { addSafetyGradeBoundaries } from "./safetyGradeBoundaries";
import { fetchAdminBoundaryStatus, type AdminBoundaryStatus } from "./adminBoundaryApi";
import { BudgetRange, ControlIcon, budgetLabel, type Panel } from "./DiscoveryControls";
import { amountFor, fetchCatalog, fetchRecommendations, normalizeDong, transactionLabel, M2_PER_PYEONG,
  fetchAreaPresets, type AreaPresets, type Catalog, type DistrictScore, type Neighborhood, type Recommendations, type Transaction } from "./neighborhoodApi";
import "leaflet/dist/leaflet.css";
import "./final-service.css";
import { userErrorMessage } from "../../services/apiClient";
import { formatMoney } from "../../services/formatMoney";

type Level = "시" | "구" | "행정동";
type Region = { id: string; name: string; parent: string; level: Level; lat: number; lng: number };
const SEOUL: Region = { id: "seoul", name: "서울특별시", parent: "", level: "시", lat: 37.5665, lng: 126.978 };
const ZOOM: Record<Level, number> = { 시: 10, 구: 11, 행정동: 13 };
const money = (n: number | null) => n === null ? "가격 정보 없음" : formatMoney(n);
const points = (n: number) => n.toLocaleString("ko-KR", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const districtOf = (r: Region) => r.level === "구" ? r.name : r.parent;
const SCORE_WEIGHT_STORAGE = "spotfit.crime-weight-percent.v1";
const DEFAULT_BUDGET_MANWON = "100000";
type TrendPoint = { year: number; value: number; kind: "observed" | "forecast" };

function TrendChart({ title, points, unit, note, connect = true }: { title: string; points: TrendPoint[]; unit: string; note: string; connect?: boolean }) {
  const valid = points.filter(point => Number.isFinite(point.value));
  if (!valid.length) return <p className="field-hint">표시할 추세 자료가 없습니다.</p>;
  const values = valid.map(point => point.value);
  const isMoney = unit.startsWith("만원");
  const valueLabel = (value: number) => isMoney ? formatMoney(value) : Math.round(value).toLocaleString("ko-KR");
  const valueUnit = isMoney ? unit.replace("만원", "") : unit;
  const yearLabelCount = Math.min(valid.length, isMoney ? 4 : 5);
  const yearLabelIndices = new Set(Array.from({ length: yearLabelCount }, (_, index) =>
    yearLabelCount === 1 ? 0 : Math.round(index * (valid.length - 1) / (yearLabelCount - 1))));
  const low = Math.min(...values), high = Math.max(...values);
  const padding = Math.max((high - low) * .2, high * .02, 1);
  const min = low - padding, max = high + padding;
  const xy = (point: TrendPoint, index: number) => ({
    x: valid.length === 1 ? 150 : 30 + index * 240 / (valid.length - 1),
    y: 116 - (point.value - min) / (max - min) * 88,
  });
  return <section className="detail-trend" aria-label={title}>
    <h3>{title}</h3>
    <svg viewBox="0 0 300 155" role="img" aria-label={`${title}: ${valid.map(p => `${p.year}년 ${valueLabel(p.value)}${valueUnit} ${p.kind === "observed" ? "대표값" : "예측"}`).join(", ")}`}>
      <line x1="30" y1="116" x2="270" y2="116" className="trend-axis" />
      {connect && valid.slice(1).map((point, index) => { const a = xy(valid[index], index), b = xy(point, index + 1); return <line key={point.year} x1={a.x} y1={a.y} x2={b.x} y2={b.y} className="trend-line" strokeDasharray={point.kind === "forecast" ? "5 4" : undefined} />; })}
      {valid.map((point, index) => { const pos = xy(point, index); const showValue = index === 0 || index === valid.length - 1 || (!isMoney && point.year === 2026); return <g key={point.year}><title>{point.year}년 {valueLabel(point.value)}{valueUnit}</title><circle cx={pos.x} cy={pos.y} r="3.5" className={point.kind === "observed" ? "trend-observed" : "trend-forecast"}/>{showValue && <text x={pos.x} y={Math.max(13, pos.y - 9)} textAnchor={isMoney && valid.length > 1 ? (index === 0 ? "start" : "end") : "middle"}>{valueLabel(point.value)}</text>}{yearLabelIndices.has(index) && <text x={pos.x} y="137" textAnchor="middle">{point.year}년</text>}</g>; })}
    </svg>
    <small>{note}</small>
  </section>;
}
function savedCrimePercent() {
  try {
    const saved = localStorage.getItem(SCORE_WEIGHT_STORAGE);
    const value = saved === null || saved.trim() === "" ? NaN : Number(saved);
    return Number.isInteger(value) && value >= 0 && value <= 100 ? value : 50;
  } catch { return 50; }
}

export function FinalServicePage() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [reload, setReload] = useState(0);
  const [horizon, setHorizon] = useState(2);
  const [crimePercent, setCrimePercent] = useState(savedCrimePercent);
  const [transaction, setTransaction] = useState<Transaction>("sale");
  const [budget, setBudget] = useState(DEFAULT_BUDGET_MANWON);
  const [minBudget, setMinBudget] = useState("0");
  const [deposit, setDeposit] = useState("5000");
  const [areaUnit, setAreaUnit] = useState<"m2" | "pyeong">("m2");
  const [activePanel, setActivePanel] = useState<Panel | null>(null);
  const [areaPresets, setAreaPresets] = useState<AreaPresets | null>(null);
  const [areaError, setAreaError] = useState("");
  const toolbar = useRef<HTMLDivElement>(null);
  const [area, setArea] = useState("");
  const [recommendations, setRecommendations] = useState<Recommendations | null>(null);
  const [recommending, setRecommending] = useState(false);
  const [recommendError, setRecommendError] = useState("");
  const [recommendationReload, setRecommendationReload] = useState(0);
  const [dongGeometry, setDongGeometry] = useState<FeatureCollection | null>(null);
  const [regions, setRegions] = useState<Region[]>([SEOUL]);
  const [boundaryStatus, setBoundaryStatus] = useState<AdminBoundaryStatus | null>(null);
  const [boundaryError, setBoundaryError] = useState("");
  const [query, setQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [selected, setSelected] = useState<Region | null>(null);
  const [level, setLevel] = useState<Level>("구");
  const [tileError, setTileError] = useState(false);
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<L.Map | null>(null);
  const detail = useRef<HTMLDialogElement>(null);
  const key = import.meta.env.VITE_VWORLD_API_KEY as string | undefined;
  const parsedArea = area.trim() ? Number(area) : null;
  const areaValid = parsedArea === null || (Number.isFinite(parsedArea) && parsedArea >= 1 && parsedArea <= 1000);
  const areaValue = areaValid ? (parsedArea ?? 84) : null;
  const depositValue = Number(deposit);
  const depositValid = deposit.trim() !== "" && Number.isFinite(depositValue) && depositValue >= 0 && depositValue <= 1_000_000_000;
  const monthly = transaction === "wolse";
  const unitLabel = areaUnit === "m2" ? "㎡" : "평";
  const shownArea = area.trim() ? (areaUnit === "m2" ? area : String(Number((Number(area) / M2_PER_PYEONG).toFixed(4)))) : "";
  const monthlyArea = monthly ? areaValue : null;
  const monthlyDeposit = monthly ? depositValue : undefined;
  const catalogMatches = catalog?.horizon === horizon && catalog.transaction === transaction &&
    Math.abs(catalog.weights.crime * 100 - crimePercent) < 1e-6 &&
    (!monthly || (depositValid && catalog.deposit_manwon === depositValue && catalog.area_m2 === areaValue));
  const districts = useMemo(() => new Map(catalog?.districts.map(row => [row.district, row]) ?? []), [catalog]);
  const neighborhoodIndex = useMemo(() => {
    const index = new Map<string, Neighborhood | null>();
    for (const row of catalog?.neighborhoods ?? []) {
      const id = `${row.district}|${normalizeDong(row.neighborhood)}`;
      index.set(id, index.has(id) ? null : row);
    }
    return index;
  }, [catalog]);
  const neighborhoodFor = (r: Region) => neighborhoodIndex.get(`${r.parent}|${normalizeDong(r.name)}`) ?? undefined;
  const priceFor = (r: Region): number | null => {
    if (!areaValid || !catalogMatches || r.level === "시") return null;
    if (r.level === "행정동") return amountFor(neighborhoodFor(r), transaction, areaValue);
    const values = (catalog?.neighborhoods ?? []).filter(row => row.district === r.name)
      .map(row => amountFor(row, transaction, areaValue)).filter((v): v is number => v !== null && Number.isFinite(v));
    return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
  };
  const visible = useMemo(() => regions.filter(r => r.level === level), [regions, level]);
  const found = useMemo(() => regions.filter(r => r.level !== "시" && normalizeDong(`${r.parent}${r.name}`).includes(normalizeDong(query))), [query, regions]);
  const selectedScore = selected ? districts.get(districtOf(selected)) : undefined;
  const selectedPrice = selected ? priceFor(selected) : null;
  const selectedProperty = selected?.level === "행정동" ? neighborhoodFor(selected) : undefined;
  const priceSeries = selected && !monthly && catalogMatches ? (() => {
    const rows = selected.level === "행정동" ? (selectedProperty ? [selectedProperty] : []) :
      (catalog?.neighborhoods ?? []).filter(row => row.district === selected.name);
    const field = transaction === "sale" ? "apart_sale_price_series" : "apart_rent_price_series";
    const years = [...new Set(rows.flatMap(row => (row[field] ?? []).map(point => point.year)))].sort();
    return years.map(year => {
      const entries = rows.flatMap(row => (row[field] ?? []).filter(point => point.year === year && point.per_pyeong_manwon !== null));
      return entries.length ? { year, value: entries.reduce((sum, point) => sum + (point.per_pyeong_manwon ?? 0), 0) / entries.length,
        kind: entries[0].kind } : null;
    }).filter((point): point is NonNullable<typeof point> => point !== null);
  })() : [];
  const districtPeerPrices = selected?.level === "행정동" && catalogMatches ? (catalog?.neighborhoods ?? [])
    .filter(row => row.district === selected.parent).map(row => ({ id: row.id, price: amountFor(row, transaction, areaValue) }))
    .filter((row): row is { id: string; price: number } => row.price !== null).sort((a, b) => a.price - b.price) : [];
  const districtPriceRank = selectedProperty ? districtPeerPrices.findIndex(row => row.id === selectedProperty.id) + 1 : 0;
  const scoreSeries = selectedScore?.score_series?.map(point => ({ year: point.year, value: point.score, kind: point.kind })) ?? [];
  const historicalReferenceSeries: TrendPoint[] = selectedScore?.historical_reference_series?.map(point =>
    ({ year: point.year, value: point.score, kind: "observed" })) ?? [];
  const propertyBaseYear = catalog?.property_models[transaction]?.data_year ?? 2026;
  const monthlyHistorySeries = selected && monthly && catalogMatches ? (() => {
    const rows = selected.level === "행정동" ? (selectedProperty ? [selectedProperty] : []) :
      (catalog?.neighborhoods ?? []).filter(row => row.district === selected.name);
    const years = [...new Set(rows.flatMap(row => (row.monthly_rent_history ?? []).map(point => point.year)))].sort();
    return years.map(year => {
      const values = rows.flatMap(row => (row.monthly_rent_history ?? []).filter(point => point.year === year));
      return values.length ? { year, value: values.reduce((sum, point) => sum + point.monthly_rent_manwon, 0) / values.length,
        kind: "observed" as const } : null;
    }).filter((point): point is NonNullable<typeof point> => point !== null);
  })() : [];
  const rankedRegions = [...visible].sort((a, b) => (districts.get(districtOf(a))?.rank ?? 999)
    - (districts.get(districtOf(b))?.rank ?? 999) || a.name.localeCompare(b.name, "ko"));

  useEffect(() => {
    try { localStorage.setItem(SCORE_WEIGHT_STORAGE, String(crimePercent)); }
    catch { /* Private mode or disabled storage: retain the selection for this visit. */ }
  }, [crimePercent]);

  useEffect(() => {
    const controller = new AbortController();
    setAreaPresets(null); setAreaError("");
    fetchAreaPresets(transaction, controller.signal).then(data => {
      if (!controller.signal.aborted) setAreaPresets(data);
    }).catch(error => {
      if (!controller.signal.aborted) setAreaError(`${userErrorMessage(error, "면적 선택지를 불러오지 못했습니다.")} 직접 입력할 수 있습니다.`);
    });
    return () => controller.abort();
  }, [transaction]);

  useEffect(() => {
    if (!activePanel) return;
    if (activePanel === "search") toolbar.current?.querySelector<HTMLInputElement>("#region-search")?.focus();
    const outside = (e: PointerEvent) => { if (!toolbar.current?.contains(e.target as Node)) setActivePanel(null); };
    const escape = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      toolbar.current?.querySelector<HTMLButtonElement>(`[data-panel="${activePanel}"]`)?.focus();
      setActivePanel(null);
    };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", escape);
    return () => { document.removeEventListener("pointerdown", outside); document.removeEventListener("keydown", escape); };
  }, [activePanel]);

  useEffect(() => {
    const controller = new AbortController();
    let retryTimer: number | undefined;
    setLoading(true); setError(""); setCatalog(null);
    if (monthly && (!depositValid || !areaValid)) {
      setError("보증금(0 이상)과 전용면적(1~1,000㎡)을 확인해 주세요."); setLoading(false);
      return () => controller.abort();
    }
    const timer = window.setTimeout(() => fetchCatalog(horizon, controller.signal, transaction, monthlyDeposit, monthlyArea, crimePercent).then(data => {
      if (!controller.signal.aborted) setCatalog(data);
    }).catch(e => {
      if (!controller.signal.aborted) {
        setError(`${userErrorMessage(e, "동네 정보를 불러오지 못했습니다.")} · 모델 복구 상태를 확인하며 10초 후 다시 시도합니다.`);
        retryTimer = window.setTimeout(() => setReload(value => value + 1), 10_000);
      }
    })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); }), monthly ? 450 : 300);
    return () => { window.clearTimeout(timer); controller.abort(); if (retryTimer !== undefined) window.clearTimeout(retryTimer); };
  }, [horizon, reload, transaction, monthlyDeposit, monthlyArea, depositValid, areaValid, crimePercent]);

  useEffect(() => {
    setRecommendations(null); setRecommendError(""); setRecommending(false);
    const value = Number(budget);
    const minimum = Number(minBudget);
    if (!minBudget.trim() || !Number.isFinite(minimum) || minimum < 0 || minimum > value ||
        !Number.isFinite(value) || value < 1 || value > 1_000_000_000 || !areaValid) {
      setRecommendError("예산 범위(0 ≤ 하한 ≤ 상한)와 면적(1~1,000㎡)을 확인해 주세요.");
      return;
    }
    if (!catalogMatches || loading || error) return;
    const controller = new AbortController();
    setRecommending(true);
    const timer = window.setTimeout(() => {
      fetchRecommendations(value, transaction, horizon, areaValue, controller.signal, minimum, monthlyDeposit, crimePercent)
        .then(data => { if (!controller.signal.aborted) setRecommendations(data); })
        .catch(e => { if (!controller.signal.aborted) setRecommendError(userErrorMessage(e, "추천 결과를 불러오지 못했습니다.")); })
        .finally(() => { if (!controller.signal.aborted) setRecommending(false); });
    }, 450);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [budget, minBudget, area, transaction, horizon, catalog, loading, error, recommendationReload, deposit, catalogMatches, crimePercent]);

  useEffect(() => {
    const controller = new AbortController();
    fetchAdminBoundaryStatus(controller.signal).then(setBoundaryStatus).catch(error => {
      if (!controller.signal.aborted) setBoundaryError(userErrorMessage(error, "행정동 기준정보를 불러오지 못했습니다."));
    });
    Promise.all(["seoul-gu.geojson", "seoul-dong.geojson"].map(async file => {
      const r = await fetch(`${import.meta.env.BASE_URL}${file}`, { signal: controller.signal });
      if (!r.ok) throw new Error("행정구역 목록을 불러오지 못했습니다.");
      return r.json() as Promise<FeatureCollection>;
    })).then(([gu, dong]) => {
      if (controller.signal.aborted) return;
      setDongGeometry(dong);
      const items: Region[] = [SEOUL];
      for (const [data, regionLevel] of [[gu, "구"], [dong, "행정동"]] as const) {
        for (const feature of data.features) {
          const full = String(feature.properties?.adm_nm ?? "");
          const name = regionLevel === "구" ? String(feature.properties?.name ?? feature.properties?.sggnm ?? "") : full.split(" ").at(-1) ?? "";
          const parent = regionLevel === "구" ? "서울특별시" : String(feature.properties?.sggnm ?? full.split(" ")[1] ?? "");
          if (!name || !parent) continue;
          const center = L.geoJSON(feature).getBounds().getCenter();
          items.push({ id: `${regionLevel}|${parent}|${name}`, name, parent, level: regionLevel, lat: center.lat, lng: center.lng });
        }
      }
      setRegions(items);
    }).catch(e => { if (!controller.signal.aborted) setBoundaryError(userErrorMessage(e, "행정구역 목록을 불러오지 못했습니다.")); });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!container.current) return;
    const instance = L.map(container.current, { zoomControl: false, minZoom: 8, maxZoom: 18, zoomSnap: 0.25 }).setView([SEOUL.lat, SEOUL.lng], 11);
    map.current = instance;
    const removeFocus = addSeoulFocus(instance);
    L.control.zoom({ position: "bottomright" }).addTo(instance);
    if (key) {
      const tiles = L.tileLayer(`https://api.vworld.kr/req/wmts/1.0.0/${encodeURIComponent(key)}/Base/{z}/{y}/{x}.png`, {
        attribution: '© <a href="https://www.vworld.kr" target="_blank" rel="noreferrer">VWorld</a>', maxZoom: 18,
      }).addTo(instance);
      tiles.on("tileerror", () => setTileError(true));
    }
    const observer = new ResizeObserver(() => instance.invalidateSize());
    observer.observe(container.current);
    return () => { observer.disconnect(); removeFocus(); instance.remove(); map.current = null; };
  }, [key]);

  useEffect(() => {
    const instance = map.current;
    if (!instance) return;
    const onZoom = () => { setLevel(instance.getZoom() < 10 ? "시" : instance.getZoom() < 13 ? "구" : "행정동"); };
    instance.on("zoomend", onZoom); onZoom();
    return () => { instance.off("zoomend", onZoom); };
  }, []);

  useEffect(() => {
    if (dongGeometry) showAll();
  }, [dongGeometry]);

  useEffect(() => {
    if (!map.current) return;
    return addAdministrativeBoundaries(map.current, level, selected?.level === level ? selected : null, visible,
      r => {
        const match = visible.find(v => v.name === r.name && v.parent === r.parent);
        if (!match) return "";
        return `${money(priceFor(match))}${monthly ? "/월" : ""}`;
      }, r => { const match = visible.find(v => v.name === r.name && v.parent === r.parent); if (match) setSelected(match); });
  }, [level, selected, visible, catalog, transaction, area, deposit, horizon]);

  useEffect(() => {
    if (!map.current || !catalog) return;
    return addSafetyGradeBoundaries(map.current, catalog.districts.map(row => ({ district: row.district, predicted_grade: row.grade })));
  }, [catalog]);

  useEffect(() => {
    if (!map.current || !recommendations) return;
    const group = L.layerGroup().addTo(map.current);
    if (dongGeometry) {
      const ids = new Set(recommendations.recommendations.map(r => `${r.district}|${normalizeDong(r.neighborhood)}`));
      L.geoJSON(dongGeometry, {
        interactive: false,
        filter: feature => {
          const full = String(feature.properties?.adm_nm ?? "").split(" ");
          return ids.has(`${feature.properties?.sggnm ?? full[1]}|${normalizeDong(full.at(-1) ?? "")}`);
        },
        style: { color: "#354f49", weight: 3, opacity: 1, fillColor: "#526e68", fillOpacity: 0.22, dashArray: "5 4" },
      }).addTo(group);
    }
    for (const row of recommendations.recommendations) {
      const region = regions.find(r => r.level === "행정동" && r.parent === row.district && normalizeDong(r.name) === normalizeDong(row.neighborhood));
      if (!region) continue;
      const badge = document.createElement("span"); badge.textContent = String(row.recommendation_rank);
      const marker = L.marker([region.lat, region.lng], { title: `${row.recommendation_rank}위 ${row.district} ${row.neighborhood}`, zIndexOffset: 1000, icon: L.divIcon({ className: "recommend-marker", html: badge, iconSize: [34, 34] }) });
      const tooltip = document.createElement("span"); tooltip.textContent = `${row.neighborhood} · ${money(row.estimated_amount_manwon)}${monthly ? "/월" : ""}`;
      marker.bindTooltip(tooltip, { permanent: map.current.getSize().x > 760, direction: "bottom", className: "recommend-tooltip", offset: [0, 14] }).on("click", () => choose(region)).addTo(group);
    }
    return () => { group.remove(); };
  }, [recommendations, regions, dongGeometry, monthly]);

  function showAll() {
    const instance = map.current;
    if (!instance || !dongGeometry) return;
    setSelected(null);
    const mobile = instance.getSize().x <= 760;
    const toolbar = document.querySelector(".map-toolbar")?.getBoundingClientRect();
    instance.fitBounds(L.geoJSON(dongGeometry).getBounds(), {
      paddingTopLeft: [mobile ? 50 : 80, (toolbar?.height ?? 100) + 40],
      paddingBottomRight: [mobile ? 50 : 80, 60],
      maxZoom: 11,
    });
  }

  function choose(r: Region) {
    setSelected(r); setLevel(r.level); setSearchOpen(false); setQuery(""); setActivePanel(null);
    map.current?.flyTo([r.lat, r.lng], ZOOM[r.level]);
  }

  function changeTransaction(next: Transaction) {
    if ((next === "wolse") !== monthly) { setBudget(next === "wolse" ? "200" : DEFAULT_BUDGET_MANWON); setMinBudget("0"); }
    setRecommendations(null); setTransaction(next);
  }

  function changeCrimePercent(value: number) {
    if (value === crimePercent) return;
    setCatalog(null); setRecommendations(null);
    setCrimePercent(Math.round(Math.max(0, Math.min(100, value))));
  }

  function monthlySummary() {
    const quote = catalogMatches ? selectedProperty?.wolse : undefined;
    if (!monthly) return null;
    return <div className="monthly-summary"><p>보증금 <b>{money(depositValid ? depositValue : null)}</b> 기준</p>
      {quote?.available ? <><p>예상 월세 범위 <b>{money(quote.p25_manwon ?? null)}–{money(quote.p75_manwon ?? null)}</b></p>
      <p>대표 연 전환율 {quote.conversion_rate_pct}% · {quote.area_band} · 건물 {quote.buildings_used}곳</p>
      <small>건물별 월세 = 예측 전환율 × (기준 전세가 − 보증금) ÷ 12. 표시 범위는 건물별 결과의 25~75% 구간입니다.</small></> :
      <p className="field-hint">{quote?.reason ?? (selected?.level === "구" ? "하위 행정동 예상 월세의 평균입니다. 동을 선택하면 전환율과 범위를 확인할 수 있습니다." : "이 동·면적대의 월세 예측 자료가 부족합니다.")}</p>}
    </div>;
  }

  function scoreSummary(score: DistrictScore) {
    return <div className="score-breakdown">
      <div><span>아동범죄노출</span><strong>{points(score.crime_score)}<small>점</small></strong><small>Feature 가중합 · {score.crime_prediction_year}년 · {crimePercent}% 반영</small></div>
      <div><span>교통안전</span><strong>{points(score.traffic_score)}<small>점</small></strong><small>Feature 가중합 · {score.traffic_prediction_year}년 · {100 - crimePercent}% 반영</small></div>
    </div>;
  }

  return <main className="spotfit-final">
    <div className="service-workspace">
      <section className="service-map" aria-label="서울 동네 추천 지도"><div className="map-canvas" ref={container}/>
        <div className="map-toolbar" ref={toolbar}>
          <div className="discovery-bar">
            <a className="brand" href="/" aria-label="SpotFit 홈"><span className="brand-icon" aria-hidden="true">⌂</span><span>SpotFit</span></a>
            <nav className="discovery-actions" aria-label="지도 검색 및 조건">
              {([
                ["search", "지역 검색", "검색"], ["transaction", "거래 유형", transactionLabel[transaction]],
                ["budget", "예산 범위", monthly ? "보증금·월세" : "예산"], ["area", "전용면적", area ? `${Number(shownArea).toLocaleString("ko-KR", {maximumFractionDigits: 2})}${unitLabel}` : "면적"],
                ["year", "예측연도", `${2025 + horizon}`], ["weights", "점수 반영 비율 설정", "비율"], ["info", "지역·이용 안내", "안내"],
              ] as [Panel, string, string][]).map(([panel, label, caption]) => <button key={panel} className={`control-button ${activePanel === panel ? "active" : ""}`} data-panel={panel} aria-label={label} title={label} aria-expanded={activePanel === panel} aria-controls={activePanel === panel ? "discovery-panel" : undefined} onClick={() => setActivePanel(p => p === panel ? null : panel)}><ControlIcon name={panel}/><span>{caption}</span></button>)}
              <button className="control-button overview-button" aria-label="서울 전체 보기" title="서울 전체 보기" onClick={() => { setActivePanel(null); showAll(); }}><ControlIcon name="locate"/><span>전체</span></button>
            </nav>
          </div>
          {activePanel && <section id="discovery-panel" className="discovery-panel" aria-label={{ search: "지역 검색 설정", transaction: "거래 유형 설정", budget: "예산 범위 설정", area: "전용면적 설정", year: "예측연도 설정", weights: "점수 반영 비율 설정", info: "지역 이용 안내" }[activePanel]}>
            <div className="panel-heading"><h2>{{ search: "어느 동네가 궁금하세요?", transaction: "거래 유형", budget: "나의 예산 범위", area: "어떤 면적을 찾으세요?", year: "범죄노출 예측연도", weights: "어떤 안전을 더 중요하게 보세요?", info: "지역·이용 안내" }[activePanel]}</h2><button aria-label="설정 닫기" onClick={() => { toolbar.current?.querySelector<HTMLButtonElement>(`[data-panel="${activePanel}"]`)?.focus(); setActivePanel(null); }}>×</button></div>
            {activePanel === "search" && <section className="search-section">
                <label className="sr-only" htmlFor="region-search">자치구 또는 행정동 검색</label>
                <div className="input-unit search-input"><span aria-hidden="true">⌕</span><input id="region-search" autoComplete="off" value={query} onFocus={() => setSearchOpen(true)} onChange={e => { setQuery(e.target.value); setSearchOpen(true); }} onKeyDown={e => { if (e.key === "Escape") setSearchOpen(false); if (e.key === "Enter" && found[0]) choose(found[0]); }} placeholder="어느 동네가 궁금하세요? 예: 역삼1동, 강남구"/><button aria-label="검색어 지우기" onClick={() => { setQuery(""); setSearchOpen(false); }}>×</button></div>
                {searchOpen && <div className="search-results"><div className="search-results-heading"><small>검색 결과 {found.length}개</small><button onClick={() => setSearchOpen(false)} aria-label="검색 결과 닫기">닫기</button></div>{found.map(r => <button key={r.id} onClick={() => choose(r)}><span>{r.name}<small>{r.parent}</small></span><small>{r.level}</small></button>)}{!found.length && <p>검색된 지역이 없습니다.</p>}</div>}

            </section>}
            {activePanel === "transaction" && <><div className="segment" aria-label="거래유형">{(["sale", "rent", "wolse"] as Transaction[]).map(t => <button type="button" key={t} aria-pressed={transaction === t} className={transaction === t ? "active" : ""} onClick={() => changeTransaction(t)}>{transactionLabel[t]}</button>)}</div><p className="field-hint">선택한 거래 유형으로 가격과 추천이 자동 변경됩니다.</p>{monthly && <button className="secondary-button" onClick={() => setActivePanel("budget")}>보증금·월세 예산 설정 →</button>}</>}
            {activePanel === "budget" && <>{monthly && <div className="filter-field"><label htmlFor="deposit">입력 보증금</label><div className="input-unit"><input id="deposit" type="number" min="0" max="1000000000" step="any" value={deposit} onChange={e => setDeposit(e.target.value)}/><span>만원</span></div><p className="field-hint">{depositValid ? `보증금 ${money(depositValue)} 기준으로 ` : "보증금을 입력하면 "}아래 월세 범위에 맞는 동네를 추천합니다.</p><h3>월세 예산 · 월 기준</h3></div>}<BudgetRange lower={minBudget} upper={budget} onLower={setMinBudget} onUpper={setBudget} monthly={monthly}/></>}
            {activePanel === "area" && <>
              <div className="segment area-unit" aria-label="면적 입력 단위">{(["m2", "pyeong"] as const).map(unit => <button key={unit} type="button" aria-pressed={areaUnit === unit} className={areaUnit === unit ? "active" : ""} onClick={() => setAreaUnit(unit)}>{unit === "m2" ? "㎡" : "평"}</button>)}</div>
              <p className="field-hint">{areaPresets?.description ?? (areaError || "거래 면적을 확인하고 있습니다…")}</p>
              <div className="area-presets">{areaPresets?.options.map(option => {
                const squareMeters = `${option.area_m2.toLocaleString("ko-KR", { maximumFractionDigits: 2 })}㎡`;
                const pyeong = `${Math.round(option.area_m2 / M2_PER_PYEONG)}평`;
                return <button key={option.area_m2} aria-pressed={area === String(option.area_m2)} className={area === String(option.area_m2) ? "active" : ""} onClick={() => setArea(String(option.area_m2))}>
                  <strong>{areaUnit === "m2" ? squareMeters : pyeong}</strong>
                  <span>{areaUnit === "m2" ? `약 ${pyeong}` : `전용 ${squareMeters}`}</span>
                  {option.transaction_count !== null && <small>{option.transaction_count.toLocaleString("ko-KR")}건</small>}
                </button>;
              })}</div>
              <div className="filter-field custom-area"><label htmlFor="area">면적 직접 입력</label><div className="input-unit"><input id="area" type="number" min={areaUnit === "m2" ? 1 : 1 / M2_PER_PYEONG} max={areaUnit === "m2" ? 1000 : 1000 / M2_PER_PYEONG} step="any" placeholder={areaUnit === "m2" ? "기본 84" : "기본 25.41"} value={shownArea} onChange={e => setArea(e.target.value === "" ? "" : String(Number(e.target.value) * (areaUnit === "m2" ? 1 : M2_PER_PYEONG)))}/><span>{unitLabel}</span></div></div>
              <button className="text-button" onClick={() => setArea("")}>기본 전용 84㎡ 사용</button>
              <p className="field-hint">{areaValid ? `전용 ${(areaValue ?? 84).toFixed(2)}㎡ ≈ ${((areaValue ?? 84) / M2_PER_PYEONG).toFixed(2)}평` : "면적을 확인해 주세요."} · {monthly ? "월세는 선택한 면적이 속하는 평형대의 건물별 결과를 사용합니다." : "행정동 평당 예측가로 총액을 환산합니다."}</p>
            </>}
            {activePanel === "year" && <><div className="segment" aria-label="예측연도">{[2, 3].map(h => <button type="button" key={h} className={horizon === h ? "active" : ""} aria-pressed={horizon === h} onClick={() => setHorizon(h)}>{2025 + h}년</button>)}</div><p className="field-hint">범죄·교통안전과 매매·전세 가격의 목표연도를 함께 선택합니다. 월세는 2026년 9월 자료 기준의 현재 조건 추정입니다.</p></>}
            {activePanel === "weights" && <div className="score-weight-settings">
              <p className="field-hint">중요하게 생각하는 영역의 비율을 높여보세요. 점수·등급과 추천 동네가 자동으로 바뀝니다.</p>
              <div className="score-weight-values"><div><span>아동범죄노출</span><strong>{crimePercent}<small>%</small></strong></div><div><span>교통안전</span><strong>{100 - crimePercent}<small>%</small></strong></div></div>
              <label className="sr-only" htmlFor="crime-weight">아동범죄노출 반영 비율</label>
              <input id="crime-weight" className="score-weight-slider" type="range" min="0" max="100" step="1" value={crimePercent} aria-valuetext={`아동범죄노출 ${crimePercent}%, 교통안전 ${100 - crimePercent}%`} aria-describedby="score-weight-help" onChange={e => changeCrimePercent(Number(e.target.value))}/>
              <div className="range-scale"><span>교통안전 중심</span><span>균형</span><span>범죄안전 중심</span></div>
              <div className="segment" aria-label="점수 비율 빠른 선택">{[[30, "교통 중심"], [50, "균형"], [70, "범죄 중심"]].map(([value, label]) => <button key={value} type="button" aria-pressed={crimePercent === value} className={crimePercent === value ? "active" : ""} onClick={() => changeCrimePercent(Number(value))}>{label}</button>)}</div>
              <div className="score-weight-footer"><span>합계 100%</span><button className="text-button" onClick={() => changeCrimePercent(50)}>기본값 50:50으로 초기화</button></div>
              <p className="field-hint" id="score-weight-help">두 영역 점수를 합치는 비율입니다. 점수가 높을수록 낮은 위험을 뜻하며, 모델 내부 가중치는 유지됩니다. 선택값은 이 브라우저에 저장됩니다.</p>
              <p className="field-hint" role="status">{error ? "비율을 적용하지 못했습니다. 다시 불러오기를 눌러 주세요." : recommendError ? "추천을 갱신하지 못했습니다. 화면의 예산·면적 또는 오류 안내를 확인해 주세요." : loading || recommending || !recommendations ? "선택한 비율로 점수와 추천을 업데이트하고 있습니다…" : `범죄 ${crimePercent}% · 교통 ${100 - crimePercent}% 적용 완료`}</p>
            </div>}
            {activePanel === "info" && <div className="info-panel">
                {boundaryStatus && <details className="boundary-status"><summary>행정동 {boundaryStatus.applied_year_month.replace("-", ".")} 기준</summary><div><b>{boundaryStatus.neighborhood_count}개 행정동</b>{boundaryStatus.upcoming_changes.length ? boundaryStatus.upcoming_changes.map(c => <a key={c.id} href={c.source_url} target="_blank" rel="noreferrer">{c.district} · {c.title}<small>{c.effective_from ?? "시행일 확인 중"}</small></a>) : <p>확인된 변경 예정이 없습니다.</p>}</div></details>}
                <details className="browse-regions"><summary>지도에 표시한 지역 <span>{visible.length}개</span></summary><div>{rankedRegions.map(r => <button key={r.id} onClick={() => choose(r)}><span>{r.name}<small>{r.parent}</small></span><span>{districts.get(districtOf(r))?.score ?? "—"}<small>소속 구 점수</small></span></button>)}</div></details>
                <details className="data-notes"><summary>점수·가격 기준 보기</summary><p>범죄·교통 모두 Feature별 가중치 합계가 100%입니다. 각 Feature의 자치구 순위와 위험 방향으로 100·80·60·40·20점의 안전점수를 만들고 가중합합니다. 종합점수에는 범죄 {crimePercent}%, 교통 {100 - crimePercent}%를 반영합니다. 상단 ‘비율’에서 변경할 수 있습니다.</p><p>선택한 비율의 동네점수로 25개 구를 정렬해 5개씩 A~E 등급을 부여합니다. 동점은 범죄안전 점수, 자치구명 순입니다.</p><p>예산 범위에 맞는 구마다 가장 저렴한 동을 하나씩 고르고, 구 점수가 높은 순으로 최대 5곳을 추천합니다. 월세는 입력 보증금에서 월세가 가장 낮은 동을 선택합니다. 가격 자료가 없는 동은 제외합니다.</p><p>매매·전세는 전용면적과 평당 예측가로 총액을 환산하며 기본 면적은 84㎡입니다. 월세는 건물별 전세 이력이 있는 같은 면적대의 결과를 제공합니다.</p>{catalog?.warnings.map(w => <p key={w}>{w}</p>)}<p>아동범죄노출은 실제 아동 피해율이 아닌 지역 범죄노출 대리지표입니다.</p></details>

            </div>}
          </section>}
          <div className="map-context" aria-live="polite"><span>아파트 {transactionLabel[transaction]} · {monthly ? `보증금 ${budgetLabel(depositValue || 0)} / 월 ` : `${2025 + horizon}년 · `}{budgetLabel(Number(minBudget) || 0)}–{budgetLabel(Number(budget) || 0)}</span><small>{recommending ? "추천 업데이트 중…" : recommendations ? `지도에 ${recommendations.recommendations.length}곳 추천` : "모델 데이터 확인 중"}</small></div>
          {(loading || error || recommendError || boundaryError || !key || tileError || (recommendations && !recommendations.recommendations.length)) && <div className="map-notices">
            {loading && <p className="status" role="status">모델 결과를 불러오는 중입니다.</p>}
            {error && <div className="error" role="alert">{error}<button onClick={() => setReload(n => n + 1)}>다시 불러오기</button></div>}
            {recommendError && <p className="error" role="alert">{recommendError}<button onClick={() => setRecommendationReload(n => n + 1)}>추천 다시 시도</button></p>}
            {boundaryError && <p className="error" role="alert">{boundaryError}</p>}
            {recommendations && !recommendations.recommendations.length && <p className="empty-state" role="status">예산 범위에 맞는 동네가 없습니다. 금액이나 면적을 조정해 보세요.</p>}
            {!key ? <p className="field-hint">배경지도 키 미설정 · 행정 경계로 표시합니다.</p> : tileError && <p className="field-hint">배경지도 연결을 확인해 주세요.</p>}
          </div>}
        </div>
        {selected && <article className="selection-card" aria-live="polite"><div className="card-heading"><span className="eyebrow">{selected.level === "행정동" ? "NEIGHBORHOOD" : "DISTRICT"}</span><button aria-label="지역 요약 닫기" onClick={() => setSelected(null)}>×</button></div><h2>{selected.name}<small>{selected.parent}</small></h2>
          <div className="price-highlight"><span>{monthly ? "입력 보증금 기준 예상 월세" : `${2025 + horizon}년 ${transactionLabel[transaction]} 예상가격`}</span><strong>{money(selectedPrice)}{monthly && selectedPrice !== null && <small>/월</small>}</strong><small>{selected.level === "구" ? "가격 자료가 있는 하위 행정동 결과의 단순평균" : monthly ? "선택 면적대 건물별 예상 월세의 가중 중앙값" : `전용 ${(areaValue ?? 84).toFixed(2)}㎡ 환산 총액`}</small></div>
          <p className="field-hint">{monthly ? `자료 기준 ${catalog?.property_models.wolse?.trained_until ?? "—"} · 현재 조건 추정` : `2026년 9월 자료 기준 → ${2025 + horizon}년 9월 예측`}</p>
          {selected.level === "행정동" && !selectedProperty && <p className="field-hint">해당 행정동의 가격 자료가 없어 소속 구 점수만 제공합니다.</p>}
          {selectedScore && <><div className="district-score-title"><span>{selectedScore.district} 동네점수<small>소속 자치구 기준</small></span><strong>{points(selectedScore.score)}<small>점</small></strong><span className={`grade-badge grade-${selectedScore.grade.toLowerCase()}`}>{selectedScore.grade}</span></div>{scoreSummary(selectedScore)}<p className="field-hint">서울 25개 구 중 {selectedScore.rank}위 · 같은 구의 행정동은 동일 점수</p></>}
          <button className="secondary-button" onClick={() => detail.current?.showModal()}>가격·점수 자세히 보기 ↗</button>
        </article>}
        <div className="map-legend" aria-label="동네점수 등급: A가 가장 높고 E가 가장 낮음">{["A", "B", "C", "D", "E"].map(g => <span key={g} className={`grade-${g.toLowerCase()}`} title={`동네점수 ${g}등급`}>{g}</span>)}</div>
      </section>
    </div>
    <dialog className="service-dialog" ref={detail}><form method="dialog"><button aria-label="상세 닫기">×</button></form>
      <span className="eyebrow">PRICE & NEIGHBORHOOD SCORE</span><h2>{selected?.name}</h2><p>{selected?.parent} · {transactionLabel[transaction]}</p>
      <div className="price-highlight"><span>{monthly ? "입력 보증금 기준 예상 월세" : `${2025 + horizon}년 예상가격`}</span><strong>{money(selectedPrice)}{monthly && selectedPrice !== null && <small>/월</small>}</strong></div>
      {monthlySummary()}
      {monthlyHistorySeries.length > 0 && <TrendChart title="과거 월세 실거래 흐름 (월 기준)" points={monthlyHistorySeries} unit="만원/월" note="연도별 실제 월세 계약의 동별 평균(동·연도당 5건 이상)입니다. 보증금·면적 조건이 달라 위의 현재 예상 월세와 직접 비교할 수 없으며, 미래 월세 예측값은 아닙니다." />}
      {monthly && monthlyHistorySeries.length === 0 && <p className="field-hint">이 지역의 과거 월세 집계가 없습니다. 현재 보증금 조건의 추정값만 표시합니다.</p>}
      {!monthly && selectedPrice !== null && <section className="detail-facts"><h3>가격을 이해하는 기준</h3>
        <p>계산: 평당 예측가 × 전용 {points(areaValue ?? 84)}㎡ ÷ {M2_PER_PYEONG.toFixed(3)}㎡/평</p>
        <p>선택 면적: {points(areaValue ?? 84)}㎡ · {points((areaValue ?? 84) / M2_PER_PYEONG)}평</p>
        {selected?.level === "구" && <p>해당 구의 가격 자료가 있는 행정동 {catalog?.neighborhoods.filter(row => row.district === selected.name && amountFor(row, transaction, areaValue) !== null).length ?? 0}곳의 단순평균입니다.</p>}
        {districtPriceRank > 0 && <p>같은 구 {districtPeerPrices.length}개 동 중 낮은 가격 순 {districtPriceRank}위 · 최저 {money(districtPeerPrices[0]?.price ?? null)} / 최고 {money(districtPeerPrices.at(-1)?.price ?? null)}</p>}
        {Number.isFinite(Number(budget)) && <p>설정한 최대 예산보다 {selectedPrice <= Number(budget) ? `${money(Number(budget) - selectedPrice)} 낮습니다.` : `${money(selectedPrice - Number(budget))} 높습니다.`}</p>}
      </section>}
      {!monthly && priceSeries.length > 0 && <TrendChart title={`${selected?.level === "구" ? "구 평균" : "행정동"} 평당가격 추세`} points={priceSeries} unit="만원/평" note={priceSeries.some(point => point.year < propertyBaseYear) ? `${propertyBaseYear}년 이전은 연도별 실거래 평균(동·연도당 5건 이상), ${propertyBaseYear}년은 최근 12개월 대표값, 이후는 모델 예측값입니다. 거래 구성 차이로 시점 간 값이 완전히 같은 조건의 비교는 아닙니다.` : `과거 연도별 가격 집계가 없어 ${propertyBaseYear}년 최근 12개월 대표값부터 표시합니다. 이후는 모델 예측값입니다.`} />}
      {selected?.level === "구" && historicalReferenceSeries.length > 0 && <TrendChart title="과거 참고 안전지표 · 자치구 상대순위" points={historicalReferenceSeries} unit="점" note="2017~2025년 실제 5대 범죄건수와 어린이 교통사고건수를 각 연도 아동인구 1만 명당 비율로 환산한 뒤, 해당 연도 25개 구의 5분위 안전점수(100·80·60·40·20)를 현재 선택한 비율로 합산했습니다. 연도별 상대위치만 나타내며 절대적 안전도 변화나 아래 예측 동네점수와 직접 비교할 수 없습니다." />}
      {selected?.level === "구" && scoreSeries.length > 0 && <TrendChart title="동네점수 예측 비교" points={scoreSeries} unit="점" connect={false} note="2027·2028년은 서로 다른 목표연도 모델의 예측 점수입니다. 실제 안전도가 이만큼 변한다는 뜻은 아니며, 위의 과거 참고지표와 계산 기준이 달라 두 그래프를 연결하거나 수치를 직접 비교하지 않습니다." />}
      {selectedScore && <section className="detail-facts"><h3>{selectedScore.district} 점수 구성</h3><p>아동범죄노출 {points(selectedScore.crime_score)} × {crimePercent}% = {points(selectedScore.crime_score * crimePercent / 100)}점</p><p>교통안전 {points(selectedScore.traffic_score)} × {100 - crimePercent}% = {points(selectedScore.traffic_score * (100 - crimePercent) / 100)}점</p><p>합계 {points(selectedScore.score)}점 · 서울 25개 구 중 {selectedScore.rank}위 · {selectedScore.grade}등급</p></section>}
      {catalog?.warnings.map(w => <p className="muted" key={w}>{w}</p>)}<p className="muted">단지별 매물가격이나 실제 아동 피해확률을 의미하지 않습니다. 매매·전세는 행정동 평당 예측가를 입력 전용면적으로 환산합니다.</p>
    </dialog>
  </main>;
}
