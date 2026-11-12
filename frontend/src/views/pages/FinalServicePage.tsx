import { useEffect, useMemo, useRef, useState } from "react";
import L from "leaflet";
import type { FeatureCollection } from "geojson";
import { addSeoulFocus } from "./seoulFocus";
import { addAdministrativeBoundaries } from "./administrativeBoundaries";
import { addSafetyGradeBoundaries } from "./safetyGradeBoundaries";
import { fetchAdminBoundaryStatus, type AdminBoundaryStatus } from "./adminBoundaryApi";
import { BudgetRange, ControlIcon, budgetLabel, type Panel } from "./DiscoveryControls";
import { amountFor, fetchCatalog, fetchRecommendations, normalizeDong,
  fetchAreaPresets, type AreaPresets, type Catalog, type DistrictScore, type Neighborhood, type Recommendations, type Transaction } from "./neighborhoodApi";
import "leaflet/dist/leaflet.css";
import "./final-service.css";

type Level = "시" | "구" | "행정동";
type Region = { id: string; name: string; parent: string; level: Level; lat: number; lng: number };
const SEOUL: Region = { id: "seoul", name: "서울특별시", parent: "", level: "시", lat: 37.5665, lng: 126.978 };
const ZOOM: Record<Level, number> = { 시: 10, 구: 11, 행정동: 13 };
const money = (n: number | null) => n === null ? "가격 정보 없음" : `${Math.round(n).toLocaleString("ko-KR")}만원`;
const points = (n: number) => n.toLocaleString("ko-KR", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const districtOf = (r: Region) => r.level === "구" ? r.name : r.parent;

export function FinalServicePage() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [reload, setReload] = useState(0);
  const [horizon, setHorizon] = useState(2);
  const [transaction, setTransaction] = useState<Transaction>("sale");
  const [budget, setBudget] = useState("50000");
  const [minBudget, setMinBudget] = useState("0");
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
  const areaValue = areaValid ? parsedArea : null;
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
    if (!areaValid || r.level === "시") return null;
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
  const rankedRegions = [...visible].sort((a, b) => (districts.get(districtOf(a))?.rank ?? 999)
    - (districts.get(districtOf(b))?.rank ?? 999) || a.name.localeCompare(b.name, "ko"));

  useEffect(() => {
    const controller = new AbortController();
    setAreaPresets(null); setAreaError("");
    fetchAreaPresets(transaction, controller.signal).then(data => {
      if (!controller.signal.aborted) setAreaPresets(data);
    }).catch(() => { if (!controller.signal.aborted) setAreaError("면적 선택지를 불러오지 못했습니다. 직접 입력할 수 있습니다."); });
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
    fetchCatalog(horizon, controller.signal).then(data => {
      if (!controller.signal.aborted) setCatalog(data);
    }).catch(e => {
      if (!controller.signal.aborted) {
        setError(`${e instanceof Error ? e.message : "조회 실패"} · 모델 복구 상태를 확인하며 10초 후 다시 시도합니다.`);
        retryTimer = window.setTimeout(() => setReload(value => value + 1), 10_000);
      }
    })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => { controller.abort(); if (retryTimer !== undefined) window.clearTimeout(retryTimer); };
  }, [horizon, reload]);

  useEffect(() => {
    setRecommendations(null); setRecommendError(""); setRecommending(false);
    const value = Number(budget);
    const minimum = Number(minBudget);
    if (!minBudget.trim() || !Number.isFinite(minimum) || minimum < 0 || minimum > value ||
        !Number.isFinite(value) || value < 1 || value > 1_000_000_000 || !areaValid) {
      setRecommendError("예산 범위(0 ≤ 하한 ≤ 상한)와 면적(1~1,000㎡)을 확인해 주세요.");
      return;
    }
    if (!catalog || catalog.horizon !== horizon || loading || error) return;
    const controller = new AbortController();
    setRecommending(true);
    const timer = window.setTimeout(() => {
      fetchRecommendations(value, transaction, horizon, areaValue, controller.signal, minimum)
        .then(data => { if (!controller.signal.aborted) setRecommendations(data); })
        .catch(e => { if (!controller.signal.aborted) setRecommendError(e instanceof Error ? e.message : "추천 실패"); })
        .finally(() => { if (!controller.signal.aborted) setRecommending(false); });
    }, 450);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [budget, minBudget, area, transaction, horizon, catalog, loading, error, recommendationReload]);

  useEffect(() => {
    const controller = new AbortController();
    fetchAdminBoundaryStatus(controller.signal).then(setBoundaryStatus).catch(() => {});
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
    }).catch(e => { if (!controller.signal.aborted) setBoundaryError(String(e)); });
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
        return money(priceFor(match));
      }, r => { const match = visible.find(v => v.name === r.name && v.parent === r.parent); if (match) setSelected(match); });
  }, [level, selected, visible, catalog, transaction, area]);

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
      const tooltip = document.createElement("span"); tooltip.textContent = `${row.neighborhood} · ${money(row.estimated_amount_manwon)}`;
      marker.bindTooltip(tooltip, { permanent: map.current.getSize().x > 760, direction: "bottom", className: "recommend-tooltip", offset: [0, 14] }).on("click", () => choose(region)).addTo(group);
    }
    return () => { group.remove(); };
  }, [recommendations, regions, dongGeometry]);

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

  function scoreSummary(score: DistrictScore) {
    return <div className="score-breakdown">
      <div><span>아동범죄노출</span><strong>{points(score.crime_score)}<small>점</small></strong><small>Feature 가중합 · {score.crime_prediction_year}년 · 50%</small></div>
      <div><span>교통안전</span><strong>{points(score.traffic_score)}<small>점</small></strong><small>5분위 점수 · {score.traffic_prediction_year}년 · 50%</small></div>
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
                ["search", "지역 검색", "검색"], ["transaction", "거래 유형", transaction === "sale" ? "매매" : "전세"],
                ["budget", "예산 범위", "예산"], ["area", "전용면적", area ? `${area}㎡` : "면적"],
                ["year", "예측연도", `${2025 + horizon}`], ["info", "지역·이용 안내", "안내"],
              ] as [Panel, string, string][]).map(([panel, label, caption]) => <button key={panel} className={`control-button ${activePanel === panel ? "active" : ""}`} data-panel={panel} aria-label={label} title={label} aria-expanded={activePanel === panel} aria-controls={activePanel === panel ? "discovery-panel" : undefined} onClick={() => setActivePanel(p => p === panel ? null : panel)}><ControlIcon name={panel}/><span>{caption}</span></button>)}
              <button className="control-button overview-button" aria-label="서울 전체 보기" title="서울 전체 보기" onClick={() => { setActivePanel(null); showAll(); }}><ControlIcon name="locate"/><span>전체</span></button>
            </nav>
          </div>
          {activePanel && <section id="discovery-panel" className="discovery-panel" aria-label={{ search: "지역 검색 설정", transaction: "거래 유형 설정", budget: "예산 범위 설정", area: "전용면적 설정", year: "예측연도 설정", info: "지역 이용 안내" }[activePanel]}>
            <div className="panel-heading"><h2>{{ search: "어느 동네가 궁금하세요?", transaction: "거래 유형", budget: "나의 예산 범위", area: "어떤 면적을 찾으세요?", year: "범죄노출 예측연도", info: "지역·이용 안내" }[activePanel]}</h2><button aria-label="설정 닫기" onClick={() => { toolbar.current?.querySelector<HTMLButtonElement>(`[data-panel="${activePanel}"]`)?.focus(); setActivePanel(null); }}>×</button></div>
            {activePanel === "search" && <section className="search-section">
                <label className="sr-only" htmlFor="region-search">자치구 또는 행정동 검색</label>
                <div className="input-unit search-input"><span aria-hidden="true">⌕</span><input id="region-search" autoComplete="off" value={query} onFocus={() => setSearchOpen(true)} onChange={e => { setQuery(e.target.value); setSearchOpen(true); }} onKeyDown={e => { if (e.key === "Escape") setSearchOpen(false); if (e.key === "Enter" && found[0]) choose(found[0]); }} placeholder="어느 동네가 궁금하세요? 예: 역삼1동, 강남구"/><button aria-label="검색어 지우기" onClick={() => { setQuery(""); setSearchOpen(false); }}>×</button></div>
                {searchOpen && <div className="search-results"><div className="search-results-heading"><small>검색 결과 {found.length}개</small><button onClick={() => setSearchOpen(false)} aria-label="검색 결과 닫기">닫기</button></div>{found.map(r => <button key={r.id} onClick={() => choose(r)}><span>{r.name}<small>{r.parent}</small></span><small>{r.level}</small></button>)}{!found.length && <p>검색된 지역이 없습니다.</p>}</div>}

            </section>}
            {activePanel === "transaction" && <><div className="segment" aria-label="거래유형">{(["sale", "rent"] as Transaction[]).map(t => <button type="button" key={t} aria-pressed={transaction === t} className={transaction === t ? "active" : ""} onClick={() => setTransaction(t)}>{t === "sale" ? "매매" : "전세"}</button>)}</div><p className="field-hint">선택한 거래 유형으로 가격과 추천이 자동 변경됩니다.</p></>}
            {activePanel === "budget" && <BudgetRange lower={minBudget} upper={budget} onLower={setMinBudget} onUpper={setBudget}/>}
            {activePanel === "area" && <>
              <p className="field-hint">{areaPresets?.description ?? (areaError || "거래 면적을 확인하고 있습니다…")}</p>
              <div className="area-presets">{areaPresets?.options.map(option => <button key={option.area_m2} aria-pressed={area === String(option.area_m2)} className={area === String(option.area_m2) ? "active" : ""} onClick={() => setArea(String(option.area_m2))}><strong>{option.label}</strong><span>전용 {option.area_m2}㎡</span>{option.transaction_count !== null && <small>{option.transaction_count.toLocaleString("ko-KR")}건</small>}</button>)}</div>
              <div className="filter-field custom-area"><label htmlFor="area">면적 직접 입력 <small>선택</small></label><div className="input-unit"><input id="area" type="number" min="1" max="1000" step="any" placeholder="지역 대표면적" value={area} onChange={e => setArea(e.target.value)}/><span>㎡</span></div></div>
              <button className="text-button" onClick={() => setArea("")}>면적 선택 해제 · 지역 대표면적 사용</button>
              <p className="field-hint">전용면적 기준입니다. 지역 대표 평당가로 총액을 환산합니다.</p>
            </>}
            {activePanel === "year" && <><div className="segment" aria-label="범죄노출 예측연도">{[2, 3].map(h => <button type="button" key={h} className={horizon === h ? "active" : ""} aria-pressed={horizon === h} onClick={() => setHorizon(h)}>{2025 + h}년</button>)}</div><p className="field-hint">범죄노출 예측연도를 선택합니다. 교통안전은 {catalog?.traffic_prediction_year ?? "—"}년 참고 결과입니다.</p></>}
            {activePanel === "info" && <div className="info-panel">
                {boundaryStatus && <details className="boundary-status"><summary>행정동 {boundaryStatus.applied_year_month.replace("-", ".")} 기준</summary><div><b>{boundaryStatus.neighborhood_count}개 행정동</b>{boundaryStatus.upcoming_changes.length ? boundaryStatus.upcoming_changes.map(c => <a key={c.id} href={c.source_url} target="_blank" rel="noreferrer">{c.district} · {c.title}<small>{c.effective_from ?? "시행일 확인 중"}</small></a>) : <p>확인된 변경 예정이 없습니다.</p>}</div></details>}
                <details className="browse-regions"><summary>지도에 표시한 지역 <span>{visible.length}개</span></summary><div>{rankedRegions.map(r => <button key={r.id} onClick={() => choose(r)}><span>{r.name}<small>{r.parent}</small></span><span>{districts.get(districtOf(r))?.score ?? "—"}<small>소속 구 점수</small></span></button>)}</div></details>
                <details className="data-notes"><summary>점수·가격 기준 보기</summary><p>아동범죄노출은 모델 Feature별 관련도를 합계 100%로 정규화하고, 각 Feature의 자치구 순위를 100·80·60·40·20점으로 바꾼 뒤 가중합합니다. 교통안전 5분위 점수와 각각 50% 반영합니다.</p><p>최종 동네점수로 25개 구를 정렬해 5개씩 A~E 등급을 부여합니다. 동점은 범죄안전 점수, 자치구명 순입니다.</p><p>예산 범위에 맞는 구마다 가장 저렴한 동을 하나씩 고르고, 구 점수가 높은 순으로 최대 5곳을 추천합니다. 가격 자료가 없는 동은 제외합니다.</p><p>면적을 입력하면 행정동 대표 평당가로 총액을 환산합니다. 미입력 시 지역 대표 총액을 사용합니다.</p>{catalog?.warnings.map(w => <p key={w}>{w}</p>)}<p>아동범죄노출은 실제 아동 피해율이 아닌 지역 범죄노출 대리지표입니다.</p></details>

            </div>}
          </section>}
          <div className="map-context" aria-live="polite"><span>아파트 {transaction === "sale" ? "매매" : "전세"} · {budgetLabel(Number(minBudget) || 0)}–{budgetLabel(Number(budget) || 0)}</span><small>{recommending ? "추천 업데이트 중…" : recommendations ? `지도에 ${recommendations.recommendations.length}곳 추천` : "모델 데이터 확인 중"}</small></div>
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
          <div className="price-highlight"><span>{transaction === "sale" ? "매매" : "전세"} 대표가격 추정</span><strong>{money(selectedPrice)}</strong><small>{areaValue ? `전용 ${areaValue}㎡ 환산 총액` : selected.level === "구" ? "하위 행정동 대표 총액의 단순평균" : "최신 거래 특성 기준 대표 총액"}</small></div>
          <p className="field-hint">가격 기준 {catalog?.property_models[transaction].data_year ?? "—"}년 · 미래가격 예측 아님</p>
          {selected.level === "행정동" && !selectedProperty && <p className="field-hint">해당 행정동의 가격 자료가 없어 소속 구 점수만 제공합니다.</p>}
          {selectedScore && <><div className="district-score-title"><span>{selectedScore.district} 동네점수<small>소속 자치구 기준</small></span><strong>{points(selectedScore.score)}<small>점</small></strong><span className={`grade-badge grade-${selectedScore.grade.toLowerCase()}`}>{selectedScore.grade}</span></div>{scoreSummary(selectedScore)}<p className="field-hint">서울 25개 구 중 {selectedScore.rank}위 · 같은 구의 행정동은 동일 점수</p></>}
          <button className="secondary-button" onClick={() => detail.current?.showModal()}>가격·점수 자세히 보기 ↗</button>
        </article>}
        <div className="map-legend" aria-label="동네점수 등급: A가 가장 높고 E가 가장 낮음">{["A", "B", "C", "D", "E"].map(g => <span key={g} className={`grade-${g.toLowerCase()}`} title={`동네점수 ${g}등급`}>{g}</span>)}</div>
      </section>
    </div>
    <dialog className="service-dialog" ref={detail}><form method="dialog"><button aria-label="상세 닫기">×</button></form><span className="eyebrow">PRICE & NEIGHBORHOOD SCORE</span><h2>{selected?.name}</h2><p>{selected?.parent} · {transaction === "sale" ? "매매" : "전세"}</p><div className="price-highlight"><span>선택 조건의 대표가격 추정</span><strong>{money(selectedPrice)}</strong></div>{selectedScore && <><h3>{selectedScore.district} 기준 동네점수 {points(selectedScore.score)}점</h3>{scoreSummary(selectedScore)}</>}{catalog?.warnings.map(w => <p className="muted" key={w}>{w}</p>)}<p className="muted">단지별 매물가격이나 실제 아동 피해확률을 의미하지 않습니다. 면적 입력 시 행정동 대표 평당가로 환산한 참고 추정값입니다.</p></dialog>
  </main>;
}
