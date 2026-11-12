import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import L from "leaflet";
import type { FeatureCollection } from "geojson";
import { addSeoulFocus } from "./seoulFocus";
import { addAdministrativeBoundaries } from "./administrativeBoundaries";
import { addSafetyGradeBoundaries } from "./safetyGradeBoundaries";
import { fetchAdminBoundaryStatus, type AdminBoundaryStatus } from "./adminBoundaryApi";
import { amountFor, fetchCatalog, fetchRecommendations, normalizeDong,
  type Catalog, type DistrictScore, type Neighborhood, type Recommendations, type Transaction } from "./neighborhoodApi";
import "leaflet/dist/leaflet.css";
import "./final-service.css";

type Level = "시" | "구" | "행정동";
type Region = { id: string; name: string; parent: string; level: Level; lat: number; lng: number };
const SEOUL: Region = { id: "seoul", name: "서울특별시", parent: "", level: "시", lat: 37.5665, lng: 126.978 };
const ZOOM: Record<Level, number> = { 시: 10, 구: 11, 행정동: 13 };
const money = (n: number | null) => n === null ? "가격 정보 없음" : `${Math.round(n).toLocaleString("ko-KR")}만원`;
const districtOf = (r: Region) => r.level === "구" ? r.name : r.parent;

export function FinalServicePage() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [reload, setReload] = useState(0);
  const [horizon, setHorizon] = useState(2);
  const [transaction, setTransaction] = useState<Transaction>("sale");
  const [budget, setBudget] = useState("50000");
  const [area, setArea] = useState("");
  const [recommendations, setRecommendations] = useState<Recommendations | null>(null);
  const [recommending, setRecommending] = useState(false);
  const [recommendError, setRecommendError] = useState("");
  const recommendationRequest = useRef<AbortController | null>(null);
  const [regions, setRegions] = useState<Region[]>([SEOUL]);
  const [boundaryStatus, setBoundaryStatus] = useState<AdminBoundaryStatus | null>(null);
  const [boundaryError, setBoundaryError] = useState("");
  const [query, setQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [selected, setSelected] = useState<Region | null>(null);
  const [level, setLevel] = useState<Level>("구");
  const [auto, setAuto] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [view, setView] = useState<"score" | "price">("score");
  const [tileError, setTileError] = useState(false);
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<L.Map | null>(null);
  const detail = useRef<HTMLDialogElement>(null);
  const key = import.meta.env.VITE_VWORLD_API_KEY as string | undefined;
  const parsedArea = area.trim() ? Number(area) : null;
  const areaValid = parsedArea === null || (Number.isFinite(parsedArea) && parsedArea > 0 && parsedArea <= 1000);
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
    recommendationRequest.current?.abort();
    setRecommendations(null); setRecommendError(""); setRecommending(false);
    return () => recommendationRequest.current?.abort();
  }, [budget, area, transaction, horizon]);

  useEffect(() => {
    const controller = new AbortController();
    fetchAdminBoundaryStatus(controller.signal).then(setBoundaryStatus).catch(() => {});
    Promise.all(["seoul-gu.geojson", "seoul-dong.geojson"].map(async file => {
      const r = await fetch(`${import.meta.env.BASE_URL}${file}`, { signal: controller.signal });
      if (!r.ok) throw new Error("행정구역 목록을 불러오지 못했습니다.");
      return r.json() as Promise<FeatureCollection>;
    })).then(([gu, dong]) => {
      if (controller.signal.aborted) return;
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
    const instance = L.map(container.current, { zoomControl: false, minZoom: 9, maxZoom: 18 }).setView([SEOUL.lat, SEOUL.lng], 11);
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
    const onZoom = () => { if (auto) setLevel(instance.getZoom() < 11 ? "시" : instance.getZoom() < 13 ? "구" : "행정동"); };
    instance.on("zoomend", onZoom); onZoom();
    return () => { instance.off("zoomend", onZoom); };
  }, [auto]);

  useEffect(() => {
    if (!map.current) return;
    return addAdministrativeBoundaries(map.current, level, selected?.level === level ? selected : null, visible,
      r => {
        const match = visible.find(v => v.name === r.name && v.parent === r.parent);
        if (!match) return "";
        if (view === "price") return money(priceFor(match));
        const score = districts.get(districtOf(match));
        return score ? `${match.level === "행정동" ? `${match.parent} 기준 · ` : ""}${score.score}점 · ${score.grade}등급` : "점수 정보 없음";
      }, r => { const match = visible.find(v => v.name === r.name && v.parent === r.parent); if (match) setSelected(match); });
  }, [level, selected, visible, view, catalog, transaction, area]);

  useEffect(() => {
    if (!map.current || view !== "score" || !catalog) return;
    return addSafetyGradeBoundaries(map.current, catalog.districts.map(row => ({ district: row.district, predicted_grade: row.grade })));
  }, [catalog, view]);

  useEffect(() => {
    if (!map.current || !recommendations) return;
    const group = L.layerGroup().addTo(map.current);
    for (const row of recommendations.recommendations) {
      const region = regions.find(r => r.level === "행정동" && r.parent === row.district && normalizeDong(r.name) === normalizeDong(row.neighborhood));
      if (!region) continue;
      const badge = document.createElement("span"); badge.textContent = String(row.recommendation_rank);
      const marker = L.marker([region.lat, region.lng], { icon: L.divIcon({ className: "recommend-marker", html: badge, iconSize: [30, 30] }) });
      const tooltip = document.createElement("span"); tooltip.textContent = `${row.neighborhood} · ${money(row.estimated_amount_manwon)}`;
      marker.bindTooltip(tooltip).on("click", () => choose(region)).addTo(group);
    }
    return () => { group.remove(); };
  }, [recommendations, regions]);

  function choose(r: Region) {
    setSelected(r); setLevel(r.level); setAuto(false); setSearchOpen(false); setQuery("");
    map.current?.flyTo([r.lat, r.lng], ZOOM[r.level]);
  }

  function chooseRecommendation(row: Neighborhood) {
    const region = regions.find(r => r.level === "행정동" && r.parent === row.district && normalizeDong(r.name) === normalizeDong(row.neighborhood));
    if (region) choose(region);
    else {
      const gu = regions.find(r => r.level === "구" && r.name === row.district);
      if (gu) choose({ ...gu, id: row.id, name: row.neighborhood, parent: row.district, level: "행정동" });
    }
  }

  async function recommend(event: FormEvent) {
    event.preventDefault();
    const value = Number(budget);
    if (!Number.isFinite(value) || value <= 0 || !areaValid) { setRecommendError("예산과 면적을 올바르게 입력해 주세요."); return; }
    recommendationRequest.current?.abort();
    const controller = new AbortController(); recommendationRequest.current = controller;
    setRecommending(true); setRecommendations(null); setRecommendError("");
    try {
      const data = await fetchRecommendations(value, transaction, horizon, areaValue, controller.signal);
      if (!controller.signal.aborted) {
        setRecommendations(data);
        setSelected(null); setAuto(false); setLevel("구"); map.current?.setView([SEOUL.lat, SEOUL.lng], 11);
      }
    } catch (e) { if (!controller.signal.aborted) setRecommendError(e instanceof Error ? e.message : "추천 실패"); }
    finally { if (!controller.signal.aborted) setRecommending(false); }
  }

  function scoreSummary(score: DistrictScore) {
    return <div className="score-breakdown">
      <div><span>아동범죄노출</span><strong>{score.crime_score}<small>점</small></strong><small>{score.crime_prediction_year}년 예측 · 50%</small></div>
      <div><span>교통안전</span><strong>{score.traffic_score}<small>점</small></strong><small>{score.traffic_prediction_year}년 참고 · 50%</small></div>
    </div>;
  }

  return <main className="spotfit-final">
    <header className="app-header"><a className="brand" href="/"><span className="brand-icon">⌂</span>SpotFit<span className="brand-subtitle">아이와 함께, 더 안전한 동네</span></a>
      <div className="header-meta"><span>주거 예산 × 동네점수</span>{boundaryStatus && <details className="boundary-status"><summary>행정동 {boundaryStatus.applied_year_month.replace("-", ".")} 기준</summary><div><b>{boundaryStatus.neighborhood_count}개 행정동</b>{boundaryStatus.upcoming_changes.length ? boundaryStatus.upcoming_changes.map(c => <a key={c.id} href={c.source_url} target="_blank" rel="noreferrer">{c.district} · {c.title}<small>{c.effective_from ?? "시행일 확인 중"}</small></a>) : <p>확인된 변경 예정이 없습니다.</p>}</div></details>}</div>
    </header>
    <div className={`service-workspace ${sidebarOpen ? "" : "sidebar-collapsed"}`}>
      <aside className="service-sidebar" id="service-sidebar" hidden={!sidebarOpen}>
        <span className="eyebrow">FIND YOUR NEXT NEIGHBORHOOD</span><h1>예산 안에서,<br/>더 안심되는 동네.</h1><p className="muted intro">아파트 가격과 아이의 생활환경을<br/>함께 살펴보세요.</p>
        <form className="budget-form" onSubmit={recommend}>
          <h2><span className="step">01</span> 나의 주거 조건</h2>
          <div className="segment" aria-label="거래유형">{(["sale", "rent"] as Transaction[]).map(t => <button type="button" key={t} aria-pressed={transaction === t} className={transaction === t ? "active" : ""} onClick={() => setTransaction(t)}>{t === "sale" ? "매매" : "전세"}</button>)}</div>
          <label htmlFor="budget">최대 예산 <small>만원 단위</small></label><div className="input-unit"><input id="budget" type="number" required min="1" max="1000000000" step="any" value={budget} onChange={e => setBudget(e.target.value)}/><span>만원</span></div>
          <p className="field-hint">{Number(budget) > 0 ? `약 ${(Number(budget) / 10000).toLocaleString("ko-KR", { maximumFractionDigits: 2 })}억원 이내` : "예산을 입력해 주세요"}</p>
          <label htmlFor="area">희망 전용면적 <small>선택 입력</small></label><div className="input-unit"><input id="area" type="number" min="1" max="1000" step="any" placeholder="미입력 시 지역 대표면적" value={area} onChange={e => setArea(e.target.value)}/><span>㎡</span></div>
          <p className="field-hint">면적 입력 시 대표 평당 추정가로 총액을 환산합니다.</p>
          <label>범죄노출 예측연도</label><div className="segment">{[2, 3].map(h => <button type="button" key={h} className={horizon === h ? "active" : ""} aria-pressed={horizon === h} onClick={() => setHorizon(h)}>{2025 + h}년 <small>{h}년 모델</small></button>)}</div>
          <div className="fixed-domains"><span>아동범죄노출 <b>50%</b></span><span>교통안전 <b>50%</b></span><small>두 영역 자동 반영 · 점수가 높을수록 낮은 위험</small></div>
          <button className="primary-button" disabled={loading || !!error || recommending}>{recommending ? "추천 동네를 찾는 중…" : "예산에 맞는 동네 5곳 찾기 →"}</button>
        </form>
        {loading && <p className="status" role="status">모델 결과를 불러오는 중입니다. 최초 로딩은 시간이 걸릴 수 있습니다.</p>}
        {error && <div className="error" role="alert">{error}<button onClick={() => setReload(n => n + 1)}>다시 불러오기</button></div>}
        {recommendError && <p className="error" role="alert">{recommendError}</p>}
        {recommendations && <section className="recommendations" aria-live="polite"><h2>추천 동네 TOP {recommendations.recommendations.length}<small>예산 충족 {recommendations.candidate_count}곳</small></h2>
          {!recommendations.recommendations.length && <div className="empty-state">조건에 맞는 동네가 없습니다.<p>예산을 높이거나 희망 면적을 조정해 보세요.</p></div>}
          {recommendations.recommendations.map(row => <button key={row.id} className="recommend-card" onClick={() => chooseRecommendation(row)}><span className="rank-number">{row.recommendation_rank}</span><span><strong>{row.neighborhood}</strong><small>{row.district} · 구 점수 {row.district_score.score}점</small><b>{money(row.estimated_amount_manwon)}</b><small>예산 여유 {money(row.budget_remaining_manwon)}</small></span><span className={`grade-badge grade-${row.district_score.grade.toLowerCase()}`}>{row.district_score.grade}</span></button>)}
          <p className="field-hint">동네점수 → 범죄안전 점수 → 낮은 가격 순입니다.</p>
        </section>}
        <section className="search-section"><h2><span className="step">02</span> 관심 지역 검색</h2><label className="sr-only" htmlFor="region-search">자치구 또는 행정동 검색</label><div className="input-unit search-input"><input id="region-search" value={query} onFocus={() => setSearchOpen(true)} onChange={e => { setQuery(e.target.value); setSearchOpen(true); }} onKeyDown={e => { if (e.key === "Escape") setSearchOpen(false); if (e.key === "Enter" && found[0]) choose(found[0]); }} placeholder="예: 역삼1동, 강남구"/><button aria-label="검색어 지우기" onClick={() => { setQuery(""); setSearchOpen(false); }}>×</button></div>
          {searchOpen && <div className="search-results"><small>검색 결과 {found.length}개</small>{found.map(r => <button key={r.id} onClick={() => choose(r)}><span>{r.name}<small>{r.parent}</small></span><small>{r.level}</small></button>)}{!found.length && <p>검색된 지역이 없습니다.</p>}</div>}
          {boundaryError && <p className="error">{boundaryError}</p>}
        </section>
        <details className="browse-regions"><summary>지도에 표시한 지역 <span>{visible.length}개</span></summary><div>{rankedRegions.map(r => <button key={r.id} onClick={() => choose(r)}><span>{r.name}<small>{r.parent}</small></span><span>{districts.get(districtOf(r))?.score ?? "—"}<small>소속 구 점수</small></span></button>)}</div></details>
        <details className="data-notes"><summary>점수·가격 기준 보기</summary><p>각 영역의 5분위 점수를 50%씩 반영합니다. 동네점수로 25개 구를 정렬해 5개씩 A~E 등급을 부여합니다. 동점은 범죄안전 점수, 자치구명 순으로 정렬합니다.</p>{catalog?.warnings.map(w => <p key={w}>{w}</p>)}<p>아동범죄노출은 실제 아동 피해율이 아닌 지역 범죄노출 대리지표입니다.</p></details>
      </aside>
      <section className="service-map" aria-label="서울 동네 추천 지도"><div className="map-canvas" ref={container}/>
        <div className="map-toolbar"><div className="toolbar-group"><button aria-controls="service-sidebar" aria-expanded={sidebarOpen} onClick={() => setSidebarOpen(v => !v)}>☰ <span>조건</span></button><button className={view === "score" ? "active" : ""} onClick={() => setView("score")}>동네점수</button><button className={view === "price" ? "active" : ""} onClick={() => setView("price")}>아파트 가격</button></div><div className="toolbar-group"><button aria-pressed={auto} className={auto ? "active" : ""} onClick={() => setAuto(true)}>자동</button>{(["시", "구", "행정동"] as Level[]).map(l => <button key={l} className={!auto && level === l ? "active" : ""} onClick={() => { setAuto(false); setLevel(l); map.current?.setZoom(ZOOM[l]); }}>{l}</button>)}</div></div>
        <div className="map-context"><span>{catalog ? `범죄 ${catalog.crime_prediction_year}년 예측 · 교통 ${catalog.traffic_prediction_year}년 참고` : "모델 데이터 확인 중"}</span><small>{!key ? "배경지도 키 미설정 · 행정 경계는 조회할 수 있습니다." : tileError ? "배경지도 연결을 확인해 주세요." : "지역을 검색하거나 지도 위 영역을 선택해 보세요."}</small></div>
        {selected && <article className="selection-card" aria-live="polite"><div className="card-heading"><span className="eyebrow">{selected.level === "행정동" ? "NEIGHBORHOOD" : "DISTRICT"}</span><button aria-label="지역 요약 닫기" onClick={() => setSelected(null)}>×</button></div><h2>{selected.name}<small>{selected.parent}</small></h2>
          <div className="price-highlight"><span>{transaction === "sale" ? "매매" : "전세"} 대표가격 추정</span><strong>{money(selectedPrice)}</strong><small>{areaValue ? `전용 ${areaValue}㎡ 환산 총액` : selected.level === "구" ? "하위 행정동 대표 총액의 단순평균" : "최신 거래 특성 기준 대표 총액"}</small></div>
          <p className="field-hint">가격 기준 {catalog?.property_models[transaction].data_year ?? "—"}년 · 미래가격 예측 아님</p>
          {selected.level === "행정동" && !selectedProperty && <p className="field-hint">해당 행정동의 가격 자료가 없어 소속 구 점수만 제공합니다.</p>}
          {selectedScore && <><div className="district-score-title"><span>{selectedScore.district} 동네점수<small>소속 자치구 기준</small></span><strong>{selectedScore.score}<small>점</small></strong><span className={`grade-badge grade-${selectedScore.grade.toLowerCase()}`}>{selectedScore.grade}</span></div>{scoreSummary(selectedScore)}<p className="field-hint">서울 25개 구 중 {selectedScore.rank}위 · 같은 구의 행정동은 동일 점수</p></>}
          <button className="secondary-button" onClick={() => detail.current?.showModal()}>가격·점수 자세히 보기 ↗</button>
        </article>}
        <div className="map-legend"><b>동네점수 · 자치구 상대등급</b><div>{["A", "B", "C", "D", "E"].map((g, i) => <span key={g}><i className={`grade-${g.toLowerCase()}`}/>{g} <small>{i * 20}–{(i + 1) * 20}%</small></span>)}</div><small>A에 가까울수록 높은 점수 · 행정동은 소속 구 점수</small></div>
      </section>
    </div>
    <dialog className="service-dialog" ref={detail}><form method="dialog"><button aria-label="상세 닫기">×</button></form><span className="eyebrow">PRICE & NEIGHBORHOOD SCORE</span><h2>{selected?.name}</h2><p>{selected?.parent} · {transaction === "sale" ? "매매" : "전세"}</p><div className="price-highlight"><span>선택 조건의 대표가격 추정</span><strong>{money(selectedPrice)}</strong></div>{selectedScore && <><h3>{selectedScore.district} 기준 동네점수 {selectedScore.score}점</h3>{scoreSummary(selectedScore)}</>}{catalog?.warnings.map(w => <p className="muted" key={w}>{w}</p>)}<p className="muted">단지별 매물가격이나 실제 아동 피해확률을 의미하지 않습니다. 면적 입력 시 행정동 대표 평당가로 환산한 참고 추정값입니다.</p></dialog>
  </main>;
}
