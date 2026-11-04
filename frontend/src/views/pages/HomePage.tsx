import { useEffect, useMemo, useRef, useState } from "react";
import L from "leaflet";
import { addSeoulFocus } from "./seoulFocus";
import { addAdministrativeBoundaries } from "./administrativeBoundaries";
import { addSafetyGradeBoundaries } from "./safetyGradeBoundaries";
import { SafetyPanel } from "./SafetyPanel";
import { fetchCrimeSafety, fetchSafetyCategoryScores, type CrimeSafetyResponse, type SafetyCategory, type SafetyCategoryResponse } from "./crimeSafetyApi";
import { buildCompositeSafety } from "./safetyComposite";
import { buildHousingSeries, HousingPriceChart, SafetyGradeChart } from "./DetailCharts";
import { fetchRecommenderForecasts, type DistrictModelForecast, type RecommenderResponse } from "./recommenderApi";
import type { FeatureCollection } from "geojson";
import "leaflet/dist/leaflet.css";
import "./housing-map.css";

/**
 * React는 화면 상태를, Leaflet은 지도 DOM과 레이어를 관리한다.
 * 상태 변경 → effect로 지도 갱신 → cleanup으로 이전 레이어 제거 순서로 동작한다.
 * 가격·시설은 UI 예시이고 지역점수는 FastAPI 예측 결과를 사용한다.
 */
type Level = "시" | "구" | "행정동";
type Mode = "현재 가격" | "예측 가격" | "예상 변동률" | "지역점수";
// id는 UI 식별자(행정코드 아님), parent는 상위 지역명, lat/lng는 버튼의 위도/경도다.
// price는 만원/㎡, growth는 연간 변동률(%)이다. 지역점수는 백엔드 API에서 받는다.
type Region = { id: string; name: string; parent: string; level: Level; lat: number; lng: number; price: number; growth: number };
// 가격 단위는 만원/㎡. 실제 거래·행정 경계·시설 정보가 아닌 화면 검토용 고정 fixture다.
const regions: Region[] = [
  { id: "seoul", name: "서울특별시", parent: "", level: "시", lat: 37.5665, lng: 126.978, price: 1450, growth: 3.2 },
  ...[
    ["mapo", "마포구", 37.563, 126.913, 1500, 3.8],
    ["seongdong", "성동구", 37.563, 127.037, 1780, 4.5],
    ["songpa", "송파구", 37.505, 127.106, 1910, 2.8],
    ["nowon", "노원구", 37.654, 127.056, 980, -1.2],
    ["gangseo", "강서구", 37.55, 126.849, 1120, 2.1],
  ].map(([id, name, lat, lng, price, growth]) => ({ id: String(id), name: String(name), parent: "서울특별시", level: "구" as Level, lat: Number(lat), lng: Number(lng), price: Number(price), growth: Number(growth) })),
  ...[
    ["seongsu", "성수1가1동", "성동구", 37.542, 127.049, 1920, 4.8],
    ["haengdang", "행당1동", "성동구", 37.558, 127.036, 1640, 3.2],
    ["sangam", "상암동", "마포구", 37.578, 126.893, 1460, 2.5],
    ["yeonnam", "연남동", "마포구", 37.565, 126.923, 1390, 1.8],
    ["jamsil", "잠실3동", "송파구", 37.509, 127.093, 2180, 3.4],
    ["gongneung", "공릉1동", "노원구", 37.623, 127.073, 940, -0.8],
    ["balsan", "발산1동", "강서구", 37.553, 126.838, 1240, 2.6],
  ].map(([id, name, parent, lat, lng, price, growth]) => ({ id: String(id), name: String(name), parent: String(parent), level: "행정동" as Level, lat: Number(lat), lng: Number(lng), price: Number(price), growth: Number(growth) })),
];
const modes: Mode[] = ["현재 가격", "예측 가격", "예상 변동률", "지역점수"];
const safetyCategories: { key: SafetyCategory; label: string; symbol: string }[] = [
  { key: "traffic", label: "교통사고", symbol: "🚦" },
  { key: "crime", label: "범죄율", symbol: "🛡" },
  { key: "infrastructure", label: "육아환경", symbol: "🏥" },
];
// 수동 단위 선택 시 사용할 확대 수준. atZoom은 자동 모드에서 반대로 단위를 구한다.
// 11 미만: 시, 11~12: 구, 13 이상: 행정동. 행정 기준이 아닌 화면 표시 기준이다.
const zoomFor: Record<Level, number> = { 시: 10, 구: 11, 행정동: 13 };
const atZoom = (z: number): Level => z < 11 ? "시" : z < 13 ? "구" : "행정동";
const format = (n: number) => Math.round(n).toLocaleString("ko-KR");
// 1200 = 100(백분율) × 12(월→년). 연 3%를 36개월 적용하면 가격 × 1.09다.
// 복리나 머신러닝 추론이 아닌 기간 비례 계산이며 API 연결 시 교체할 부분이다.
const forecast = (r: Region, months: number) => r.price * (1 + r.growth * months / 1200);
const value = (r: Region, mode: Exclude<Mode, "지역점수">, months: number) => mode === "현재 가격" ? r.price : mode === "예측 가격" ? forecast(r, months) : r.growth * months / 12;
const label = (r: Region, mode: Exclude<Mode, "지역점수">, months: number) => `${mode === "예상 변동률" ? value(r, mode, months).toFixed(1) : format(value(r, mode, months))}${mode === "예상 변동률" ? "%" : "만"}`;
const housingPinLabel = (r: Region, mode: Mode, months: number, safetyLabel: string, model?: DistrictModelForecast) => {
  if (mode === "지역점수") return safetyLabel;
  if (mode === "예상 변동률") return label(r, mode, months);
  const sale = model?.apart_sale ?? (mode === "현재 가격" ? r.price : forecast(r, months));
  const jeonse = model?.apart_rent ?? sale * 0.54;
  return `매매 ${format(sale)}만 · 전세 ${format(jeonse)}만`;
};

export function HomePage() {
  const [safety, setSafety] = useState<CrimeSafetyResponse | null>(null);
  const [categoryScores, setCategoryScores] = useState<SafetyCategoryResponse | null>(null);
  const [recommender, setRecommender] = useState<RecommenderResponse | null>(null);
  const [administrativeRegions, setAdministrativeRegions] = useState<Region[]>([]);
  const [selectedCategories, setSelectedCategories] = useState<SafetyCategory[]>(["traffic", "crime", "infrastructure"]);
  const [safetyLoading, setSafetyLoading] = useState(true);
  const [safetyError, setSafetyError] = useState("");
  const displayedSafety = useMemo(
    () => buildCompositeSafety(safety, categoryScores, selectedCategories),
    [safety, categoryScores, selectedCategories],
  );
  const categoryWeight = selectedCategories.length ? 100 / selectedCategories.length : 0;
  const safetyText = (r: Region) => {
    if (r.level !== "구") return "";
    const result = displayedSafety?.forecasts.find(item => item.district === r.name);
    return result ? `${result.predicted_grade}등급 · ${result.predicted_rank}위` : "데이터 준비 중";
  };
  const modelFor = (r: Region): DistrictModelForecast | undefined => {
    const district = r.level === "구" ? r.name : r.parent;
    const base = recommender?.forecasts.find(item => item.district === district);
    if (!base || r.level !== "행정동") return base;
    const neighborhood = recommender?.neighborhoods.find(
      item => item.district === district && item.neighborhood === r.name,
    );
    return neighborhood ? {
      ...base,
      apart_sale: neighborhood.apart_sale_per_pyeong,
      apart_sale_amount: neighborhood.apart_sale_amount,
      apart_rent: neighborhood.apart_rent_per_pyeong,
      apart_rent_amount: neighborhood.apart_rent_amount,
    } : { ...base, apart_sale: null, apart_sale_amount: null, apart_rent: null, apart_rent_amount: null };
  };
  const displayLabel = (r: Region, m: Mode, n: number) => {
    if (m === "지역점수") return safetyText(r);
    const model = modelFor(r);
    if ((m === "현재 가격" || m === "예측 가격") && model?.apart_sale != null) return `${format(model.apart_sale)}만`;
    return label(r, m, n);
  };
  // ref는 렌더 사이에 DOM/지도 인스턴스를 보존한다. 변경해도 React 재렌더를 유발하지 않는다.
  // 반면 useState 값은 버튼·카드·지도 갱신에 필요하므로 변경 시 재렌더를 유발한다.
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<L.Map | null>(null);
  const [level, setLevel] = useState<Level>("구");
  // auto=false일 때는 줌이 변해도 사용자가 직접 선택한 지역 단위를 유지한다.
  const [auto, setAuto] = useState(true);
  const [mode, setMode] = useState<Mode>("지역점수");
  const [months, setMonths] = useState(12);
  // selected에는 기본 지역이 있지만 summaryOpen=false여서 첫 화면에 카드/강조는 없다.
  const [selected, setSelected] = useState<Region>(regions[2]);
  // 지도 탐색용 선택과 카드 표시를 분리해 검색·영역 클릭으로 카드가 자동 노출되지 않게 한다.
  const [summaryOpen, setSummaryOpen] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [query, setQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [regionSort, setRegionSort] = useState<"rank" | "amount">("rank");
  const [tileError, setTileError] = useState(false);
  const detail = useRef<HTMLDialogElement>(null);
  // VITE_ 환경변수는 브라우저 번들에 포함된다. 서버 비밀키를 넣는 용도가 아니다.
  // VWorld 웹 키는 공급자 설정에서 허용 도메인·사용 범위를 제한해야 한다.
  const key = import.meta.env.VITE_VWORLD_API_KEY as string | undefined;
  const allRegions = useMemo(
    () => administrativeRegions.length ? [regions[0], ...administrativeRegions] : regions,
    [administrativeRegions],
  );
  const visible = useMemo(() => allRegions.filter(region => region.level === level), [allRegions, level]);
  const searchableRegions = allRegions.filter(region => region.level !== "시");
  const normalizedQuery = query.trim().replace(/\s+/g, "");
  const found = normalizedQuery
    ? searchableRegions.filter(region => `${region.parent}${region.name}`.replace(/\s+/g, "").includes(normalizedQuery))
    : searchableRegions;
  const sortedVisible = useMemo(() => [...visible].sort((left, right) => {
    if (regionSort === "amount") {
      const leftAmount = modelFor(left)?.apart_sale_amount ?? -Infinity;
      const rightAmount = modelFor(right)?.apart_sale_amount ?? -Infinity;
      return rightAmount - leftAmount || left.name.localeCompare(right.name, "ko");
    }
    const districtOf = (region: Region) => region.level === "구" ? region.name : region.parent;
    const rankOf = (region: Region) => displayedSafety?.forecasts.find(
      item => item.district === districtOf(region),
    )?.predicted_rank ?? Number.MAX_SAFE_INTEGER;
    return rankOf(left) - rankOf(right) || left.name.localeCompare(right.name, "ko");
  }), [visible, regionSort, recommender, displayedSafety]);
  const selectedSafety = selected.level === "구"
    ? displayedSafety?.forecasts.find(item => item.district === selected.name)
    : undefined;
  const selectedCategoryScore = selected.level === "구"
    ? categoryScores?.scores.find(item => item.district === selected.name)
    : undefined;
  const selectedModel = modelFor(selected);
  const selectedHousingSeries = useMemo(
    () => buildHousingSeries(selectedModel?.apart_sale ?? selected.price, selected.growth, selectedModel?.apart_rent ?? undefined),
    [selected.price, selected.growth, selectedModel?.apart_sale, selectedModel?.apart_rent],
  );

  useEffect(() => {
    const controller = new AbortController();
    setSafetyLoading(true);
    Promise.all([
      fetchCrimeSafety(controller.signal),
      fetchSafetyCategoryScores(controller.signal),
      fetchRecommenderForecasts(controller.signal),
    ])
      .then(([forecast, categories, modelForecasts]) => {
        setSafety(forecast);
        setCategoryScores(categories);
        setRecommender(modelForecasts);
        setSafetyError("");
      })
      .catch(error => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setSafetyError(error instanceof Error ? error.message : "지역점수를 불러오지 못했습니다.");
      })
      .finally(() => { if (!controller.signal.aborted) setSafetyLoading(false); });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      fetch(`${import.meta.env.BASE_URL}seoul-gu.geojson`, { signal: controller.signal }),
      fetch(`${import.meta.env.BASE_URL}seoul-dong.geojson`, { signal: controller.signal }),
    ])
      .then(async responses => {
        if (responses.some(response => !response.ok)) throw new Error("행정구역 목록 로딩 실패");
        return Promise.all(responses.map(response => response.json() as Promise<FeatureCollection>));
      })
      .then(([guData, dongData]) => {
        const center = (feature: FeatureCollection["features"][number]) => {
          const bounds = L.geoJSON(feature).getBounds();
          return bounds.getCenter();
        };
        const guRegions: Region[] = guData.features.map(feature => {
          const name = String(feature.properties?.name ?? feature.properties?.sggnm ?? "");
          const point = center(feature);
          return { id: `gu-${name}`, name, parent: "서울특별시", level: "구" as Level, lat: point.lat, lng: point.lng, price: 0, growth: 0 };
        }).filter(region => region.name);
        const dongRegions: Region[] = dongData.features.map(feature => {
          const fullName = String(feature.properties?.adm_nm ?? "");
          const district = String(feature.properties?.sggnm ?? fullName.split(" ")[1] ?? "");
          const name = fullName.split(" ").at(-1) ?? "";
          const point = center(feature);
          return { id: `dong-${String(feature.properties?.adm_cd2 ?? fullName)}`, name, parent: district, level: "행정동" as Level, lat: point.lat, lng: point.lng, price: 0, growth: 0 };
        }).filter(region => region.name && region.parent);
        setAdministrativeRegions([...guRegions, ...dongRegions]);
      })
      .catch(error => { if (!(error instanceof DOMException && error.name === "AbortError")) setSafetyError(String(error)); });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!container.current) return;
    // effect는 DOM 생성 후 실행된다. JSX 렌더 안에서 지도를 만들면 재렌더마다 중복 생성된다.
    // React StrictMode의 재마운트 시 지도·리스너를 정리해 중복 초기화를 방지한다.
    const instance = L.map(container.current, { zoomControl: false, minZoom: 9, maxZoom: 18 }).setView([37.5665, 126.978], 11);
    map.current = instance;
    const removeFocus = addSeoulFocus(instance);
    L.control.zoom({ position: "bottomright" }).addTo(instance);
    instance.attributionControl.addAttribution("가격·시설은 더미 데이터");
    if (key) {
      // WMTS 타일의 z/y/x 자리에 Leaflet이 줌·타일 행·열을 넣어 배경 이미지를 요청한다.
      // 키가 없어도 마커와 경계는 독립적으로 동작하도록 배경지도만 생략한다.
      const tiles = L.tileLayer(`https://api.vworld.kr/req/wmts/1.0.0/${encodeURIComponent(key)}/Base/{z}/{y}/{x}.png`, {
        attribution: '© <a href="https://www.vworld.kr" target="_blank" rel="noreferrer">VWorld</a>', maxZoom: 18,
      }).addTo(instance);
      tiles.on("tileerror", () => setTileError(true));
    }
    // 사이드바 토글로 지도 컨테이너 폭이 바뀌면 Leaflet의 내부 크기를 다시 계산한다.
    // 이를 생략하면 빈 영역이 남거나 화면상의 지도 중심이 어긋날 수 있다.
    const observer = new ResizeObserver(() => instance.invalidateSize());
    observer.observe(container.current);
    return () => { observer.disconnect(); removeFocus(); instance.remove(); map.current = null; };
  }, [key]);

  // 자동 모드에만 zoomend 리스너를 통해 표시 단위를 동기화한다.
  // 마지막 change()는 사용자가 자동 모드를 켠 순간의 현재 줌도 즉시 반영한다.
  useEffect(() => {
    const instance = map.current;
    if (!instance) return;
    const change = () => { if (auto) setLevel(atZoom(instance.getZoom())); };
    instance.on("zoomend", change);
    change();
    return () => { instance.off("zoomend", change); };
  }, [auto]);

  useEffect(() => {
    if (!map.current || visible.length === 0) return;
    return addAdministrativeBoundaries(
      map.current,
      level,
      summaryOpen && selected.level === level ? selected : null,
      visible,
      region => {
        const matched = visible.find(item => item.name === region.name && item.parent === region.parent);
        return matched ? housingPinLabel(matched, mode, months, safetyText(matched), modelFor(matched)) : region.name;
      },
      region => {
        const matched = visible.find(item => item.name === region.name && item.parent === region.parent);
        if (matched) { setSelected(matched); setSummaryOpen(true); }
      },
    );
  }, [level, visible, mode, months, summaryOpen, selected, displayedSafety, recommender]);

  useEffect(() => {
    if (!map.current || mode !== "지역점수" || !displayedSafety) return;
    return addSafetyGradeBoundaries(map.current, displayedSafety.forecasts);
  }, [mode, displayedSafety]);

  function choose(r: Region) {
    // 검색/목록 선택은 지도 위치만 안내한다. 카드와 음영은 가격 버튼을 눌러야 열린다.
    // flyTo는 부드럽게 이동시키며 수동 모드로 바꿔 줌 이동 중 단위 변경을 막는다.
    setSelected(r); setSummaryOpen(false); setLevel(r.level); setAuto(false); setQuery("");
    map.current?.flyTo([r.lat, r.lng], zoomFor[r.level]);
  }
  return <main className="housing-app">
    {/* 화면 구조: 헤더 → 사이드바와 지도 → 상세 dialog. Leaflet 캔버스 안은 React가 직접 그리지 않는다. */}
    <header className="topbar"><a className="brand" href="/"><span className="brand-symbol">⌂</span> SpotFit<span className="brand-sub">아이와 함께, 더 안전한 동네</span></a><span className="prototype">PROTOTYPE · 안전등급 모델 연동</span><a className="header-link" href="#region-list">지역 둘러보기 ↗</a></header>
    <div className={`workspace ${sidebarOpen ? "" : "sidebar-collapsed"}`}>
      <aside id="map-sidebar" className="sidebar" hidden={!sidebarOpen}>
        <div className="eyebrow">OUR NEXT NEIGHBORHOOD</div><h1>아이와 함께<br/>더 안전한 동네.</h1><p className="muted">주거 가격과 지역점수를<br/>하나의 지도에서 살펴보세요.</p>
        <div className="section-title">지역점수 영역 <span>복수 선택 가능</span></div>
        <div className="facility-grid safety-category-grid">{safetyCategories.map(category => {
          const active = selectedCategories.includes(category.key);
          return <button className={active ? "enabled" : ""} key={category.key} aria-pressed={active} onClick={() => setSelectedCategories(current => active ? (current.length === 1 ? current : current.filter(key => key !== category.key)) : [...current, category.key])}><span>{category.symbol}</span>{category.label}<small>{active ? `${categoryWeight.toFixed(selectedCategories.length === 3 ? 1 : 0)}%` : "미포함"}</small></button>;
        })}</div>
        <SafetyPanel data={displayedSafety} loading={safetyLoading} error={safetyError}/>
        {/* hidden은 패널을 숨기되 입력 상태는 유지한다. 검색·영역 선택·예측 기간을 여기서 조작한다. */}
        <label className="search-label" htmlFor="region-search">지역 검색 <small>25개 자치구 · 427개 행정동</small></label>
        <div className="search-box"><span>⌕</span><input id="region-search" value={query} onFocus={() => setSearchOpen(true)} onChange={e => { setQuery(e.target.value); setSearchOpen(true); }} placeholder="자치구 또는 행정동 검색" /><button aria-label="검색어 지우기" onClick={() => { setQuery(""); setSearchOpen(true); }}>×</button></div>
        {searchOpen && <div className="search-results"><div className="search-result-count">검색 결과 {found.length}개</div>{found.length ? found.map(r => <button key={r.id} onClick={() => { choose(r); setSearchOpen(false); }}>{r.parent} {r.name}<small>{r.level}</small></button>) : <p>일치하는 자치구 또는 행정동이 없습니다.</p>}</div>}
        <div className="section-title">예측 시점 <span>2026.10 기준 예시</span></div>
        <div className="segmented">{[12, 36].map(n => <button key={n} aria-pressed={months === n} className={months === n ? "chosen" : ""} onClick={() => setMonths(n)}>{n / 12}년 후</button>)}</div>
        <p className="muted">5년·10년 예측은 데이터 확보 후 개발 예정입니다.</p>
        <details className="region-accordion" id="region-list" open><summary><span>지도에 표시한 지역</span><small>{visible.length}개</small></summary><div className="region-sort" aria-label="지역 정렬 기준"><button className={regionSort === "rank" ? "chosen" : ""} aria-pressed={regionSort === "rank"} onClick={() => setRegionSort("rank")}>순위별</button><button className={regionSort === "amount" ? "chosen" : ""} aria-pressed={regionSort === "amount"} onClick={() => setRegionSort("amount")}>금액 높은 순</button></div><section className="region-list">{sortedVisible.map((r, index) => <button key={r.id} className={`region-item ${selected.id === r.id ? "selected" : ""}`} onClick={() => choose(r)}><span><strong>{regionSort === "rank" ? `${index + 1}. ` : ""}{r.name}</strong><small>{r.parent || "서울 전체"}{r.level === "구" ? ` · 지역점수 ${safetyText(r)}` : ""}</small></span><b>{regionSort === "amount" && modelFor(r)?.apart_sale_amount != null ? `${format(modelFor(r)!.apart_sale_amount!)}만` : displayLabel(r, mode, months)}<small>{regionSort === "amount" ? "대표 매매금액" : mode === "현재 가격" || mode === "예측 가격" ? "만원/평" : mode}</small></b></button>)}</section></details>
        <p className="sidebar-foot">가격·시설은 UI 예시입니다.<br/>안전등급은 공식 등급이 아닌 모델 기반 상대평가입니다.</p>
      </aside>
      <section className="map-shell" aria-label="서울 주거환경 지도">
        {/* 이 빈 div에 Leaflet이 타일·마커·경계 DOM을 삽입한다. 주변 도구 모음은 React가 관리한다. */}
        <div ref={container} className="map-canvas" />
        <div className="map-toolbar">
          {/* aria-expanded/pressed는 화면의 선택 상태를 보조기술에도 전달한다. */}
          {/* 같은 도구 모음 안에서 패널 제어와 지도 표시 모드를 구분하되 하나의 디자인으로 묶는다. */}
          <div className="map-primary-controls">
            <button className="sidebar-toggle" aria-expanded={sidebarOpen} aria-controls="map-sidebar" aria-label={sidebarOpen ? "사이드바 숨기기" : "사이드바 열기"} title={sidebarOpen ? "사이드바 숨기기" : "사이드바 열기"} onClick={() => setSidebarOpen(open => !open)}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="3"/><path d="M9 4v16"/><path d={sidebarOpen ? "m16 9-3 3 3 3" : "m13 9 3 3-3 3"}/></svg>
              <span>사이드바</span>
            </button>
            <div className="mode-tabs">{modes.map(m => <button key={m} aria-pressed={mode === m} className={mode === m ? "chosen" : ""} onClick={() => { setMode(m); if (m === "지역점수") { setSidebarOpen(true); setAuto(false); setLevel("구"); map.current?.setZoom(zoomFor["구"]); } }}>{m}</button>)}</div>
          </div>
          <div className="level-tabs"><button className={auto ? "chosen" : ""} aria-pressed={auto} onClick={() => setAuto(true)}>자동</button>{(["시","구","행정동"] as Level[]).map(l => <button key={l} className={level === l ? "chosen" : ""} onClick={() => { setAuto(false); setLevel(l); map.current?.setZoom(zoomFor[l]); }}>{l}</button>)}</div>
        </div>
        <div className="map-notice" role="status">{!key ? "VWorld 키 미설정 · 가격 버튼을 누르면 선택 지역을 강조합니다." : tileError ? "배경지도 로딩 실패 · 키 권한·등록 도메인·네트워크를 확인하세요." : "가격 버튼을 누르면 해당 지역을 녹색 음영으로 강조합니다 · 가격·시설은 가상 데이터입니다."}</div>
        {summaryOpen && <article className="summary-card" aria-live="polite"><div className="card-top"><span className="pill">{selected.level} · recommender v1</span><span className="muted">모델 결과</span><button className="summary-close" aria-label="지역 요약 닫기" onClick={() => setSummaryOpen(false)}>×</button></div><h2>{selected.name} <small>{selected.parent}</small></h2><p className="muted">{selected.level === "구" ? "하위 행정동 모델 예측값의 산술평균" : "최신 거래 특성 기준 행정동 대표가격 추정"}</p><div className="housing-forecast-row"><div><small>매매 평당금액</small><strong>{selectedModel?.apart_sale == null ? "-" : format(selectedModel.apart_sale)}<em>만원/평</em></strong></div><div><small>대표 매매금액</small><strong>{selectedModel?.apart_sale_amount == null ? "-" : format(selectedModel.apart_sale_amount)}<em>만원</em></strong></div><div><small>전세 평당금액</small><strong>{selectedModel?.apart_rent == null ? "-" : format(selectedModel.apart_rent)}<em>만원/평</em></strong></div></div>{selected.level === "구" && <div className="card-stats"><span>육아환경 3년 <b>{selectedModel?.child_environment_3y == null ? "-" : `${selectedModel.child_environment_3y.toFixed(1)}점`}</b></span><span>교통사고 예측 <b>{selectedModel?.traffic_accidents == null ? "-" : `${format(selectedModel.traffic_accidents)}건`}</b></span><span>지역점수 <b>{safetyText(selected)}</b></span></div>}{selectedSafety && <p className="estimate">선택영역 종합점수 <strong>{selectedSafety.predicted_score.toFixed(1)}</strong><small>{selectedCategories.length}개 영역 · 각 {categoryWeight.toFixed(selectedCategories.length === 3 ? 1 : 0)}% 반영</small></p>}<button className="detail-button" onClick={() => detail.current?.showModal()}>지역 자세히 보기 <span>↗</span></button></article>}
        <div className="legend"><strong>{mode} <span>· {level} · {mode === "지역점수" ? `${selectedCategories.length}개 영역 동일가중` : `${months / 12}년 후 선택`}</span></strong><small>{mode === "지역점수" ? "자치구 색상: 선택 영역 종합등급" : "녹색 음영: 선택한 지역"}</small><small>{mode === "지역점수" ? "마우스 오버: 종합등급·서울 순위" : mode === "예상 변동률" ? "마우스 오버: 변동률 (%)" : "마우스 오버: 가격 (만원/평)"} · 등급은 자치구 단위</small></div>
      </section>
    </div>
    <dialog ref={detail} className="detail-dialog detail-expanded"><form method="dialog"><button className="close-dialog" aria-label="상세 닫기">×</button></form><span className="eyebrow">RECOMMENDER V1</span><h2>{selected.name}</h2><p>v1 모델 결과와 화면용 추이 참고 그래프입니다.</p><div className="detail-charts"><HousingPriceChart series={selectedHousingSeries}/>{selected.level === "구" && <SafetyGradeChart score={selectedCategoryScore}/>}</div><p className="muted">매매·전세 값은 미래가격이 아니라 최신 거래 특성의 대표가격 추정입니다. 그래프의 연도별 가격 변화는 모델 예측이 아닌 화면용 참고 투영이며, 지역점수 등급은 서울 자치구 상대등급입니다.</p></dialog>
  </main>;
}
