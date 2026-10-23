import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import { addSeoulFocus } from "./seoulFocus";
import { addAdministrativeBoundaries } from "./administrativeBoundaries";
import { CarePanel } from "./CarePanel";
import { calculate, metrics, type MetricKey, type WeightMode } from "./careScoring";
import "leaflet/dist/leaflet.css";
import "./housing-map.css";

/**
 * React는 화면 상태를, Leaflet은 지도 DOM과 레이어를 관리한다.
 * 상태 변경 → effect로 지도 갱신 → cleanup으로 이전 레이어 제거 순서로 동작한다.
 * 현재 공개 경계 외 가격·점수·시설은 예시이며 백엔드 추론 API는 연결하지 않았다.
 */
type Level = "시" | "구" | "행정동";
type Mode = "현재 가격" | "예측 가격" | "예상 변동률" | "양육환경";
type Facility = "학교" | "어린이집" | "공원" | "의료";
// id는 UI 식별자(행정코드 아님), parent는 상위 지역명, lat/lng는 버튼의 위도/경도다.
// price는 만원/㎡, growth는 연간 변동률(%), care는 100점 기준 가상 양육환경 점수다.
type Region = { id: string; name: string; parent: string; level: Level; lat: number; lng: number; price: number; growth: number; care: number };
// 가격 단위는 만원/㎡. 실제 거래·행정 경계·시설 정보가 아닌 화면 검토용 고정 fixture다.
const regions: Region[] = [
  { id: "seoul", name: "서울특별시", parent: "", level: "시", lat: 37.5665, lng: 126.978, price: 1450, growth: 3.2, care: 78 },
  ...[
    ["mapo", "마포구", 37.563, 126.913, 1500, 3.8, 84],
    ["seongdong", "성동구", 37.563, 127.037, 1780, 4.5, 88],
    ["songpa", "송파구", 37.505, 127.106, 1910, 2.8, 91],
    ["nowon", "노원구", 37.654, 127.056, 980, -1.2, 86],
    ["gangseo", "강서구", 37.55, 126.849, 1120, 2.1, 80],
  ].map(([id, name, lat, lng, price, growth, care]) => ({ id: String(id), name: String(name), parent: "서울특별시", level: "구" as Level, lat: Number(lat), lng: Number(lng), price: Number(price), growth: Number(growth), care: Number(care) })),
  ...[
    ["seongsu", "성수1가1동", "성동구", 37.542, 127.049, 1920, 4.8, 89],
    ["haengdang", "행당1동", "성동구", 37.558, 127.036, 1640, 3.2, 92],
    ["sangam", "상암동", "마포구", 37.578, 126.893, 1460, 2.5, 87],
    ["yeonnam", "연남동", "마포구", 37.565, 126.923, 1390, 1.8, 76],
    ["jamsil", "잠실3동", "송파구", 37.509, 127.093, 2180, 3.4, 94],
    ["gongneung", "공릉1동", "노원구", 37.623, 127.073, 940, -0.8, 83],
    ["balsan", "발산1동", "강서구", 37.553, 126.838, 1240, 2.6, 88],
  ].map(([id, name, parent, lat, lng, price, growth, care]) => ({ id: String(id), name: String(name), parent: String(parent), level: "행정동" as Level, lat: Number(lat), lng: Number(lng), price: Number(price), growth: Number(growth), care: Number(care) })),
];
const modes: Mode[] = ["현재 가격", "예측 가격", "예상 변동률", "양육환경"];
const types: Facility[] = ["학교", "어린이집", "공원", "의료"];
const symbols = ["학", "돌", "숲", "+"];
// 수동 단위 선택 시 사용할 확대 수준. atZoom은 자동 모드에서 반대로 단위를 구한다.
// 11 미만: 시, 11~12: 구, 13 이상: 행정동. 행정 기준이 아닌 화면 표시 기준이다.
const zoomFor: Record<Level, number> = { 시: 10, 구: 11, 행정동: 13 };
const atZoom = (z: number): Level => z < 11 ? "시" : z < 13 ? "구" : "행정동";
const format = (n: number) => Math.round(n).toLocaleString("ko-KR");
// 1200 = 100(백분율) × 12(월→년). 연 3%를 36개월 적용하면 가격 × 1.09다.
// 복리나 머신러닝 추론이 아닌 기간 비례 계산이며 API 연결 시 교체할 부분이다.
const forecast = (r: Region, months: number) => r.price * (1 + r.growth * months / 1200);
const value = (r: Region, mode: Mode, months: number) => mode === "현재 가격" ? r.price : mode === "예측 가격" ? forecast(r, months) : mode === "예상 변동률" ? r.growth * months / 12 : r.care;
const label = (r: Region, mode: Mode, months: number) => `${mode === "예상 변동률" ? value(r, mode, months).toFixed(1) : format(value(r, mode, months))}${mode === "양육환경" ? "점" : mode === "예상 변동률" ? "%" : "만"}`;

export function HomePage() {
  // 선택 기반 점수는 구 단위만 제공한다. 기존 가상 care 필드 대신 이 결과를 표시한다.
  const [selectedMetrics, setSelectedMetrics] = useState<MetricKey[]>(metrics.map(m => m.key));
  const [weightMode, setWeightMode] = useState<WeightMode>("weighted");
  const careResults = calculate(selectedMetrics, weightMode).results;
  const careText = (r: Region) => {
    if (r.level !== "구") return "자치구만 평가";
    const score = careResults.find(item => item.name === r.name)?.total;
    return score == null ? "데이터 부족" : score.toFixed(2) + "점";
  };
  // 지도 마커와 목록도 같은 선택 기반 점수를 사용해 추천 패널과 표시가 다르지 않게 한다.
  const displayLabel = (r: Region, m: Mode, n: number) => m === "양육환경" ? careText(r) : label(r, m, n);
  // ref는 렌더 사이에 DOM/지도 인스턴스를 보존한다. 변경해도 React 재렌더를 유발하지 않는다.
  // 반면 useState 값은 버튼·카드·지도 갱신에 필요하므로 변경 시 재렌더를 유발한다.
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<L.Map | null>(null);
  const [level, setLevel] = useState<Level>("구");
  // auto=false일 때는 줌이 변해도 사용자가 직접 선택한 지역 단위를 유지한다.
  const [auto, setAuto] = useState(true);
  const [mode, setMode] = useState<Mode>("예측 가격");
  const [months, setMonths] = useState(12);
  // selected에는 기본 지역이 있지만 summaryOpen=false여서 첫 화면에 카드/강조는 없다.
  const [selected, setSelected] = useState<Region>(regions[2]);
  // 지도 탐색용 선택과 카드 표시를 분리해 검색·영역 클릭으로 카드가 자동 노출되지 않게 한다.
  const [summaryOpen, setSummaryOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [enabled, setEnabled] = useState<Facility[]>(["학교", "공원"]);
  const [tileError, setTileError] = useState(false);
  const detail = useRef<HTMLDialogElement>(null);
  // VITE_ 환경변수는 브라우저 번들에 포함된다. 서버 비밀키를 넣는 용도가 아니다.
  // VWorld 웹 키는 공급자 설정에서 허용 도메인·사용 범위를 제한해야 한다.
  const key = import.meta.env.VITE_VWORLD_API_KEY as string | undefined;
  const visible = regions.filter(r => r.level === level);
  // 검색은 등록된 더미 지역의 부분 문자열 검색이다. 주소 검색 API나 서울 전체 검색이 아니다.
  const found = regions.filter(r => `${r.parent} ${r.name}`.includes(query.trim()));

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

  // 가격 버튼/시설 레이어만 재생성하며 배경지도는 유지한다.
  // 의존 배열의 값이 바뀌면 cleanup이 먼저 실행되어 중복 마커와 이벤트 참조를 정리한다.
  useEffect(() => {
    const instance = map.current;
    if (!instance) return;
    const layer = L.layerGroup().addTo(instance);
    visible.forEach(r => {
      // 실제 경계는 별도 레이어로 표시하고 가격 버튼만 더미 지역 좌표에 배치한다.
      const button = document.createElement("button");
      // Leaflet divIcon은 실제 DOM을 받는다. React JSX 버튼과 달리 onclick을 직접 연결한다.
      // textContent는 지역명을 HTML로 해석하지 않아 외부 데이터 연결 시에도 안전한 기본값이다.
      button.className = `price-pin ${summaryOpen && r.id === selected.id ? "active" : ""}`;
      button.textContent = `${r.name} · ${displayLabel(r, mode, months)}`;
      button.setAttribute("aria-label", `${r.name} 요약 보기`);
      // 명시적으로 가격 버튼을 눌렀을 때만 해당 지역 요약을 연다.
      button.onclick = () => { setSelected(r); setSummaryOpen(true); };
      L.marker([r.lat, r.lng], { icon: L.divIcon({ html: button, className: "pin-container", iconSize: [150, 38], iconAnchor: [75, 19] }) }).addTo(layer);
    });
    // 모든 시설도 고정된 가상 위치다. 시설 수나 접근성 계산 근거로 사용하지 않는다.
    regions.filter(r => r.level === "행정동").forEach(r => {
      types.forEach((type, i) => {
        if (!enabled.includes(type)) return;
        const marker = L.circleMarker([r.lat + .002 * (i - 1), r.lng + .003 * (i - 1)], { radius: 7, color: "#fff", weight: 2, fillColor: ["#467ab1", "#d99a47", "#40876a", "#c87583"][i], fillOpacity: 1 }).addTo(layer);
        const text = document.createElement("span"); text.textContent = `[가상 시설] ${r.name} ${type}`;
        marker.bindTooltip(text);
      });
    });
    return () => { layer.remove(); };
  }, [level, mode, months, selected.id, enabled, summaryOpen, selectedMetrics, weightMode]);

  useEffect(() => {
    // 카드와 강조 영역을 같은 상태로 관리해 선택 변경·닫기 시 함께 반영한다.
    if (map.current) return addAdministrativeBoundaries(map.current, level, summaryOpen && selected.level === level ? selected : null);
  }, [level, key, summaryOpen, selected]);

  // 확대/축소로 지역 단위가 바뀌면 이전 단위의 카드가 지도에 남지 않도록 닫는다.
  useEffect(() => { setSummaryOpen(false); }, [level]);

  function choose(r: Region) {
    // 검색/목록 선택은 지도 위치만 안내한다. 카드와 음영은 가격 버튼을 눌러야 열린다.
    // flyTo는 부드럽게 이동시키며 수동 모드로 바꿔 줌 이동 중 단위 변경을 막는다.
    setSelected(r); setSummaryOpen(false); setLevel(r.level); setAuto(false); setQuery("");
    map.current?.flyTo([r.lat, r.lng], zoomFor[r.level]);
  }
  return <main className="housing-app">
    {/* 화면 구조: 헤더 → 사이드바와 지도 → 상세 dialog. Leaflet 캔버스 안은 React가 직접 그리지 않는다. */}
    <header className="topbar"><a className="brand" href="/"><span className="brand-symbol">⌂</span> SpotFit<span className="brand-sub">아이와 함께, 더 좋은 동네</span></a><span className="prototype">PROTOTYPE · 더미 데이터</span><a className="header-link" href="#region-list">지역 둘러보기 ↗</a></header>
    <div className={`workspace ${sidebarOpen ? "" : "sidebar-collapsed"}`}>
      <aside id="map-sidebar" className="sidebar" hidden={!sidebarOpen}>
        <CarePanel selected={selectedMetrics} mode={weightMode} onSelect={setSelectedMetrics} onMode={setWeightMode}/>
        {/* hidden은 패널을 숨기되 입력 상태는 유지한다. 검색·시설 선택·예측 기간을 여기서 조작한다. */}
        <div className="eyebrow">OUR NEXT NEIGHBORHOOD</div><h1>아이의 내일이<br/>자라는 동네.</h1><p className="muted">주거 가격과 양육환경을<br/>하나의 지도에서 살펴보세요.</p>
        <label className="search-label" htmlFor="region-search">어떤 지역이 궁금하세요?</label>
        <div className="search-box"><span>⌕</span><input id="region-search" value={query} onChange={e => setQuery(e.target.value)} placeholder="서울, 마포구, 상암동 검색" /><button aria-label="검색어 지우기" onClick={() => setQuery("")}>×</button></div>
        {query.trim() && <div className="search-results">{found.length ? found.map(r => <button key={r.id} onClick={() => choose(r)}>{r.parent} {r.name}<small>{r.level}</small></button>) : <p>등록된 더미 지역이 없습니다.</p>}</div>}
        <div className="section-title">예측 시점 <span>2026.10 기준 예시</span></div>
        <div className="segmented">{[12, 36].map(n => <button key={n} aria-pressed={months === n} className={months === n ? "chosen" : ""} onClick={() => setMonths(n)}>{n / 12}년 후</button>)}</div>
        <p className="muted">5년·10년 예측은 데이터 확보 후 개발 예정입니다.</p><div className="section-title">우리 아이 생활시설 <span>가상 위치</span></div>
        <div className="facility-grid">{types.map((t, i) => <button className={enabled.includes(t) ? "enabled" : ""} key={t} aria-pressed={enabled.includes(t)} onClick={() => setEnabled(v => v.includes(t) ? v.filter(x => x !== t) : [...v, t])}><span>{symbols[i]}</span>{t}</button>)}</div>
        <section className="region-list" id="region-list"><div className="section-title">지도에 표시한 지역 <span>{visible.length}개 예시</span></div>{visible.map(r => <button key={r.id} className={`region-item ${selected.id === r.id ? "selected" : ""}`} onClick={() => choose(r)}><span><strong>{r.name}</strong><small>{r.parent || "서울 전체"} · 양육환경 {careText(r)}</small></span><b>{displayLabel(r, mode, months)}<small>{mode === "현재 가격" || mode === "예측 가격" ? "만원/㎡" : mode}</small></b></button>)}</section>
        <p className="sidebar-foot">실거래·모델 연결 전 UI 초안입니다.<br/>표시된 수치는 실제 지역 평가가 아닙니다.</p>
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
            <div className="mode-tabs">{modes.map(m => <button key={m} aria-pressed={mode === m} className={mode === m ? "chosen" : ""} onClick={() => { setMode(m); if (m === "양육환경") { setSidebarOpen(true); setAuto(false); setLevel("구"); map.current?.setZoom(zoomFor["구"]); } }}>{m}</button>)}</div>
          </div>
          <div className="level-tabs"><button className={auto ? "chosen" : ""} aria-pressed={auto} onClick={() => setAuto(true)}>자동</button>{(["시","구","행정동"] as Level[]).map(l => <button key={l} className={level === l ? "chosen" : ""} onClick={() => { setAuto(false); setLevel(l); map.current?.setZoom(zoomFor[l]); }}>{l}</button>)}</div>
        </div>
        <div className="map-notice" role="status">{!key ? "VWorld 키 미설정 · 가격 버튼을 누르면 선택 지역을 강조합니다." : tileError ? "배경지도 로딩 실패 · 키 권한·등록 도메인·네트워크를 확인하세요." : "가격 버튼을 누르면 해당 지역을 녹색 음영으로 강조합니다 · 가격·시설은 가상 데이터입니다."}</div>
        {summaryOpen && <article className="summary-card" aria-live="polite"><div className="card-top"><span className="pill">{selected.level} · 가상 데이터</span><span className="muted">기준 2026.10</span><button className="summary-close" aria-label="지역 요약 닫기" onClick={() => setSummaryOpen(false)}>×</button></div><h2>{selected.name} <small>{selected.parent}</small></h2><p className="muted">아파트 · ㎡당 대표가격 예시</p><div className="price-row"><div><small>현재 가격</small><strong>{format(selected.price)}<em>만원/㎡</em></strong></div><span>→</span><div><small>{months / 12}년 후 예상</small><strong className="green">{format(forecast(selected,months))}<em>만원/㎡</em></strong></div></div><div className="card-stats"><span>예상 변동률 <b>{(selected.growth * months / 12).toFixed(1)}%</b></span><span>양육환경 <b>{careText(selected)}</b></span></div><p className="estimate">전용 84㎡ 단순 환산 <strong>{(forecast(selected,months)*84/10000).toFixed(2)}억원</strong><small>개별 매물 가격이 아닌 화면용 계산 예시</small></p><button className="detail-button" onClick={() => detail.current?.showModal()}>지역 자세히 보기 <span>↗</span></button></article>}
        <div className="legend"><strong>{mode} <span>· {level} · {months / 12}년 후 선택</span></strong><small>녹색 음영: 선택한 지역 (가격 크기와 무관)</small><small>{mode === "양육환경" ? "마커: 양육환경 점수 / 100" : mode === "예상 변동률" ? "마커: 변동률 (%)" : "마커: 가격 (만원/㎡)"} · 구·행정동 경계 2026.10 기준</small></div>
      </section>
    </div>
    <dialog ref={detail} className="detail-dialog"><form method="dialog"><button className="close-dialog" aria-label="상세 닫기">×</button></form><span className="eyebrow">NEIGHBORHOOD PREVIEW</span><h2>{selected.name}</h2><p>가격 예측·양육환경 상세 화면의 연결 예시입니다.</p><dl><dt>현재 대표가격</dt><dd>{format(selected.price)}만원/㎡</dd><dt>{months / 12}년 후 가격</dt><dd>{format(forecast(selected,months))}만원/㎡</dd><dt>양육환경</dt><dd>{careText(selected)} / 선택 기반 가상 평가</dd><dt>예측 근거</dt><dd>고정 더미 변동률을 적용한 단순 계산입니다. 학습 모델·실거래·시설 통계와 연결되지 않았습니다.</dd></dl><p className="muted">가격·시설은 더미 데이터이며 실제 부동산 의사결정에 사용할 수 없습니다.</p></dialog>
  </main>;
}
