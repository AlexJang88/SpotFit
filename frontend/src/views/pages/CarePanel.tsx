import { useState } from "react";
import { calculate, metrics, dataVersion, scoringVersion, type MetricKey, type WeightMode } from "./careScoring";
import "./care-panel.css";

/** 체크박스는 지도 시설 표시와 별개다. 선택은 점수/추천만 바꾸며 마커 표시를 바꾸지 않는다. */
export function CarePanel({ selected, mode, onSelect, onMode }: {
  selected: MetricKey[]; mode: WeightMode; onSelect: (keys: MetricKey[]) => void; onMode: (mode: WeightMode) => void;
}) {
  const [expanded, setExpanded] = useState<string | null>(null);
  let result: ReturnType<typeof calculate> | undefined;
  let error = "";
  try { result = calculate(selected, mode); } catch (e) { error = e instanceof Error ? e.message : "계산 오류"; }
  return <section className="care-panel" aria-label="맞춤 육아환경 추천">
    <h2>나에게 맞는 육아환경</h2>
    <p className="care-warning">25개 자치구 가상 데이터 · 실제 추천이 아닌 UI 시연입니다.</p>
    <fieldset><legend>중요한 항목 선택 ({selected.length}/9)</legend>
      {metrics.map(m => <label key={m.key}><input type="checkbox" checked={selected.includes(m.key)}
        disabled={selected.length === 1 && selected.includes(m.key)}
        onChange={() => onSelect(selected.includes(m.key) ? selected.filter(k => k !== m.key) : [...selected, m.key])}/>{m.name}<small>{m.weight * 100}%</small></label>)}
    </fieldset>
    <p className="muted">최소 1개 유지 · 시설 표시 토글과 별개입니다.</p>
    <label>점수 합산 방식 <select value={mode} onChange={e => onMode(e.target.value as WeightMode)}><option value="weighted">기본 중요도</option><option value="equal">동일 중요도</option></select></label>
    <p className="muted">선택 항목의 가중치만 다시 합계 100%로 환산합니다. 25~100점 상대평가이며 선택 조건이 다른 점수끼리는 직접 비교할 수 없습니다. 접근 거리·정원·품질은 평가하지 않습니다.</p>
    <small>{scoringVersion}<br/>{dataVersion}<br/>기준 집단: 서울 25개 자치구 고정</small>
    {error && <p role="alert">{error}</p>}
    {result?.excluded.map(m => <p key={m.key}>제외: {m.name} — {m.reason}</p>)}
    <h3>선택 조건별 추천 순위</h3>
    <p className="muted" role="status">선택 {selected.length}개 · {mode === "weighted" ? "기본" : "동일"} 중요도로 계산</p>
    <div className="care-ranking">{result?.results.map(r => <div key={r.code}>
      <button className="care-row" aria-expanded={expanded === r.code} onClick={() => setExpanded(expanded === r.code ? null : r.code)}>
        <span>{r.rank === null ? "—" : r.rank + "위"} · {r.name}</span><strong>{r.total === null ? "데이터 부족" : r.total.toFixed(2) + "점"}</strong>
      </button>
      {expanded === r.code && <div className="care-breakdown">
        {r.missing.length > 0 && <p>누락: {r.missing.join(", ")}</p>}
        {r.details.map(d => <div key={d.key}><strong>{d.name}: {d.score ?? "데이터 부족"}점</strong>
          <p>적용 {(d.normalizedWeight*100).toFixed(2)}% · 기여 {d.contribution?.toFixed(2) ?? "—"}점</p>
          <small>가상 지표 {d.raw ?? "누락"} {d.unit}<br/>평균 {d.mean.toFixed(2)} · 표준편차 {d.std.toFixed(2)} · 유효 {d.n}구</small>
        </div>)}
      </div>}
    </div>)}</div>
    <p className="muted">공원과 어린이공원은 일부 중복될 수 있습니다. 기본 가중치는 프로젝트 설계값이며 논문의 AHP 결과가 아닙니다.</p>
  </section>;
}
