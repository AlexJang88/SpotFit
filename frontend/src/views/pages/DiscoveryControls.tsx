import type { CSSProperties, PointerEvent } from "react";

export type Panel = "search" | "transaction" | "budget" | "area" | "year" | "info";

export function ControlIcon({ name }: { name: Panel | "locate" }) {
  const paths = {
    search: <><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/></>,
    transaction: <><path d="m3 10 9-7 9 7M6 9v12h12V9M10 21v-7h4v7"/></>,
    budget: <><rect x="3" y="5" width="18" height="15" rx="3"/><path d="M3 9h18M16 14h5"/></>,
    area: <><rect x="4" y="4" width="16" height="16" rx="2"/><path d="M8 4v4M12 4v3M16 4v4M4 12h4M4 16h3"/></>,
    year: <><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M7 3v4M17 3v4M3 11h18M8 15h2M14 15h2"/></>,
    info: <><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7v1"/></>,
    locate: <><circle cx="12" cy="12" r="6"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/></>,
  };
  return <svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

export const budgetLabel = (amount: number) => amount >= 10000
  ? `${(amount / 10000).toLocaleString("ko-KR", { maximumFractionDigits: 2 })}억원`
  : `${amount.toLocaleString("ko-KR")}만원`;

export function BudgetRange({ lower, upper, onLower, onUpper }: {
  lower: string; upper: string; onLower: (value: string) => void; onUpper: (value: string) => void;
}) {
  const min = Math.max(0, Number(lower) || 0);
  const max = Math.max(min, Number(upper) || 0);
  const ceiling = Math.max(300000, Math.ceil(max / 100000) * 100000);
  const setAtPointer = (event: PointerEvent<HTMLDivElement>, thumb?: "lower" | "upper") => {
    const rect = event.currentTarget.getBoundingClientRect();
    const value = Math.round(Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)) * ceiling / 1000) * 1000;
    const which = thumb ?? (Math.abs(value - min) <= Math.abs(value - max) ? "lower" : "upper");
    if (which === "lower") onLower(String(Math.min(value, max)));
    else onUpper(String(Math.max(1000, value, min)));
    return which;
  };
  return <div className="budget-range">
    <div className="budget-range-value">{budgetLabel(min)} <span>—</span> {budgetLabel(max)}</div>
    <div className="range-track" style={{ "--range-start": `${min / ceiling * 100}%`, "--range-end": `${max / ceiling * 100}%` } as CSSProperties}
      onPointerDown={e => {
        if ((e.target as HTMLElement).tagName === "INPUT") return;
        e.currentTarget.dataset.thumb = setAtPointer(e);
        e.currentTarget.setPointerCapture(e.pointerId);
      }}
      onPointerMove={e => { if (e.currentTarget.hasPointerCapture(e.pointerId)) setAtPointer(e, e.currentTarget.dataset.thumb as "lower" | "upper"); }}>
      <div className="range-rail"/><div className="range-fill"/>
      <input className="range-lower" type="range" aria-label="예산 하한 슬라이더" aria-valuetext={budgetLabel(min)} min="0" max={ceiling} step="1000" value={min} onChange={e => onLower(String(Math.min(Number(e.target.value), max)))}/>
      <input className="range-upper" type="range" aria-label="예산 상한 슬라이더" aria-valuetext={budgetLabel(max)} min="0" max={ceiling} step="1000" value={max} onChange={e => onUpper(String(Math.max(1000, Number(e.target.value), min)))}/>
    </div>
    <div className="range-scale"><span>0원</span><span>{budgetLabel(ceiling)}</span></div>
    <div className="budget-inputs">
      <div className="filter-field"><label htmlFor="budget-min">최소 금액</label><div className="input-unit"><input id="budget-min" type="number" min="0" max="1000000000" step="any" value={lower} onChange={e => onLower(e.target.value)}/><span>만원</span></div></div>
      <span aria-hidden="true">—</span>
      <div className="filter-field"><label htmlFor="budget">최대 금액</label><div className="input-unit"><input id="budget" type="number" min="1" max="1000000000" step="any" value={upper} onChange={e => onUpper(e.target.value)}/><span>만원</span></div></div>
    </div>
    <p className="field-hint">두 손잡이를 움직이거나 금액을 직접 입력하세요. 슬라이더는 1천만원 단위입니다.</p>
  </div>;
}
