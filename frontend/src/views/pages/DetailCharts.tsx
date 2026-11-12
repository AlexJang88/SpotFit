import type { SafetyCategoryScore } from "./crimeSafetyApi";
import "./detail-charts.css";

const YEARS = [2023, 2024, 2025, 2026, 2027, 2028, 2029];
const FORECAST_START = 2027;

export type HousingPoint = {
  year: number;
  sale: number;
  jeonse: number;
  monthly: number;
};

export function buildHousingSeries(basePrice: number, annualGrowth: number, jeonsePrice?: number): HousingPoint[] {
  return YEARS.map(year => {
    const sale = basePrice * Math.pow(1 + annualGrowth / 100, year - 2026);
    return {
      year,
      sale,
      jeonse: jeonsePrice == null
        ? sale * (0.54 + (year - 2026) * 0.004)
        : jeonsePrice * Math.pow(1 + annualGrowth / 100, year - 2026),
      monthly: sale * (0.0031 + (year - 2026) * 0.00003),
    };
  });
}

const categoryMeta = [
  { key: "traffic_score" as const, label: "교통사고", color: "#3478b8" },
  { key: "crime_score" as const, label: "범죄", color: "#c05b65" },
  { key: "infrastructure_score" as const, label: "인프라", color: "#3d9366" },
];

const priceMeta = [
  { key: "sale" as const, label: "매매", color: "#25664e" },
  { key: "jeonse" as const, label: "전세", color: "#497db5" },
  { key: "monthly" as const, label: "월세", color: "#d28745" },
];

const grade = (score: number) => (["E", "D", "C", "B", "A"] as const)[Math.min(4, Math.floor(score / 20))];
const fmt = (value: number) => Math.round(value).toLocaleString("ko-KR");

function points(values: number[], min: number, max: number, width = 600, height = 220) {
  const left = 48, top = 18, innerWidth = width - 68, innerHeight = height - 52;
  return values.map((value, index) => {
    const x = left + index * innerWidth / (values.length - 1);
    const y = top + (max - value) * innerHeight / Math.max(1, max - min);
    return `${x},${y}`;
  }).join(" ");
}

export function HousingPriceChart({ series }: { series: HousingPoint[] }) {
  const all = series.flatMap(row => [row.sale, row.jeonse, row.monthly]);
  const min = Math.min(...all) * 0.9, max = Math.max(...all) * 1.05;
  return <section className="detail-chart"><div className="chart-heading"><h3>아파트 가격 추이</h3><span>점선 이후는 더미 예측</span></div>
    <div className="chart-legend">{priceMeta.map(meta => <span key={meta.key}><i style={{background: meta.color}}/>{meta.label}</span>)}</div>
    <svg viewBox="0 0 600 220" role="img" aria-label="2023년부터 2029년까지 아파트 매매 전세 월세 가격 추이">
      {[0,1,2,3,4].map(i => <line key={i} x1="48" x2="580" y1={18+i*42} y2={18+i*42} className="chart-grid"/>)}
      <line x1="358" x2="358" y1="12" y2="188" className="forecast-divider"/>
      {priceMeta.map(meta => <polyline key={meta.key} points={points(series.map(row => row[meta.key]), min, max)} fill="none" stroke={meta.color} strokeWidth="3" strokeDasharray={meta.key === "monthly" ? "5 4" : undefined}/>) }
      {series.map((row, index) => <text key={row.year} x={48+index*532/6} y="211" textAnchor="middle" className={row.year >= FORECAST_START ? "future-year" : ""}>{row.year}</text>)}
    </svg>
    <div className="chart-latest">2029년 예상 · 매매 {fmt(series[6].sale)}만원/㎡ · 전세 {fmt(series[6].jeonse)}만원/㎡ · 월세 {fmt(series[6].monthly)}만원/㎡</div>
  </section>;
}

function buildSafetySeries(score: SafetyCategoryScore) {
  return YEARS.map(year => {
    const delta = year - 2026;
    return {
      year,
      traffic_score: Math.max(1, Math.min(100, score.traffic_score + delta * 1.8)),
      crime_score: Math.max(1, Math.min(100, score.crime_score + delta * 1.1)),
      infrastructure_score: Math.max(1, Math.min(100, score.infrastructure_score + delta * 2.2)),
    };
  });
}

export function SafetyGradeChart({ score }: { score: SafetyCategoryScore | undefined }) {
  if (!score) return <section className="detail-chart"><h3>안전영역 등급 추이</h3><p>자치구 데이터가 없습니다.</p></section>;
  const series = buildSafetySeries(score);
  return <section className="detail-chart"><div className="chart-heading"><h3>안전영역 등급 추이</h3><span>2027년부터 더미 예측</span></div>
    <div className="chart-legend">{categoryMeta.map(meta => <span key={meta.key}><i style={{background:meta.color}}/>{meta.label}</span>)}</div>
    <svg viewBox="0 0 600 220" role="img" aria-label="교통사고 범죄 인프라 등급의 과거 3년과 미래 3년 비교">
      {["A","B","C","D","E"].map((label,index) => <g key={label}><line x1="48" x2="580" y1={18+index*42} y2={18+index*42} className="chart-grid"/><text x="27" y={23+index*42}>{label}</text></g>)}
      <line x1="358" x2="358" y1="12" y2="188" className="forecast-divider"/>
      {categoryMeta.map(meta => <polyline key={meta.key} points={points(series.map(row => row[meta.key]), 0, 100)} fill="none" stroke={meta.color} strokeWidth="3"/>) }
      {series.map((row,index) => <g key={row.year}><text x={48+index*532/6} y="211" textAnchor="middle" className={row.year >= FORECAST_START ? "future-year" : ""}>{row.year}</text>{categoryMeta.map(meta => { const x=48+index*532/6; const value=row[meta.key]; const y=18+(100-value)*168/100; return <circle key={meta.key} cx={x} cy={y} r="4" fill={meta.color}><title>{row.year} {meta.label} {grade(value)}등급</title></circle>; })}</g>)}
    </svg>
    <div className="chart-latest">2029년 예상 · {categoryMeta.map(meta => `${meta.label} ${grade(series[6][meta.key])}`).join(" · ")}</div>
  </section>;
}
