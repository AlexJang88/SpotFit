/** UI 실험용 점수 엔진. 원자료가 아닌 인구 대비 지표 X fixture를 사용한다.
 * 기준 집단의 통계는 최초 한 번 계산하고 사용자 선택/지도 검색으로 재계산하지 않는다. */
export const metrics = [
  { key: "daycare", name: "어린이집", weight: .2, unit: "0~5세 1천명당 개" },
  { key: "kindergarten", name: "유치원", weight: .15, unit: "3~5세 1천명당 개" },
  { key: "childcare_center", name: "보육정보센터", weight: .1, unit: "0~5세 1만명당 개" },
  { key: "park_area", name: "공원면적", weight: .1, unit: "주민 1명당 ㎡" },
  { key: "children_park_area", name: "어린이공원면적", weight: .1, unit: "0~5세 1천명당 ㎡" },
  { key: "library", name: "공공도서관", weight: .1, unit: "주민 1만명당 개" },
  { key: "department_store", name: "백화점", weight: .05, unit: "주민 100만명당 개" },
  { key: "public_health_center", name: "보건소", weight: .1, unit: "주민 10만명당 개" },
  { key: "clinic", name: "의원 수", weight: .1, unit: "주민 1천명당 개" },
] as const;
export type MetricKey = typeof metrics[number]["key"];
export type WeightMode = "weighted" | "equal";
export const dataVersion = "fixture-25gu-v1 (가상·실제 기준연도 없음)";
export const scoringVersion = "infra-z4-v1";
// 법정 시군구 코드 순으로 동점 표시 순서를 고정한다. 값은 실제 지역 평가와 무관하다.
const names = ["종로구","중구","용산구","성동구","광진구","동대문구","중랑구","성북구","강북구","도봉구","노원구","은평구","서대문구","마포구","양천구","강서구","구로구","금천구","영등포구","동작구","관악구","서초구","강남구","송파구","강동구"];
const codes = ["11110","11140","11170","11200","11215","11230","11260","11290","11305","11320","11350","11380","11410","11440","11470","11500","11530","11545","11560","11590","11620","11650","11680","11710","11740"];
export const fixture = names.map((name, i) => ({
  name, code: codes[i],
  // 단순한 결정적 값: 새로고침할 때마다 무작위 순위가 바뀌지 않도록 한다.
  values: metrics.map((_, j): number | null => ((i * (j + 3) + j * 7) % 29) + j / 10),
}));
export const grade = (x: number, mean: number, std: number) =>
  x < mean - std ? 25 : x < mean ? 50 : x < mean + std ? 75 : 100;
export function prepare(rows: typeof fixture) {
  return metrics.map((_, j) => {
    const valid = rows.map(r => r.values[j]).filter((v): v is number => v !== null && Number.isFinite(v) && v >= 0);
    const mean = valid.length ? valid.reduce((a,b) => a+b, 0) / valid.length : 0;
    const std = Math.sqrt(valid.reduce((a,b) => a + (b-mean)**2, 0) / (valid.length || 1));
    return { mean, std, n: valid.length, reason: valid.length < 2 ? "유효 지역 2개 미만" : std === 0 ? "모든 지역의 값이 같음" : null };
  });
}
const reference = prepare(fixture);
/** 지역별 결측 항목을 임의로 빼고 재가중하지 않는다. 공통 유효 항목 집합만 재가중한다. */
export function calculate(selected: MetricKey[], mode: WeightMode, rows = fixture, stats = reference) {
  if (!selected.length || new Set(selected).size !== selected.length || selected.some(k => !metrics.some(m => m.key === k))) throw new Error("항목을 중복 없이 1개 이상 선택하세요.");
  const chosen = metrics.map((m,j) => ({ ...m, j, ...stats[j] })).filter(m => selected.includes(m.key));
  const excluded = chosen.filter(m => m.reason);
  const effective = chosen.filter(m => !m.reason);
  if (!effective.length) throw new Error("계산 가능한 항목이 없습니다.");
  const denominator = effective.reduce((s,m) => s + (mode === "equal" ? 1 : m.weight), 0);
  const results = rows.map(r => {
    const details = effective.map(m => {
      const raw = r.values[m.j];
      const score = raw === null || !Number.isFinite(raw) || raw < 0 ? null : grade(raw, m.mean, m.std);
      const weight = (mode === "equal" ? 1 : m.weight) / denominator;
      return { ...m, raw, score, normalizedWeight: weight, contribution: score === null ? null : score * weight };
    });
    const missing = details.filter(d => d.score === null).map(d => d.name);
    return { ...r, details, missing, total: missing.length ? null : details.reduce((s,d) => s + d.contribution!, 0), rank: null as number | null };
  }).sort((a,b) => (b.total ?? -Infinity) - (a.total ?? -Infinity) || a.code.localeCompare(b.code));
  results.forEach((r,i) => { if (r.total !== null) r.rank = i > 0 && r.total === results[i-1].total ? results[i-1].rank : i+1; });
  return { results, excluded };
}
