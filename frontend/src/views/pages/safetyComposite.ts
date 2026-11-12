import type {
  CrimeSafetyResponse,
  SafetyCategory,
  SafetyCategoryResponse,
  SafetyGrade,
} from "./crimeSafetyApi";

const scoreField: Record<SafetyCategory, "traffic_score" | "crime_score" | "infrastructure_score"> = {
  traffic: "traffic_score",
  crime: "crime_score",
  infrastructure: "infrastructure_score",
};

const gradeForRank = (rank: number): SafetyGrade =>
  (["A", "B", "C", "D", "E"] as SafetyGrade[])[Math.min(4, Math.floor((rank - 1) / 5))];

export function buildCompositeSafety(
  base: CrimeSafetyResponse | null,
  categories: SafetyCategoryResponse | null,
  selected: SafetyCategory[],
): CrimeSafetyResponse | null {
  if (!base || !categories || selected.length === 0) return null;
  const previous = new Map(base.forecasts.map(item => [item.district, item]));
  const ranked = categories.scores
    .map(item => ({
      district: item.district,
      score: selected.reduce((sum, category) => sum + item[scoreField[category]], 0) / selected.length,
    }))
    .sort((a, b) => b.score - a.score || a.district.localeCompare(b.district, "ko"));

  return {
    ...base,
    algorithm: `동일가중 ${selected.length}개 영역`,
    target: "선택 안전영역 종합 백분위",
    prediction_year: Math.max(...selected.map(category => categories.source_years[category])),
    forecasts: ranked.map((item, index) => {
      const rank = index + 1;
      const old = previous.get(item.district);
      return {
        ...old,
        district: item.district,
        data_year: base.data_year,
        prediction_year: Math.max(...selected.map(category => categories.source_years[category])),
        predicted_score: Number(item.score.toFixed(4)),
        predicted_rank: rank,
        predicted_grade: gradeForRank(rank),
        previous_rank: old?.previous_rank ?? rank,
        previous_grade: old?.previous_grade ?? gradeForRank(rank),
        grade_change_expected: false,
        grade_change_direction: "유지",
      };
    }),
  };
}
