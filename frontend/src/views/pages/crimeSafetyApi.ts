export type SafetyGrade = "A" | "B" | "C" | "D" | "E";
export type GradeDirection = "상승" | "유지" | "하락";
export type SafetyCategory = "traffic" | "crime" | "infrastructure";

export type CrimeSafetyForecast = {
  district: string;
  data_year: number;
  prediction_year: number;
  predicted_score: number;
  predicted_rank: number;
  predicted_grade: SafetyGrade;
  previous_rank: number;
  previous_grade: SafetyGrade;
  grade_change_expected: boolean;
  grade_change_direction: GradeDirection;
};

export type CrimeSafetyResponse = {
  model_name: string;
  model_version: string;
  algorithm: string;
  target: string;
  data_year: number;
  prediction_year: number;
  district_count: number;
  grade_definition: Record<SafetyGrade, string>;
  warning: string;
  forecasts: CrimeSafetyForecast[];
};

export type SafetyCategoryScore = {
  district: string;
  traffic_score: number;
  crime_score: number;
  infrastructure_score: number;
};

export type SafetyCategoryResponse = {
  district_count: number;
  source_years: Record<SafetyCategory, number>;
  placeholders: Record<SafetyCategory, boolean>;
  scores: SafetyCategoryScore[];
};

export async function fetchCrimeSafety(signal?: AbortSignal): Promise<CrimeSafetyResponse> {
  const response = await fetch("/api/safety/crime/district-forecasts", { signal });
  if (!response.ok) throw new Error(`지역안전지수 조회 실패 (${response.status})`);
  return response.json() as Promise<CrimeSafetyResponse>;
}

export async function fetchSafetyCategoryScores(signal?: AbortSignal): Promise<SafetyCategoryResponse> {
  const response = await fetch("/api/safety/crime/category-scores", { signal });
  if (!response.ok) throw new Error(`안전영역 점수 조회 실패 (${response.status})`);
  return response.json() as Promise<SafetyCategoryResponse>;
}

