import { apiRequest } from "../../services/apiClient";

export type Transaction = "sale" | "rent";
export type Grade = "A" | "B" | "C" | "D" | "E";
export type DistrictScore = {
  district: string; score: number; rank: number; grade: Grade;
  crime_score: number; traffic_score: number;
  crime_prediction_year: number; traffic_prediction_year: number;
  predicted_exposure_y: number;
};
export type Neighborhood = {
  id: string; district: string; neighborhood: string;
  apart_sale_amount: number | null; apart_sale_per_pyeong: number | null;
  apart_rent_amount: number | null; apart_rent_per_pyeong: number | null;
};
export type Catalog = {
  version: string; horizon: number; crime_base_year: number;
  crime_prediction_year: number; traffic_prediction_year: number;
  districts: DistrictScore[]; neighborhoods: Neighborhood[];
  property_models: Record<Transaction, { data_year: number; is_future_forecast: boolean; method: string }>;
  weights: { crime: number; traffic: number }; warnings: string[];
  score_method: string; tie_rule: string; crime_score_method: string;
  crime_score_formula: { feature: string; direction: "higher_risk" | "lower_risk"; weight_pct: number }[];
};
export type Recommendation = Neighborhood & {
  recommendation_rank: number; estimated_amount_manwon: number;
  district_score: DistrictScore; budget_remaining_manwon: number; price_basis: string;
};
export type Recommendations = {
  recommendations: Recommendation[]; candidate_count: number;
  excluded_missing_count: number; budget_manwon: number; transaction: Transaction;
  area_m2: number | null; warnings: string[]; sort_rule: string;
};
export type AreaPresets = {
  transaction: Transaction; basis: "transactions" | "default" | "unavailable";
  data_year: number | null; description: string;
  options: { label: string; area_m2: number; transaction_count: number | null }[];
};

export function fetchAreaPresets(transaction: Transaction, signal?: AbortSignal) {
  return read<AreaPresets>(`/api/neighborhoods/area-presets?transaction=${transaction}`, signal);
}

async function read<T>(url: string, signal?: AbortSignal): Promise<T> {
  return apiRequest<T>(url, { signal, fallbackMessage: "데이터를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요." });
}

export function fetchCatalog(horizon: number, signal?: AbortSignal) {
  return read<Catalog>(`/api/neighborhoods/catalog?horizon=${horizon}`, signal);
}

export function fetchRecommendations(budget: number, transaction: Transaction, horizon: number,
  area: number | null, signal?: AbortSignal, minBudget = 0) {
  const query = new URLSearchParams({ budget_manwon: String(budget), transaction, horizon: String(horizon) });
  if (area !== null) query.set("area_m2", String(area));
  query.set("min_budget_manwon", String(minBudget));
  return read<Recommendations>(`/api/neighborhoods/recommendations?${query}`, signal);
}

export const normalizeDong = (value: string) => value.replace(/\s+/g, "").replace(/제+(?=\d)/g, "").replace(/(?<=\d)[.·ㆍ](?=\d)/g, "·");
export const amountFor = (row: Neighborhood | undefined, transaction: Transaction, area: number | null) => {
  if (!row) return null;
  const perPyeong = transaction === "sale" ? row.apart_sale_per_pyeong : row.apart_rent_per_pyeong;
  const amount = transaction === "sale" ? row.apart_sale_amount : row.apart_rent_amount;
  return area !== null ? (perPyeong === null ? null : perPyeong * area / (400 / 121)) : amount;
};
