import { apiRequest } from "../../services/apiClient";

export type Transaction = "sale" | "rent" | "wolse";
export type ScoreWeights = { crime: number; traffic: number };
export const transactionLabel: Record<Transaction, string> = { sale: "매매", rent: "전세", wolse: "월세" };
export const M2_PER_PYEONG = 400 / 121;
export type MonthlyQuote = {
  available: boolean; monthly_rent_manwon: number | null; deposit_manwon?: number;
  p25_manwon?: number | null; p75_manwon?: number | null; conversion_rate_pct?: number | null;
  area_band?: string; buildings_used?: number; reason?: string | null; trained_until?: string;
};
export type Grade = "A" | "B" | "C" | "D" | "E";
export type DistrictScore = {
  district: string; score: number; rank: number; grade: Grade;
  crime_score: number; traffic_score: number;
  crime_prediction_year: number; traffic_prediction_year: number;
  predicted_exposure_y: number;
  score_series?: { year: number; score: number; kind: "forecast" }[];
  historical_reference_series?: { year: number; score: number; kind: "historical_reference" }[];
};
export type Neighborhood = {
  id: string; district: string; neighborhood: string;
  apart_sale_amount: number | null; apart_sale_per_pyeong: number | null;
  apart_rent_amount: number | null; apart_rent_per_pyeong: number | null;
  apart_sale_price_series?: { year: number; per_pyeong_manwon: number | null; kind: "observed" | "forecast" }[];
  apart_rent_price_series?: { year: number; per_pyeong_manwon: number | null; kind: "observed" | "forecast" }[];
  monthly_rent_history?: { year: number; monthly_rent_manwon: number; transaction_count: number; kind: "observed" }[];
  wolse?: MonthlyQuote;
};
export type Catalog = {
  version: string; horizon: number; crime_base_year: number;
  transaction: Transaction; deposit_manwon: number | null; area_m2: number | null;
  crime_prediction_year: number; traffic_prediction_year: number;
  districts: DistrictScore[]; neighborhoods: Neighborhood[];
  property_models: Partial<Record<Transaction, { data_year: number; is_future_forecast: boolean; method: string; prediction_year?: number; trained_until?: string }>>;
  weights: ScoreWeights; warnings: string[];
  score_method: string; tie_rule: string; crime_score_method: string;
  crime_score_formula: { feature: string; direction: "higher_risk" | "lower_risk"; weight_pct: number }[];
};
export type Recommendation = Neighborhood & {
  recommendation_rank: number; estimated_amount_manwon: number;
  district_score: DistrictScore; budget_remaining_manwon: number; price_basis: string;
};
export type Recommendations = {
  weights: ScoreWeights;
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

export function fetchCatalog(horizon: number, signal?: AbortSignal, transaction: Transaction = "sale", deposit?: number, area?: number | null, crimePercent = 50) {
  const query = new URLSearchParams({ horizon: String(horizon), transaction });
  query.set("crime_weight", String(crimePercent / 100));
  query.set("traffic_weight", String((100 - crimePercent) / 100));
  if (transaction === "wolse") {
    if (deposit !== undefined) query.set("deposit_manwon", String(deposit));
    if (area != null) query.set("area_m2", String(area));
  }
  return read<Catalog>(`/api/neighborhoods/catalog?${query}`, signal);
}

export function fetchRecommendations(budget: number, transaction: Transaction, horizon: number,
  area: number | null, signal?: AbortSignal, minBudget = 0, deposit?: number, crimePercent = 50) {
  const query = new URLSearchParams({ budget_manwon: String(budget), transaction, horizon: String(horizon) });
  query.set("crime_weight", String(crimePercent / 100));
  query.set("traffic_weight", String((100 - crimePercent) / 100));
  if (area !== null) query.set("area_m2", String(area));
  query.set("min_budget_manwon", String(minBudget));
  if (transaction === "wolse" && deposit !== undefined) query.set("deposit_manwon", String(deposit));
  return read<Recommendations>(`/api/neighborhoods/recommendations?${query}`, signal);
}

export const normalizeDong = (value: string) => value.replace(/\s+/g, "").replace(/제+(?=\d)/g, "").replace(/(?<=\d)[.·ㆍ](?=\d)/g, "·");
export const amountFor = (row: Neighborhood | undefined, transaction: Transaction, area: number | null) => {
  if (!row) return null;
  if (transaction === "wolse") return row.wolse?.monthly_rent_manwon ?? null;
  const perPyeong = transaction === "sale" ? row.apart_sale_per_pyeong : row.apart_rent_per_pyeong;
  const amount = transaction === "sale" ? row.apart_sale_amount : row.apart_rent_amount;
  return area !== null ? (perPyeong === null ? null : perPyeong * area / (400 / 121)) : amount;
};
