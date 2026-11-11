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
  score_method: string; tie_rule: string;
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

async function read<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : `데이터 조회 실패 (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export function fetchCatalog(horizon: number, signal?: AbortSignal) {
  return read<Catalog>(`/api/neighborhoods/catalog?horizon=${horizon}`, signal);
}

export function fetchRecommendations(budget: number, transaction: Transaction, horizon: number,
  area: number | null, signal?: AbortSignal) {
  const query = new URLSearchParams({ budget_manwon: String(budget), transaction, horizon: String(horizon) });
  if (area !== null) query.set("area_m2", String(area));
  return read<Recommendations>(`/api/neighborhoods/recommendations?${query}`, signal);
}

export const normalizeDong = (value: string) => value.replace(/\s+/g, "").replace(/제+(?=\d)/g, "").replace(/(?<=\d)[.·ㆍ](?=\d)/g, "·");
export const amountFor = (row: Neighborhood | undefined, transaction: Transaction, area: number | null) => {
  if (!row) return null;
  const perPyeong = transaction === "sale" ? row.apart_sale_per_pyeong : row.apart_rent_per_pyeong;
  const amount = transaction === "sale" ? row.apart_sale_amount : row.apart_rent_amount;
  return area !== null ? (perPyeong === null ? null : perPyeong * area / (400 / 121)) : amount;
};
