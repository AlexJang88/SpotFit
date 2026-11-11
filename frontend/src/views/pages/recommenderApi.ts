export type DistrictModelForecast = {
  district: string;
  apart_sale: number | null;
  apart_sale_amount: number | null;
  apart_rent: number | null;
  apart_rent_amount: number | null;
  child_environment_1y: number | null;
  child_environment_3y: number | null;
  crime_score: number | null;
  crime_rank: number | null;
  crime_grade: string | null;
  traffic_score: number | null;
  traffic_accidents: number | null;
  traffic_rank: number | null;
  traffic_grade: string | null;
};

export type NeighborhoodModelForecast = {
  district: string;
  neighborhood: string;
  apart_sale_per_pyeong: number | null;
  apart_sale_amount: number | null;
  apart_rent_per_pyeong: number | null;
  apart_rent_amount: number | null;
};

export type RecommenderResponse = {
  version: string;
  district_count: number;
  models: Record<string, unknown>;
  warnings: string[];
  forecasts: DistrictModelForecast[];
  neighborhood_count: number;
  neighborhoods: NeighborhoodModelForecast[];
};

export async function fetchRecommenderForecasts(signal?: AbortSignal): Promise<RecommenderResponse> {
  const response = await fetch("/api/recommender/district-forecasts", { signal });
  if (!response.ok) throw new Error(`v1 모델 예측값 조회 실패 (${response.status})`);
  return response.json() as Promise<RecommenderResponse>;
}
