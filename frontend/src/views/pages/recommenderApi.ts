import { apiRequest } from "../../services/apiClient";

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
  return apiRequest<RecommenderResponse>("/api/recommender/district-forecasts", {
    signal,
    fallbackMessage: "가격 예측 정보를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.",
  });
}
