import { apiRequest } from "../../services/apiClient";

export type UpcomingBoundaryChange = {
  id: string;
  district: string;
  title: string;
  change_type: "rename" | "split" | "merge" | "boundary_adjustment" | "unknown";
  announced_at: string;
  effective_from: string | null;
  status: "detected" | "confirmed" | "scheduled";
  source_url: string;
};

export type AdminBoundaryStatus = {
  applied_version: string;
  applied_year_month: string;
  effective_from: string;
  verified_at: string;
  district_count: number;
  neighborhood_count: number;
  boundary_file: string;
  upcoming_changes: UpcomingBoundaryChange[];
};

export async function fetchAdminBoundaryStatus(signal?: AbortSignal): Promise<AdminBoundaryStatus> {
  return apiRequest<AdminBoundaryStatus>("/api/admin-boundaries/status", {
    signal,
    fallbackMessage: "행정동 기준정보를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.",
  });
}
