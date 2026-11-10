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
  const response = await fetch("/api/admin-boundaries/status", { signal });
  if (!response.ok) throw new Error(`행정동 기준정보 조회 실패 (${response.status})`);
  return response.json() as Promise<AdminBoundaryStatus>;
}
