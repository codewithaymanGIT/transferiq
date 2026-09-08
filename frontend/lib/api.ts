// Server Components fetch at request time *inside the frontend container*,
// where "localhost" means the frontend container itself, not the backend.
// Client-side code (if any is added later) needs the browser-facing URL
// instead. Use API_INTERNAL_URL (Docker service name) on the server, and
// NEXT_PUBLIC_API_URL (localhost, for the browser) everywhere else.
const API_URL =
  typeof window === "undefined"
    ? process.env.API_INTERNAL_URL ?? "http://backend:8000"
    : process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface Player {
  id: number;
  name: string;
  date_of_birth: string | null;
  nationality: string | null;
  position: "GK" | "DF" | "MF" | "FW";
  current_club_id: number | null;
}

export interface PlayerListResponse {
  items: Player[];
  total: number;
  page: number;
  page_size: number;
}

export interface Valuation {
  player_id: number;
  model_version: string;
  predicted_value: string;
  low_bound: string;
  high_bound: string;
  confidence: "High" | "Medium" | "Low";
  benchmark_value: string | null;
  benchmark_difference_pct: number | null;
  shap_contributions: Record<string, number>;
}

/** True while the backend has no players loaded yet -- distinct from a network/server failure. */
export class ApiUnavailableError extends Error {}

async function apiFetch<T>(path: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, { cache: "no-store" });
  } catch {
    throw new ApiUnavailableError(`Could not reach the API at ${API_URL}. Is the backend running?`);
  }
  if (!res.ok) {
    if (res.status === 404) {
      throw new ApiUnavailableError("not_found");
    }
    throw new Error(`API error ${res.status} on ${path}`);
  }
  return res.json() as Promise<T>;
}

export function listPlayers(params: { position?: string; page?: number } = {}) {
  const qs = new URLSearchParams();
  if (params.position) qs.set("position", params.position);
  if (params.page) qs.set("page", String(params.page));
  const suffix = qs.toString() ? `?${qs}` : "";
  return apiFetch<PlayerListResponse>(`/api/v1/players${suffix}`);
}

export function getPlayer(id: number) {
  return apiFetch<Player>(`/api/v1/players/${id}`);
}

export async function getValuation(id: number): Promise<Valuation | null> {
  try {
    return await apiFetch<Valuation>(`/api/v1/players/${id}/valuation`);
  } catch (err) {
    if (err instanceof ApiUnavailableError && err.message === "not_found") return null;
    throw err;
  }
}
