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
  club_name: string | null;
  last_season_label: string | null;
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

// Read-only public demo: when NEXT_PUBLIC_DEMO_MODE=1 every call is answered
// from frontend/data/demo.json, a snapshot of the real API written by
// scripts/docs/export_demo_data.py. Nothing is generated here -- the file in
// git is empty until that script is run against a loaded database.
import demoSnapshot from "@/data/demo.json";

export const DEMO_MODE = process.env.NEXT_PUBLIC_DEMO_MODE === "1";
export const DEMO_GENERATED_AT: string | null = (demoSnapshot as { generated_at: string | null }).generated_at;

const PAGE_SIZE = 25;

interface DemoSnapshot {
  players: Player[];
  valuations: Record<string, Valuation | null>;
  top: TopValuation[];
}
const demo = demoSnapshot as unknown as DemoSnapshot;

function paginate<T>(rows: T[], page = 1) {
  return {
    items: rows.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE),
    total: rows.length,
    page,
    page_size: PAGE_SIZE,
  };
}

function demoPlayers(params: { position?: string; q?: string; page?: number }): PlayerListResponse {
  const q = params.q?.toLowerCase();
  const rows = demo.players.filter(
    (p) => (!params.position || p.position === params.position) && (!q || p.name.toLowerCase().includes(q)),
  );
  return paginate(rows, params.page);
}

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

export function listPlayers(params: { position?: string; q?: string; page?: number } = {}) {
  const qs = new URLSearchParams();
  if (params.position) qs.set("position", params.position);
  if (params.q) qs.set("q", params.q);
  if (params.page) qs.set("page", String(params.page));
  const suffix = qs.toString() ? `?${qs}` : "";
  if (DEMO_MODE) return Promise.resolve(demoPlayers(params));
  return apiFetch<PlayerListResponse>(`/api/v1/players${suffix}`);
}

export function getPlayer(id: number) {
  if (DEMO_MODE) {
    const player = demo.players.find((p) => p.id === id);
    return player ? Promise.resolve(player) : Promise.reject(new ApiUnavailableError("not_found"));
  }
  return apiFetch<Player>(`/api/v1/players/${id}`);
}

export async function getValuation(id: number): Promise<Valuation | null> {
  if (DEMO_MODE) return demo.valuations[String(id)] ?? null;
  try {
    return await apiFetch<Valuation>(`/api/v1/players/${id}/valuation`);
  } catch (err) {
    if (err instanceof ApiUnavailableError && err.message === "not_found") return null;
    throw err;
  }
}


export interface TopValuation {
  player_id: number;
  player_name: string;
  position: "GK" | "DF" | "MF" | "FW";
  club_name: string | null;
  predicted_value: string;
  confidence: "High" | "Medium" | "Low";
  top_driver_feature: string | null;
  top_driver_value: number | null;
}

export interface TopValuationsResponse {
  items: TopValuation[];
  total: number;
  page: number;
  page_size: number;
}

export function listTopValuations(params: { position?: string; page?: number } = {}) {
  const qs = new URLSearchParams();
  if (params.position) qs.set("position", params.position);
  if (params.page) qs.set("page", String(params.page));
  const suffix = qs.toString() ? `?${qs}` : "";
  if (DEMO_MODE) {
    const rows = demo.top.filter((t) => !params.position || t.position === params.position);
    return Promise.resolve(paginate(rows, params.page));
  }
  return apiFetch<TopValuationsResponse>(`/api/v1/predictions/top${suffix}`);
}
