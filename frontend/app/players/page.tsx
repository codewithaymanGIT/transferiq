import Link from "next/link";
import { listPlayers } from "@/lib/api";
import { PlayerAvatar } from "@/components/PlayerAvatar";
import { ClubBadge } from "@/components/ClubBadge";

export const dynamic = "force-dynamic";

const POSITIONS = ["GK", "DF", "MF", "FW"] as const;

const POSITION_COLOR: Record<string, string> = {
  GK: "text-accent border-accent/30 bg-accent/10",
  DF: "text-positive border-positive/30 bg-positive/10",
  MF: "text-foreground border-border bg-raised",
  FW: "text-negative border-negative/30 bg-negative/10",
};

export default async function PlayersPage({
  searchParams,
}: {
  searchParams: { position?: string; q?: string };
}) {
  const position = searchParams.position;
  const q = searchParams.q;
  let items: Awaited<ReturnType<typeof listPlayers>>["items"] = [];
  let total = 0;
  let errored = false;

  try {
    const data = await listPlayers({ position, q, page: 1 });
    items = data.items;
    total = data.total;
  } catch {
    errored = true;
  }

  const baseParams = new URLSearchParams();
  if (q) baseParams.set("q", q);

  return (
    <div className="space-y-6">
      <div className="flex items-baseline justify-between">
        <h1 className="font-display text-2xl font-semibold tracking-tight">Players</h1>
        <span className="font-mono text-sm text-muted">{total.toLocaleString()} total</span>
      </div>

      <form action="/players" method="get" className="flex gap-3">
        {position && <input type="hidden" name="position" value={position} />}
        <input
          type="text"
          name="q"
          defaultValue={q ?? ""}
          placeholder="Search players by name"
          className="w-full max-w-xs rounded-lg bg-surface px-3 py-2 text-sm text-foreground placeholder:text-muted transition-shadow duration-150 focus:outline-none focus:ring-2 focus:ring-accent/40"
        />
        <button
          type="submit"
          className="rounded-lg border border-border px-4 py-2 text-sm text-muted transition-colors duration-150 hover:border-accent/40 hover:text-foreground"
        >
          Search
        </button>
      </form>

      <div className="flex gap-2 text-sm">
        <FilterLink searchParams={baseParams} position={undefined} active={!position}>
          All
        </FilterLink>
        {POSITIONS.map((p) => (
          <FilterLink key={p} searchParams={baseParams} position={p} active={position === p}>
            {p}
          </FilterLink>
        ))}
      </div>

      {errored && (
        <p className="rounded-lg border border-negative/40 bg-negative/10 px-5 py-4 text-sm text-negative">
          Can&rsquo;t reach the API right now.
        </p>
      )}

      {!errored && items.length === 0 && (
        <div className="rounded-lg border border-border px-6 py-10 text-center text-sm text-muted">
          No players match this search.
        </div>
      )}

      {!errored && items.length > 0 && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {items.map((p) => (
            <Link
              key={p.id}
              href={`/players/${p.id}`}
              className="group flex items-center gap-3 rounded-xl bg-surface p-4 shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:bg-raised hover:shadow-lg hover:shadow-black/20"
            >
              <PlayerAvatar name={p.name} position={p.position} />
              <div className="min-w-0 flex-1">
                <p className="truncate font-display text-sm font-medium text-foreground transition-colors group-hover:text-accent">
                  {p.name}
                </p>
                <div className="mt-1 flex items-center gap-1.5">
                  <span
                    className={`rounded border px-1.5 py-0.5 font-mono text-[10px] ${POSITION_COLOR[p.position]}`}
                  >
                    {p.position}
                  </span>
                  {p.club_name && (
                    <span className="flex items-center gap-1 truncate text-xs text-muted">
                      <ClubBadge name={p.club_name} size="sm" />
                      {p.club_name}
                    </span>
                  )}
                </div>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

function FilterLink({
  searchParams,
  position,
  active,
  children,
}: {
  searchParams: URLSearchParams;
  position: string | undefined;
  active: boolean;
  children: React.ReactNode;
}) {
  const params = new URLSearchParams(searchParams);
  if (position) params.set("position", position);
  const href = params.toString() ? `/players?${params}` : "/players";

  return (
    <Link
      href={href}
      className={`rounded-lg border px-3 py-1 transition-colors duration-150 ${
        active ? "border-accent text-accent" : "border-border text-muted hover:text-foreground"
      }`}
    >
      {children}
    </Link>
  );
}
