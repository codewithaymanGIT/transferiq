import Link from "next/link";
import { listTopValuations } from "@/lib/api";
import { PlayerAvatar } from "@/components/PlayerAvatar";
import { ClubBadge } from "@/components/ClubBadge";

export const dynamic = "force-dynamic";

const POSITIONS = ["GK", "DF", "MF", "FW"] as const;

export default async function RankingsPage({
  searchParams,
}: {
  searchParams: { position?: string };
}) {
  const position = searchParams.position;
  let items: Awaited<ReturnType<typeof listTopValuations>>["items"] = [];
  let total = 0;
  let errored = false;

  try {
    const data = await listTopValuations({ position, page: 1 });
    items = data.items;
    total = data.total;
  } catch {
    errored = true;
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-2xl font-semibold tracking-tight">Rankings</h1>
        <p className="mt-1 text-sm text-muted">
          Every real prediction from the active model, ranked by estimated value.
        </p>
      </div>

      <div className="flex gap-2 text-sm">
        <FilterLink position={undefined} active={!position}>
          All
        </FilterLink>
        {POSITIONS.map((p) => (
          <FilterLink key={p} position={p} active={position === p}>
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
          No predictions available yet.
        </div>
      )}

      {!errored && items.length > 0 && (
        <div className="overflow-hidden rounded-xl bg-surface shadow-sm">
          {items.map((p, idx) => (
            <Link
              key={p.player_id}
              href={`/players/${p.player_id}`}
              className="group flex items-center gap-4 border-b border-border/60 px-5 py-3.5 transition-colors duration-150 last:border-b-0 hover:bg-raised"
            >
              <span className="w-6 shrink-0 text-right font-mono text-sm text-muted">{idx + 1}</span>
              <PlayerAvatar name={p.player_name} position={p.position} size="sm" />
              <div className="min-w-0 flex-1">
                <p className="truncate font-display text-sm font-medium text-foreground transition-colors group-hover:text-accent">
                  {p.player_name}
                </p>
                <div className="mt-0.5 flex items-center gap-1.5 text-xs text-muted">
                  <span className="font-mono">{p.position}</span>
                  {p.club_name && (
                    <>
                      <span className="text-border">&middot;</span>
                      <ClubBadge name={p.club_name} size="sm" />
                      <span className="truncate">{p.club_name}</span>
                    </>
                  )}
                </div>
              </div>
              {p.top_driver_feature && (
                <span className="hidden shrink-0 rounded-full border border-border px-2 py-0.5 font-mono text-[10px] text-muted sm:inline-block">
                  driven by {p.top_driver_feature}
                </span>
              )}
              <p className="shrink-0 font-mono text-sm text-accent">
                &pound;{Number(p.predicted_value).toLocaleString()}m
              </p>
            </Link>
          ))}
        </div>
      )}

      {!errored && items.length > 0 && (
        <p className="text-center text-xs text-muted">
          Showing top {items.length} of {total.toLocaleString()} real predictions.
        </p>
      )}
    </div>
  );
}

function FilterLink({
  position,
  active,
  children,
}: {
  position: string | undefined;
  active: boolean;
  children: React.ReactNode;
}) {
  const href = position ? `/rankings?position=${position}` : "/rankings";
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
