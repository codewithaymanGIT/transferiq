import Link from "next/link";
import { listPlayers, listTopValuations } from "@/lib/api";
import { PlayerAvatar } from "@/components/PlayerAvatar";
import { ClubBadge } from "@/components/ClubBadge";

export const dynamic = "force-dynamic";

export default async function DashboardPage() {
  let totalPlayers = 0;
  let totalPredictions = 0;
  let topPlayers: Awaited<ReturnType<typeof listTopValuations>>["items"] = [];
  let apiError: string | null = null;

  try {
    const [players, rankings] = await Promise.all([
      listPlayers({ page: 1 }),
      listTopValuations({ page: 1 }),
    ]);
    totalPlayers = players.total;
    totalPredictions = rankings.total;
    topPlayers = rankings.items.slice(0, 4);
  } catch (err) {
    apiError = err instanceof Error ? err.message : "Unknown error";
  }

  return (
    <div className="space-y-12">
      <section>
        <h1 className="font-display text-3xl font-semibold tracking-tight sm:text-4xl">
          Football market intelligence
        </h1>
        <p className="mt-3 max-w-xl text-sm text-muted sm:text-base">
          Real player valuations from a Random Forest model trained on
          historical performance and transfer-market data, with every
          prediction explained by real SHAP contributions, not a black box.
        </p>
        <div className="mt-5 flex gap-3">
          <Link
            href="/rankings"
            className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-base transition-transform duration-150 hover:scale-[1.02]"
          >
            View rankings
          </Link>
          <Link
            href="/players"
            className="rounded-lg border border-border px-4 py-2 text-sm text-muted transition-colors duration-150 hover:border-accent/40 hover:text-foreground"
          >
            Browse players
          </Link>
        </div>
      </section>

      {apiError && (
        <div className="rounded-xl border border-negative/40 bg-negative/10 px-5 py-4 text-sm">
          <p className="font-medium text-negative">Can&rsquo;t reach the API.</p>
          <p className="mt-1 text-muted">
            The backend isn&rsquo;t responding. Run{" "}
            <code className="font-mono text-foreground">docker compose up --build backend postgres</code>{" "}
            and reload.
          </p>
        </div>
      )}

      {!apiError && (
        <section className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <Stat label="Players tracked" value={totalPlayers.toLocaleString()} icon="players" />
          <Stat label="Real predictions issued" value={totalPredictions.toLocaleString()} icon="valuations" />
          <Stat
            label="Top current valuation"
            value={topPlayers[0] ? `£${Number(topPlayers[0].predicted_value).toLocaleString()}m` : "—"}
            icon="model"
          />
        </section>
      )}

      {!apiError && topPlayers.length > 0 && (
        <section>
          <div className="flex items-baseline justify-between">
            <h2 className="font-display text-lg font-semibold">Top valued players</h2>
            <Link href="/rankings" className="text-sm text-accent transition-opacity hover:opacity-80">
              View full rankings &rarr;
            </Link>
          </div>
          <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {topPlayers.map((p) => (
              <Link
                key={p.player_id}
                href={`/players/${p.player_id}`}
                className="group flex items-center gap-3 rounded-xl bg-surface p-4 shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:bg-raised hover:shadow-lg hover:shadow-black/20"
              >
                <PlayerAvatar name={p.player_name} position={p.position} />
                <div className="min-w-0 flex-1">
                  <p className="truncate font-display text-sm font-medium text-foreground transition-colors group-hover:text-accent">
                    {p.player_name}
                  </p>
                  <div className="mt-1 flex items-center gap-1.5 text-xs text-muted">
                    {p.club_name && <ClubBadge name={p.club_name} size="sm" />}
                    <span className="truncate">{p.club_name ?? p.position}</span>
                  </div>
                  <p className="mt-1 font-mono text-sm text-accent">
                    &pound;{Number(p.predicted_value).toLocaleString()}m
                  </p>
                </div>
              </Link>
            ))}
          </div>
        </section>
      )}

      {!apiError && (
        <section className="grid grid-cols-1 gap-4 border-t border-border pt-8 sm:grid-cols-3">
          <Feature
            title="Honest confidence"
            body="Every prediction carries a real, model-derived range and an explicit confidence label — never dressed up to look more certain than the underlying data supports."
          />
          <Feature
            title="Real explainability"
            body="Each valuation is broken down by real SHAP contributions, including tackles and interceptions for defenders — not just goals and assists."
          />
          <Feature
            title="Kept current"
            body="Squad and club data reflects the most recently loaded Premier League season, not a stale snapshot from years ago."
          />
        </section>
      )}

      {!apiError && totalPlayers === 0 && (
        <section className="rounded-xl border border-border px-6 py-10 text-center">
          <p className="font-display text-lg">No player data loaded yet</p>
          <p className="mx-auto mt-2 max-w-md text-sm text-muted">
            This is the working backend and empty database — real player stats
            and transfer data haven&rsquo;t been ingested yet. Once{" "}
            <code className="font-mono">scripts/ingest/run_ingest.py</code> has
            been run against a real data source and loaded into Postgres,
            players will appear here.
        </p>
        </section>
      )}
    </div>
  );
}

const ICONS: Record<string, JSX.Element> = {
  players: (
    <path
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M15 19v-1a4 4 0 0 0-4-4H7a4 4 0 0 0-4 4v1M9 11a3 3 0 1 0 0-6 3 3 0 0 0 0 6Zm10 8v-1a3.5 3.5 0 0 0-2.5-3.36M15 5.13a3 3 0 0 1 0 5.74"
    />
  ),
  valuations: (
    <path
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M3 17V9m6 8V5m6 12v-6m6 6V3M3 21h18"
    />
  ),
  model: (
    <path
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      d="m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Zm0 0v18m-8-13.5 8 4.5 8-4.5"
    />
  ),
};

function Stat({ label, value, icon }: { label: string; value: string; icon: string }) {
  return (
    <div className="group rounded-xl bg-surface px-6 py-5 shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:shadow-lg hover:shadow-black/20">
      <div className="flex items-center gap-2 text-muted">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" className="h-4 w-4 text-accent/70">
          {ICONS[icon]}
        </svg>
        <p className="text-sm">{label}</p>
      </div>
      <p className="mt-2 font-mono text-2xl text-accent">{value}</p>
    </div>
  );
}

function Feature({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-xl bg-surface px-5 py-5">
      <p className="font-display text-sm font-semibold text-foreground">{title}</p>
      <p className="mt-1.5 text-xs leading-relaxed text-muted">{body}</p>
    </div>
  );
}
