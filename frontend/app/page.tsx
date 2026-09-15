import { listPlayers } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function DashboardPage() {
  let total = 0;
  let defenders = 0;
  let forwards = 0;
  let apiError: string | null = null;

  try {
    const [all, df, fw] = await Promise.all([
      listPlayers({ page: 1 }),
      listPlayers({ position: "DF", page: 1 }),
      listPlayers({ position: "FW", page: 1 }),
    ]);
    total = all.total;
    defenders = df.total;
    forwards = fw.total;
  } catch (err) {
    apiError = err instanceof Error ? err.message : "Unknown error";
  }

  return (
    <div className="space-y-10">
      <section>
        <h1 className="font-display text-3xl font-semibold tracking-tight">
          Football market intelligence
        </h1>
        <p className="mt-2 max-w-lg text-sm text-muted">
          Player valuations from a Random Forest regression model trained on
          historical performance and transfer-market data, explained with SHAP.
        </p>
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
          <Stat label="Players in database" value={total.toLocaleString()} icon="players" />
          <Stat label="Defenders" value={defenders.toLocaleString()} icon="valuations" />
          <Stat label="Forwards" value={forwards.toLocaleString()} icon="model" />
        </section>
      )}

      {!apiError && total === 0 && (
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

function Stat({
  label,
  value,
  icon,
  mono = true,
}: {
  label: string;
  value: string;
  icon: string;
  mono?: boolean;
}) {
  return (
    <div className="group rounded-xl bg-surface px-6 py-5 shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:shadow-lg hover:shadow-black/20">
      <div className="flex items-center gap-2 text-muted">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" className="h-4 w-4 text-accent/70">
          {ICONS[icon]}
        </svg>
        <p className="text-sm">{label}</p>
      </div>
      <p className={`mt-2 text-2xl text-accent ${mono ? "font-mono" : "font-display font-semibold"}`}>
        {value}
      </p>
    </div>
  );
}
