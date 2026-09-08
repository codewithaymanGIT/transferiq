import { ApiUnavailableError, listPlayers } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function DashboardPage() {
  let total = 0;
  let apiError: string | null = null;

  try {
    const data = await listPlayers({ page: 1 });
    total = data.total;
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
          Player valuations from a gradient-boosted regression model trained on
          historical performance and transfer-market data, explained with SHAP.
        </p>
      </section>

      {apiError && (
        <div className="rounded-none border border-negative/40 bg-negative/10 px-5 py-4 text-sm">
          <p className="font-medium text-negative">Can&rsquo;t reach the API.</p>
          <p className="mt-1 text-muted">
            The backend isn&rsquo;t responding. Run{" "}
            <code className="font-mono text-foreground">docker compose up --build backend postgres</code>{" "}
            and reload.
          </p>
        </div>
      )}

      {!apiError && (
        <section className="grid grid-cols-1 gap-px overflow-hidden border border-border bg-border sm:grid-cols-3">
          <Stat label="Players in database" value={total.toLocaleString()} />
          <Stat label="Model valuations issued" value="0" />
          <Stat label="Active model version" value="—" />
        </section>
      )}

      {!apiError && total === 0 && (
        <section className="border border-border px-6 py-10 text-center">
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

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-surface px-6 py-5">
      <p className="text-sm text-muted">{label}</p>
      <p className="mt-2 font-mono text-2xl text-accent">{value}</p>
    </div>
  );
}
