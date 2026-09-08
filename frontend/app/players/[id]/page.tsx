import { notFound } from "next/navigation";
import { getPlayer, getValuation } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function PlayerProfilePage({ params }: { params: { id: string } }) {
  const id = Number(params.id);
  if (Number.isNaN(id)) notFound();

  let player;
  try {
    player = await getPlayer(id);
  } catch {
    notFound();
  }

  const valuation = await getValuation(id);

  return (
    <div className="space-y-8">
      <div>
        <p className="font-mono text-sm text-muted">{player.position}</p>
        <h1 className="font-display text-3xl font-semibold tracking-tight">{player.name}</h1>
        <p className="mt-1 text-sm text-muted">{player.nationality ?? "Nationality unknown"}</p>
      </div>

      <section className="border border-border px-6 py-6">
        <h2 className="font-display text-lg">Estimated market value</h2>

        {valuation ? (
          <div className="mt-4 space-y-3">
            <p className="font-mono text-4xl text-accent">
              £{Number(valuation.predicted_value).toLocaleString()}
            </p>
            <p className="text-sm text-muted">
              Range £{Number(valuation.low_bound).toLocaleString()} – £
              {Number(valuation.high_bound).toLocaleString()} · {valuation.confidence.toLowerCase()}{" "}
              confidence · model {valuation.model_version}
            </p>
            {valuation.benchmark_value && (
              <p className="text-sm text-muted">
                Benchmark £{Number(valuation.benchmark_value).toLocaleString()}
                {valuation.benchmark_difference_pct != null && (
                  <>
                    {" "}
                    ({valuation.benchmark_difference_pct >= 0 ? "+" : ""}
                    {valuation.benchmark_difference_pct.toFixed(1)}% model premium)
                  </>
                )}
              </p>
            )}
          </div>
        ) : (
          <p className="mt-3 max-w-md text-sm text-muted">
            No model valuation yet. This player hasn&rsquo;t been scored by a
            trained model — that lands once the ML pipeline (Phase 6/7) has run
            against real ingested data. This page never shows a placeholder
            number in its place.
          </p>
        )}
      </section>
    </div>
  );
}
