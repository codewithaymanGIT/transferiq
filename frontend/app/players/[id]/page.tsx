import { notFound } from "next/navigation";
import { getPlayer, getValuation } from "@/lib/api";
import { ShapBarChart } from "@/components/ShapBarChart";

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

      <section className="bg-surface px-6 py-6">
        <h2 className="font-display text-lg">Estimated market value</h2>

        {valuation ? (
          <div className="mt-4 space-y-3">
            <div className="flex items-baseline gap-3">
              <p className="font-mono text-4xl text-accent">
                &pound;{Number(valuation.predicted_value).toLocaleString()}m
              </p>
              <span
                className={`border px-2 py-0.5 text-xs ${
                  valuation.confidence === "Low"
                    ? "border-negative/40 text-negative"
                    : valuation.confidence === "Medium"
                      ? "border-accent/40 text-accent"
                      : "border-positive/40 text-positive"
                }`}
              >
                {valuation.confidence} confidence
              </span>
            </div>
            <p className="text-sm text-muted">
              Range &pound;{Number(valuation.low_bound).toLocaleString()}m to &pound;
              {Number(valuation.high_bound).toLocaleString()}m, model {valuation.model_version}
            </p>
            {valuation.benchmark_value && (
              <p className="text-sm text-muted">
                Benchmark &pound;{Number(valuation.benchmark_value).toLocaleString()}
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
            No model valuation yet. This player has not been scored by a
            trained model, since that lands once the ML pipeline has run
            against real ingested data. This page never shows a placeholder
            number in its place.
          </p>
        )}
      </section>

      {valuation && Object.keys(valuation.shap_contributions).length > 0 && (
        <section className="bg-surface px-6 py-6">
          <h2 className="font-display text-lg">Why this valuation</h2>
          <p className="mt-1 text-sm text-muted">
            Approximate contribution of each feature to the prediction above, from a
            real SHAP explainer on the trained model. A first-order approximation,
            not an exact decomposition.
          </p>
          <div className="mt-6">
            <ShapBarChart
              contributions={Object.entries(valuation.shap_contributions)
                .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                .slice(0, 8)
                .map(([feature, value]) => ({ feature, value }))}
            />
          </div>
        </section>
      )}
    </div>
  );
}
