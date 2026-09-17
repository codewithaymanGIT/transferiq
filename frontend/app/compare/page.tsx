import { getPlayer, getValuation, listPlayers, type Player, type Valuation } from "@/lib/api";
import { PlayerAvatar } from "@/components/PlayerAvatar";
import { ClubBadge } from "@/components/ClubBadge";
import { ShapBarChart } from "@/components/ShapBarChart";

export const dynamic = "force-dynamic";

async function resolvePlayer(name: string): Promise<Player | null> {
  try {
    const data = await listPlayers({ q: name, page: 1 });
    return data.items[0] ?? null;
  } catch {
    return null;
  }
}

export default async function ComparePage({
  searchParams,
}: {
  searchParams: { n1?: string; n2?: string };
}) {
  const n1 = searchParams.n1;
  const n2 = searchParams.n2;

  let player1: Player | null = null;
  let player2: Player | null = null;
  let valuation1: Valuation | null = null;
  let valuation2: Valuation | null = null;
  let errored = false;

  try {
    if (n1) player1 = await resolvePlayer(n1);
    if (n2) player2 = await resolvePlayer(n2);
    if (player1) valuation1 = await getValuation(player1.id);
    if (player2) valuation2 = await getValuation(player2.id);
  } catch {
    errored = true;
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-2xl font-semibold tracking-tight">Compare players</h1>
        <p className="mt-1 text-sm text-muted">
          Search two players by name to compare their real valuations and SHAP drivers side by side.
        </p>
      </div>

      <form action="/compare" method="get" className="flex flex-col gap-3 sm:flex-row">
        <input
          type="text"
          name="n1"
          defaultValue={n1 ?? ""}
          placeholder="First player name"
          className="w-full rounded-lg bg-surface px-3 py-2 text-sm text-foreground placeholder:text-muted focus:outline-none focus:ring-2 focus:ring-accent/40"
        />
        <input
          type="text"
          name="n2"
          defaultValue={n2 ?? ""}
          placeholder="Second player name"
          className="w-full rounded-lg bg-surface px-3 py-2 text-sm text-foreground placeholder:text-muted focus:outline-none focus:ring-2 focus:ring-accent/40"
        />
        <button
          type="submit"
          className="shrink-0 rounded-lg border border-border px-4 py-2 text-sm text-muted transition-colors duration-150 hover:border-accent/40 hover:text-foreground"
        >
          Compare
        </button>
      </form>

      {errored && (
        <p className="rounded-lg border border-negative/40 bg-negative/10 px-5 py-4 text-sm text-negative">
          Can&rsquo;t reach the API right now.
        </p>
      )}

      {!errored && (n1 || n2) && (!player1 || !player2) && (
        <div className="rounded-lg border border-border px-6 py-8 text-center text-sm text-muted">
          {n1 && !player1 && <p>No player found matching &ldquo;{n1}&rdquo;.</p>}
          {n2 && !player2 && <p>No player found matching &ldquo;{n2}&rdquo;.</p>}
        </div>
      )}

      {!errored && player1 && player2 && (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <PlayerColumn player={player1} valuation={valuation1} />
          <PlayerColumn player={player2} valuation={valuation2} />
        </div>
      )}
    </div>
  );
}

function PlayerColumn({ player, valuation }: { player: Player; valuation: Valuation | null }) {
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 rounded-xl bg-surface px-5 py-5 shadow-sm">
        <PlayerAvatar name={player.name} position={player.position} size="lg" />
        <div className="min-w-0">
          <p className="font-mono text-xs text-muted">{player.position}</p>
          <p className="truncate font-display text-lg font-semibold">{player.name}</p>
          <div className="mt-1 flex items-center gap-1.5 text-xs text-muted">
            {player.club_name && (
              <>
                <ClubBadge name={player.club_name} size="sm" />
                <span className="truncate">{player.club_name}</span>
              </>
            )}
          </div>
        </div>
      </div>

      <div className="rounded-xl bg-surface px-5 py-5 shadow-sm">
        <h3 className="font-display text-sm font-semibold text-muted">Estimated value</h3>
        {valuation ? (
          <>
            <div className="mt-2 flex items-baseline gap-2">
              <p className="font-mono text-2xl text-accent">
                &pound;{Number(valuation.predicted_value).toLocaleString()}m
              </p>
              <span
                className={`rounded-full border px-2 py-0.5 text-xs ${
                  valuation.confidence === "Low"
                    ? "border-negative/40 text-negative"
                    : valuation.confidence === "Medium"
                      ? "border-accent/40 text-accent"
                      : "border-positive/40 text-positive"
                }`}
              >
                {valuation.confidence}
              </span>
            </div>
            <p className="mt-1 text-xs text-muted">
              Range &pound;{Number(valuation.low_bound).toLocaleString()}m &ndash; &pound;
              {Number(valuation.high_bound).toLocaleString()}m
            </p>
          </>
        ) : (
          <p className="mt-2 text-sm text-muted">No model valuation yet for this player.</p>
        )}
      </div>

      {valuation && Object.keys(valuation.shap_contributions).length > 0 && (
        <div className="rounded-xl bg-surface px-5 py-5 shadow-sm">
          <h3 className="font-display text-sm font-semibold text-muted">Top drivers</h3>
          <div className="mt-3">
            <ShapBarChart
              contributions={Object.entries(valuation.shap_contributions)
                .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                .slice(0, 5)
                .map(([feature, value]) => ({ feature, value }))}
          />
          </div>
        </div>
      )}
    </div>
  );
}
