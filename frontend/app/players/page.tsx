import Link from "next/link";
import { listPlayers } from "@/lib/api";

export const dynamic = "force-dynamic";

const POSITIONS = ["GK", "DF", "MF", "FW"] as const;

const POSITION_COLOR: Record<string, string> = {
  GK: "text-accent border-accent/40",
  DF: "text-positive border-positive/40",
  MF: "text-foreground border-border",
  FW: "text-negative border-negative/40",
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
          className="w-full max-w-xs bg-surface px-3 py-2 text-sm text-foreground placeholder:text-muted focus:outline-none"
        />
        <button
          type="submit"
          className="border border-border px-4 py-2 text-sm text-muted transition-colors hover:text-foreground"
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
        <p className="border border-negative/40 bg-negative/10 px-5 py-4 text-sm text-negative">
          Can&rsquo;t reach the API right now.
        </p>
      )}

      {!errored && items.length === 0 && (
        <div className="border border-border px-6 py-10 text-center text-sm text-muted">
          No players match this search.
        </div>
      )}

      {!errored && items.length > 0 && (
        <div className="bg-surface">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="border-b border-border text-left text-muted">
                <th className="px-5 py-3 font-normal">Name</th>
                <th className="px-5 py-3 font-normal">Position</th>
                <th className="px-5 py-3 font-normal">Nationality</th>
              </tr>
            </thead>
            <tbody>
              {items.map((p) => (
                <tr key={p.id} className="border-b border-border/60 transition-colors hover:bg-raised">
                  <td className="px-5 py-3">
                    <Link href={`/players/${p.id}`} className="hover:text-accent transition-colors">
                      {p.name}
                    </Link>
                  </td>
                  <td className="px-5 py-3">
                    <span
                      className={`inline-block border px-2 py-0.5 font-mono text-xs ${POSITION_COLOR[p.position]}`}
                    >
                      {p.position}
                    </span>
                  </td>
                  <td className="px-5 py-3 text-muted">{p.nationality ?? "\u2014"}</td>
                </tr>
              ))}
            </tbody>
          </table>
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
      className={`border px-3 py-1 transition-colors ${
        active ? "border-accent text-accent" : "border-border text-muted hover:text-foreground"
      }`}
    >
      {children}
    </Link>
  );
}
