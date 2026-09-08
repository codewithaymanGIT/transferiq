import Link from "next/link";
import { listPlayers } from "@/lib/api";

export const dynamic = "force-dynamic";

const POSITIONS = ["GK", "DF", "MF", "FW"] as const;

export default async function PlayersPage({
  searchParams,
}: {
  searchParams: { position?: string };
}) {
  const position = searchParams.position;
  let items: Awaited<ReturnType<typeof listPlayers>>["items"] = [];
  let total = 0;
  let errored = false;

  try {
    const data = await listPlayers({ position });
    items = data.items;
    total = data.total;
  } catch {
    errored = true;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-baseline justify-between">
        <h1 className="font-display text-2xl font-semibold tracking-tight">Players</h1>
        <span className="font-mono text-sm text-muted">{total} total</span>
      </div>

      <div className="flex gap-2 text-sm">
        <FilterLink href="/players" active={!position}>
          All
        </FilterLink>
        {POSITIONS.map((p) => (
          <FilterLink key={p} href={`/players?position=${p}`} active={position === p}>
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
          No players match this filter yet.
        </div>
      )}

      {!errored && items.length > 0 && (
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr className="border-b border-border text-left text-muted">
              <th className="py-2 font-normal">Name</th>
              <th className="py-2 font-normal">Position</th>
              <th className="py-2 font-normal">Nationality</th>
            </tr>
          </thead>
          <tbody>
            {items.map((p) => (
              <tr key={p.id} className="border-b border-border/60">
                <td className="py-3">
                  <Link href={`/players/${p.id}`} className="hover:text-accent transition-colors">
                    {p.name}
                  </Link>
                </td>
                <td className="py-3 font-mono text-muted">{p.position}</td>
                <td className="py-3 text-muted">{p.nationality ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function FilterLink({
  href,
  active,
  children,
}: {
  href: string;
  active: boolean;
  children: React.ReactNode;
}) {
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
