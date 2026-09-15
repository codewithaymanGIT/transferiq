type Contribution = {
  feature: string;
  value: number;
};

/**
 * Diverging horizontal bar chart for signed feature contributions.
 * Pure server-rendered SVG, no client JS or chart library needed --
 * bars grow left (negative, muted red) or right (positive, teal-green)
 * from a center zero-line, scaled to the largest magnitude shown.
 */
export function ShapBarChart({ contributions }: { contributions: Contribution[] }) {
  if (contributions.length === 0) return null;

  const maxAbs = Math.max(...contributions.map((c) => Math.abs(c.value)), 0.01);
  const trackHalfWidth = 46; // percent of track width on each side of center

  return (
    <div className="space-y-3">
      {contributions.map(({ feature, value }) => {
        const widthPct = (Math.abs(value) / maxAbs) * trackHalfWidth;
        const isPositive = value >= 0;

        return (
          <div key={feature} className="grid grid-cols-[7rem_1fr_4.5rem] items-center gap-3">
            <span className="truncate text-sm text-muted" title={feature}>
              {feature}
            </span>

            <div className="relative h-4">
              <div className="absolute left-1/2 top-0 h-full w-px bg-border" />
              <div
                className={`absolute top-1/2 h-2.5 -translate-y-1/2 ${
                  isPositive ? "bg-positive" : "bg-negative"
                }`}
                style={
                  isPositive
                    ? { left: "50%", width: `${widthPct}%` }
                    : { right: "50%", width: `${widthPct}%` }
                }
              />
            </div>

            <span
              className={`text-right font-mono text-sm ${
                isPositive ? "text-positive" : "text-negative"
              }`}
            >
              {isPositive ? "+" : ""}£{value.toFixed(1)}m
            </span>
          </div>
        );
      })}
    </div>
  );
}
