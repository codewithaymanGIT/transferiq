const SIZE_STYLES: Record<string, string> = {
  sm: "h-5 w-5 text-[9px]",
  md: "h-7 w-7 text-[11px]",
  lg: "h-10 w-10 text-sm",
};

// A small set of deterministic accent tints so different clubs read as
// visually distinct at a glance, without needing real crest artwork.
const TINTS = [
  "bg-accent/15 text-accent",
  "bg-positive/15 text-positive",
  "bg-negative/15 text-negative",
  "bg-foreground/10 text-foreground",
];

export function hashTint(name: string): string {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return TINTS[h % TINTS.length];
}

export function clubInitials(name: string): string {
  const stripped = name.replace(/\b(FC|AFC|CF|United|City)\b/gi, "").trim();
  const parts = (stripped || name).split(/\s+/).filter(Boolean);
  if (parts.length === 1) return parts[0].slice(0, 3).toUpperCase();
  return parts
    .slice(0, 3)
    .map((p) => p[0])
    .join("")
    .toUpperCase();
}

/**
 * Small square "badge" placeholder for a club crest. No real crest source
 * is available without either a new scraping step or an unreliable
 * third-party name-lookup API (confirmed to sometimes mismatch clubs) --
 * this deterministic initials badge is the honest, consistent choice.
 */
export function ClubBadge({ name, size = "md" }: { name: string; size?: "sm" | "md" | "lg" }) {
  return (
    <div
      className={`flex shrink-0 items-center justify-center rounded-md font-display font-semibold tracking-tight ${hashTint(name)} ${SIZE_STYLES[size]}`}
      title={name}
    >
      {clubInitials(name)}
    </div>
  );
}
