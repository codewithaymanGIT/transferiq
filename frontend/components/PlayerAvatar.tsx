const POSITION_STYLES: Record<string, string> = {
  GK: "from-accent/25 to-accent/5 text-accent ring-accent/30",
  DF: "from-positive/25 to-positive/5 text-positive ring-positive/30",
  MF: "from-foreground/20 to-foreground/5 text-foreground ring-border",
  FW: "from-negative/25 to-negative/5 text-negative ring-negative/30",
};

const SIZE_STYLES: Record<string, string> = {
  sm: "h-9 w-9 text-xs",
  md: "h-14 w-14 text-base",
  lg: "h-20 w-20 text-2xl",
};

export function initials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

/**
 * Circular initials avatar, tinted by position. No real player photos are
 * available in the data pipeline (FBref stats pulls don't include them),
 * so this is a deliberate, consistent placeholder rather than a broken
 * image -- the same pattern used by Linear, GitHub, and similar premium
 * tools for entities without a photo on file.
 */
export function PlayerAvatar({
  name,
  position,
  size = "md",
}: {
  name: string;
  position?: string;
  size?: "sm" | "md" | "lg";
}) {
  const posKey = (position ?? "").split(",")[0].trim();
  const style = POSITION_STYLES[posKey] ?? POSITION_STYLES.MF;

  return (
    <div
      className={`flex shrink-0 items-center justify-center rounded-full bg-gradient-to-br font-display font-semibold ring-1 transition-transform duration-200 group-hover:scale-105 ${style} ${SIZE_STYLES[size]}`}
    >
      {initials(name)}
    </div>
  );
}
