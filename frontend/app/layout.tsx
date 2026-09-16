import type { Metadata } from "next";
import { Space_Grotesk, IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-space-grotesk",
  weight: ["500", "600", "700"],
});
const plexSans = IBM_Plex_Sans({
  subsets: ["latin"],
  variable: "--font-plex-sans",
  weight: ["400", "500", "600"],
});
const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  variable: "--font-plex-mono",
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "TransferIQ — Football Market Intelligence",
  description:
    "Premier League player valuation, powered by a gradient-boosted regression model trained on historical performance and transfer-market data.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${spaceGrotesk.variable} ${plexSans.variable} ${plexMono.variable}`}>
      <body>
        <header className="sticky top-0 z-10 border-b border-border/80 bg-base/85 backdrop-blur-md">
          <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
            <a href="/" className="flex items-center gap-2 font-display text-lg font-semibold tracking-tight">
              <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-accent to-accentDim text-sm text-base">
                T
              </span>
              TransferIQ
            </a>
            <nav className="flex gap-1 font-body text-sm text-muted">
              <a
                href="/"
                className="rounded-md px-3 py-1.5 transition-colors duration-150 hover:bg-raised hover:text-foreground"
              >
                Dashboard
              </a>
              <a
                href="/players"
                className="rounded-md px-3 py-1.5 transition-colors duration-150 hover:bg-raised hover:text-foreground"
              >
                Players
              </a>
              <a
                href="/rankings"
                className="rounded-md px-3 py-1.5 transition-colors duration-150 hover:bg-raised hover:text-foreground"
              >
                Rankings
              </a>
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-6 py-10">{children}</main>
      </body>
    </html>
  );
}
