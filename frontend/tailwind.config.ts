import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        base: "#0C0F13",       // deep charcoal-navy, not pure black
        surface: "#12161C",
        raised: "#181D25",
        border: "#232A34",
        muted: "#7C8896",
        foreground: "#E7ECF1",
        accent: "#C99A4B",     // muted gold -- for value/currency figures
        accentDim: "#8A7038",
        positive: "#5FA88A",
        negative: "#C4695C",
      },
      fontFamily: {
        display: ["var(--font-space-grotesk)", "sans-serif"],
        body: ["var(--font-plex-sans)", "sans-serif"],
        mono: ["var(--font-plex-mono)", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;
