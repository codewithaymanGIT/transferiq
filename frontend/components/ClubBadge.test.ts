import { describe, it, expect } from "vitest";
import { clubInitials, hashTint } from "./ClubBadge";

describe("clubInitials", () => {
  it("strips a common suffix like United, leaving one real word", () => {
    expect(clubInitials("Manchester United")).toBe("MAN");
  });

  it("builds initials from multiple real words", () => {
    expect(clubInitials("Nottingham Forest")).toBe("NF");
  });

  it("falls back to a 3-letter prefix for a single-word name", () => {
    expect(clubInitials("Arsenal")).toBe("ARS");
  });
});

describe("hashTint", () => {
  it("is deterministic for the same club name", () => {
    expect(hashTint("Chelsea")).toBe(hashTint("Chelsea"));
  });

  it("returns one of the known tint classes", () => {
    expect(hashTint("Liverpool")).toMatch(/^bg-/);
  });
});
