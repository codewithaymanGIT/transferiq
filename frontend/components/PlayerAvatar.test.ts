import { describe, it, expect } from "vitest";
import { initials } from "./PlayerAvatar";

describe("initials", () => {
  it("returns first+last initial for a full name", () => {
    expect(initials("Bukayo Saka")).toBe("BS");
  });

  it("handles a middle name by using first and last only", () => {
    expect(initials("Erling Braut Haaland")).toBe("EH");
  });

  it("returns a two-letter prefix for a single-word name", () => {
    expect(initials("Willian")).toBe("WI");
  });

  it("ignores extra whitespace", () => {
    expect(initials("  Kevin   De Bruyne  ")).toBe("KB");
  });

  it("returns a fallback for an empty name", () => {
    expect(initials("")).toBe("?");
  });
});
