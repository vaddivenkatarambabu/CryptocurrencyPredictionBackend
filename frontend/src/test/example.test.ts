import { describe, expect, it, vi } from "vitest";

import { getApi } from "@/services/api";
import { Coin } from "@/services/coingecko";

describe("api service", () => {
  it("returns data when the backend request succeeds", async () => {
    const mockData = [
      {
        id: "bitcoin",
        symbol: "btc",
        name: "Bitcoin",
        current_price: 50000,
      },
    ];

    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockData),
      } as Response)
    );

    const result = await getApi<Coin[]>("/coins");

    expect(result[0].id).toBe("bitcoin");
  });

  it("throws the backend error message when the request fails", async () => {
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: false,
        json: () =>
          Promise.resolve({
            error: { message: "Server Error" },
          }),
      } as Response)
    );

    await expect(getApi<Coin[]>("/coins")).rejects.toThrow("Server Error");
  });
});
