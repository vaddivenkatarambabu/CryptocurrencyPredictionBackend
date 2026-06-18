import { useQuery } from "@tanstack/react-query";

import { getApi } from "@/services/api";
import { Coin } from "@/services/coingecko";

export function useCoins() {
  return useQuery({
    queryKey: ["coins"],
    queryFn: () => getApi<Coin[]>("/coins?page=1&per_page=50"),
    staleTime: 1000 * 60,
    refetchInterval: 1000 * 60,
  });
}
