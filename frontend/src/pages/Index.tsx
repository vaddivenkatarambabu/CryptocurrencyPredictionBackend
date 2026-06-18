import { useMemo, useState } from "react";

import CoinTable from "@/components/CoinTable";
import ErrorState from "@/components/ErrorState";
import LoadingSkeleton from "@/components/LoadingSkeleton";
import MarketStats from "@/components/MarketStats";
import PredictionPanel from "@/components/PredictionPanel";
import SearchBar from "@/components/SearchBar";
import { useCoins } from "@/hooks/useCoins";

type Tab = "market" | "predict";

const Index = () => {
  const { data: coins, isLoading, error, refetch } = useCoins();
  const [search, setSearch] = useState("");
  const [tab, setTab] = useState<Tab>("market");

  const filteredCoins = useMemo(() => {
    if (!coins) return [];
    if (!search.trim()) return coins;

    const query = search.toLowerCase();
    return coins.filter(
      (coin) =>
        coin.name.toLowerCase().includes(query) ||
        coin.symbol.toLowerCase().includes(query)
    );
  }, [coins, search]);

  return (
    <div className="min-h-screen bg-background bg-grid-pattern relative">
      <header className="sticky top-0 z-50 glass-card border-b border-border/50">
        <div className="container max-w-6xl mx-auto px-4 py-4 flex flex-col sm:flex-row items-center justify-between gap-3">
          <h1 className="font-display text-xl font-bold text-gradient-neon glow-text-cyan tracking-wider">
            CryptoForge
          </h1>

          <div className="flex items-center gap-3">
            <div className="flex rounded-lg border border-border/60 overflow-hidden">
              <button
                onClick={() => setTab("market")}
                className={`px-4 py-2 text-xs font-semibold tracking-wider transition-colors ${
                  tab === "market"
                    ? "bg-primary text-primary-foreground"
                    : "bg-transparent text-muted-foreground hover:text-foreground"
                }`}
              >
                Market
              </button>
              <button
                onClick={() => setTab("predict")}
                className={`px-4 py-2 text-xs font-semibold tracking-wider transition-colors ${
                  tab === "predict"
                    ? "bg-primary text-primary-foreground"
                    : "bg-transparent text-muted-foreground hover:text-foreground"
                }`}
              >
                Predict
              </button>
            </div>

            {tab === "market" && <SearchBar value={search} onChange={setSearch} />}
          </div>
        </div>
      </header>

      <main className="container max-w-6xl mx-auto px-4 py-6 space-y-6">
        {isLoading && <LoadingSkeleton />}

        {error && (
          <ErrorState
            message={
              error instanceof Error ? error.message : "Something went wrong"
            }
            onRetry={() => refetch()}
          />
        )}

        {coins && tab === "market" && (
          <>
            <MarketStats coins={coins} />
            {filteredCoins.length === 0 ? (
              <div className="glass-card rounded-xl p-8 text-center">
                <p className="text-muted-foreground">
                  No coins match "{search}"
                </p>
              </div>
            ) : (
              <CoinTable coins={filteredCoins} />
            )}
          </>
        )}

        {coins && tab === "predict" && <PredictionPanel coins={coins} />}
      </main>
    </div>
  );
};

export default Index;
