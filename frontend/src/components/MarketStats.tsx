import { Coin } from "@/services/coingecko";
import { TrendingUp, TrendingDown, BarChart3 } from "lucide-react";

const formatMarketCap = (mc: number) => {
  if (!mc) return "$0";
  if (mc >= 1e12) return `$${(mc / 1e12).toFixed(2)}T`;
  if (mc >= 1e9) return `$${(mc / 1e9).toFixed(2)}B`;
  return `$${(mc / 1e6).toFixed(2)}M`;
};

interface MarketStatsProps {
  coins: Coin[];
}

const MarketStats = ({ coins }: MarketStatsProps) => {
  if (!coins || coins.length === 0) return null;

  const totalMcap = coins.reduce(
    (sum, coin) => sum + (coin.market_cap || 0),
    0
  );

  const gainers = coins.filter(
    (c) => (c.price_change_percentage_24h ?? 0) > 0
  ).length;

  const losers = coins.length - gainers;

  const topGainer = [...coins]
    .filter((c) => c.price_change_percentage_24h !== undefined)
    .sort(
      (a, b) =>
        (b.price_change_percentage_24h ?? 0) -
        (a.price_change_percentage_24h ?? 0)
    )[0];

  const isTopPositive =
    (topGainer?.price_change_percentage_24h ?? 0) >= 0;

  const stats = [
    {
      label: "Total Market Cap",
      value: formatMarketCap(totalMcap),
      icon: BarChart3,
      color: "text-foreground",
    },
    {
      label: "Gainers / Losers",
      value: `${gainers} / ${losers}`,
      icon: gainers >= losers ? TrendingUp : TrendingDown,
      color: gainers >= losers ? "text-neon-green" : "text-neon-red",
    },
    {
      label: "Top Gainer",
      value: topGainer
        ? `${topGainer.symbol.toUpperCase()} ${
            isTopPositive ? "+" : ""
          }${topGainer.price_change_percentage_24h?.toFixed(1)}%`
        : "—",
      icon: isTopPositive ? TrendingUp : TrendingDown,
      color: isTopPositive ? "text-neon-green" : "text-neon-red",
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 animate-slide-up">
      {stats.map((s) => (
        <div
          key={s.label}
          className="glass-card rounded-xl p-4 glow-cyan"
        >
          <div className="flex items-center gap-2 mb-1">
            <s.icon className={`h-4 w-4 ${s.color}`} />
            <span className="text-xs text-muted-foreground uppercase tracking-wide">
              {s.label}
            </span>
          </div>
          <p className={`text-lg font-bold font-mono ${s.color}`}>
            {s.value}
          </p>
        </div>
      ))}
    </div>
  );
};

export default MarketStats;