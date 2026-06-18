import { Coin } from "@/services/coingecko";

const formatPrice = (price: number) =>
  price >= 1
    ? price.toLocaleString("en-US", {
        style: "currency",
        currency: "USD",
      })
    : `$${price.toPrecision(4)}`;

const formatMarketCap = (mc: number) => {
  if (!mc) return "$0";
  if (mc >= 1e12) return `$${(mc / 1e12).toFixed(2)}T`;
  if (mc >= 1e9) return `$${(mc / 1e9).toFixed(2)}B`;
  if (mc >= 1e6) return `$${(mc / 1e6).toFixed(2)}M`;
  return `$${mc.toLocaleString()}`;
};

const MiniSparkline = ({ data }: { data: number[] }) => {
  if (!data || data.length < 2) return null;

  const filtered = data.filter((_, i) => i % Math.ceil(data.length / 30) === 0);
  if (filtered.length < 2) return null;

  const min = Math.min(...filtered);
  const max = Math.max(...filtered);
  const range = max - min || 1;

  const w = 100;
  const h = 32;

  const points = filtered
    .map(
      (v, i) =>
        `${(i / (filtered.length - 1)) * w},${
          h - ((v - min) / range) * h
        }`
    )
    .join(" ");

  const isUp = filtered[filtered.length - 1] >= filtered[0];

  return (
    <svg width={w} height={h} className="shrink-0">
      <polyline
        points={points}
        fill="none"
        stroke={
          isUp
            ? "hsl(var(--neon-green))"
            : "hsl(var(--neon-red))"
        }
        strokeWidth="1.5"
      />
    </svg>
  );
};

interface CoinRowProps {
  coin: Coin;
}

const CoinRow = ({ coin }: CoinRowProps) => {
  const changePercent = coin.price_change_percentage_24h ?? 0;
  const changePositive = changePercent >= 0;

  return (
    <tr className="border-b border-border/50 hover:bg-muted/50 transition-colors group">
      <td className="py-3 px-2 text-muted-foreground text-sm w-10 text-center">
        {coin.market_cap_rank ?? "-"}
      </td>

      <td className="py-3 px-2">
        <div className="flex items-center gap-3">
          <img
            src={coin.image}
            alt={coin.name}
            className="w-7 h-7 rounded-full"
            loading="lazy"
            onError={(e) => {
              (e.target as HTMLImageElement).src =
                "https://via.placeholder.com/28";
            }}
          />
          <div>
            <span className="font-semibold text-foreground text-sm">
              {coin.name}
            </span>
            <span className="ml-2 text-xs text-muted-foreground uppercase">
              {coin.symbol}
            </span>
          </div>
        </div>
      </td>

      <td className="py-3 px-2 text-right font-mono text-sm text-foreground">
        {formatPrice(coin.current_price ?? 0)}
      </td>

      <td
        className={`py-3 px-2 text-right font-mono text-sm font-semibold ${
          changePositive
            ? "text-neon-green"
            : "text-neon-red"
        }`}
      >
        {changePositive ? "+" : ""}
        {changePercent.toFixed(2)}%
      </td>

      <td className="py-3 px-2 text-right text-sm text-muted-foreground hidden md:table-cell">
        {formatMarketCap(coin.market_cap ?? 0)}
      </td>

      <td className="py-3 px-2 text-right hidden lg:table-cell">
        <MiniSparkline data={coin.sparkline_in_7d?.price ?? []} />
      </td>
    </tr>
  );
};

export default CoinRow;