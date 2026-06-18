import { memo } from "react";
import { Coin } from "@/services/coingecko";
import CoinRow from "./CoinRow";

interface CoinTableProps {
  coins: Coin[];
}

const CoinTable = ({ coins }: CoinTableProps) => {
  if (!coins || coins.length === 0) {
    return (
      <div className="glass-card rounded-xl p-8 text-center glow-cyan animate-slide-up">
        <p className="text-muted-foreground text-sm">
          No coins available.
        </p>
      </div>
    );
  }

  return (
    <div className="glass-card rounded-xl overflow-hidden glow-cyan animate-slide-up">
      <div className="overflow-x-auto scrollbar-thin">
        <table className="w-full min-w-[600px]">
          <thead>
            <tr className="border-b border-border text-xs text-muted-foreground uppercase tracking-wider">
              <th className="py-3 px-2 text-center w-10">#</th>
              <th className="py-3 px-2 text-left">Coin</th>
              <th className="py-3 px-2 text-right">Price</th>
              <th className="py-3 px-2 text-right">24h %</th>
              <th className="py-3 px-2 text-right hidden md:table-cell">
                Market Cap
              </th>
              <th className="py-3 px-2 text-right hidden lg:table-cell">
                7d Chart
              </th>
            </tr>
          </thead>

          <tbody>
            {coins.map((coin) => (
              <MemoizedRow key={coin.id} coin={coin} />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

/* 🔥 Prevent unnecessary re-renders */
const MemoizedRow = memo(CoinRow);

export default CoinTable;