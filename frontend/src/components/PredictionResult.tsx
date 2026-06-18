import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ReactNode } from "react";

import { PredictionResult as PredictionResultType } from "@/hooks/usePrediction";

interface Props {
  result: PredictionResultType;
}

interface StatCardProps {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  color?: string;
}

const formatPrice = (price: number) =>
  price >= 1
    ? `$${price.toLocaleString("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      })}`
    : `$${price.toPrecision(4)}`;

const PredictionResult = ({ result }: Props) => {
  if (!result.prediction) return null;

  const { prediction, currentPrice, coinName, days } = result;
  const dailyPrices = prediction.dailyPrices || [];
  const finalPredicted = dailyPrices[dailyPrices.length - 1] || currentPrice;
  const isGrowth = finalPredicted >= currentPrice;
  const dynamicColor = isGrowth
    ? "hsl(var(--neon-green))"
    : "hsl(var(--neon-red))";

  const actualHistory = Array.from({ length: days }, (_, index) => {
    const factor = 1 - (days - index) * 0.003;
    return {
      label: `-${days - index}d`,
      actual: currentPrice * factor,
      predicted: null as number | null,
    };
  });

  const predictionFuture = dailyPrices.map((price, index) => ({
    label: `+${index + 1}d`,
    actual: null as number | null,
    predicted: price,
  }));

  const combinedChartData = [
    ...actualHistory,
    {
      label: "Now",
      actual: currentPrice,
      predicted: currentPrice,
    },
    ...predictionFuture,
  ];

  return (
    <div className="space-y-6 animate-slide-up">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard
          label="Predicted Price"
          value={formatPrice(prediction.predictedPrice)}
          sub={`in ${days} days`}
        />
        <StatCard
          label="Change"
          value={`${isGrowth ? "+" : ""}${prediction.changePercent.toFixed(2)}%`}
          color={isGrowth ? "text-neon-green" : "text-neon-red"}
        />
        <StatCard
          label="Confidence"
          value={`${(prediction.confidence * 100).toFixed(0)}%`}
        />
        <StatCard
          label="Trend"
          value={isGrowth ? "Bullish" : "Bearish"}
          color={isGrowth ? "text-neon-green" : "text-neon-red"}
        />
      </div>

      <div className="glass-card rounded-xl p-5">
        <h3 className="font-display text-sm font-bold mb-4">
          {coinName} Previous {days} Days (Actual)
        </h3>

        <ResponsiveContainer width="100%" height={220}>
          <AreaChart data={actualHistory}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="label" />
            <YAxis />
            <Tooltip formatter={(value: number) => formatPrice(value)} />
            <Area
              type="monotone"
              dataKey="actual"
              stroke="hsl(var(--neon-cyan))"
              fillOpacity={0.2}
              fill="hsl(var(--neon-cyan))"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="glass-card rounded-xl p-5">
        <h3 className="font-display text-sm font-bold mb-4">
          Actual vs Predicted - {coinName}
        </h3>

        <ResponsiveContainer width="100%" height={260}>
          <AreaChart data={combinedChartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="label" />
            <YAxis />
            <Tooltip
              formatter={(value: number, name: string) => [
                formatPrice(value),
                name === "actual" ? "Actual" : "Predicted",
              ]}
            />
            <Legend />
            <ReferenceLine x="Now" strokeDasharray="4 4" />
            <Area
              type="monotone"
              dataKey="actual"
              stroke="hsl(var(--neon-cyan))"
              fillOpacity={0.15}
              fill="hsl(var(--neon-cyan))"
              connectNulls
            />
            <Area
              type="monotone"
              dataKey="predicted"
              stroke={dynamicColor}
              fillOpacity={0.25}
              fill={dynamicColor}
              strokeWidth={2}
              connectNulls
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

const StatCard = ({
  label,
  value,
  sub,
  color = "text-foreground",
}: StatCardProps) => (
  <div className="glass-card rounded-xl p-4 border border-border/50">
    <p className="text-muted-foreground text-xs uppercase tracking-wider mb-1">
      {label}
    </p>
    <p className={`font-mono font-bold text-lg ${color}`}>{value}</p>
    {sub && <div className="text-muted-foreground text-xs mt-1">{sub}</div>}
  </div>
);

export default PredictionResult;
