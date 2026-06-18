import { useEffect, useRef, useState } from "react";

import ErrorState from "./ErrorState";
import PredictionForm from "./PredictionForm";
import PredictionResult from "./PredictionResult";
import { usePrediction } from "@/hooks/usePrediction";
import { Coin } from "@/services/coingecko";

interface PredictionPanelProps {
  coins: Coin[];
}

const FEATURES = [
  {
    label: "Model",
    title: "Deep Learning Model",
    desc: "Uses a trained neural network to forecast future price movement.",
  },
  {
    label: "Trend",
    title: "Trend Detection",
    desc: "Classifies the prediction as bullish, bearish, or neutral.",
  },
  {
    label: "Score",
    title: "Confidence Score",
    desc: "Shows a simple confidence value for the prediction result.",
  },
  {
    label: "Chart",
    title: "Visual Price Charts",
    desc: "Compares recent price movement with the predicted path.",
  },
  {
    label: "Range",
    title: "Support and Resistance",
    desc: "Highlights possible lower and upper prediction levels.",
  },
  {
    label: "Days",
    title: "1-365 Day Forecast",
    desc: "Supports short and longer forecast periods up to one year.",
  },
];

const PredictionPanel = ({ coins }: PredictionPanelProps) => {
  const [selectedCoin, setSelectedCoin] = useState<Coin | null>(null);
  const [days, setDays] = useState(7);
  const { predict, clearPrediction, isLoading, error, result } = usePrediction();
  const resultRef = useRef<HTMLDivElement | null>(null);

  const handleCoinChange = (coin: Coin | null) => {
    setSelectedCoin(coin);
    clearPrediction();
  };

  const handleDaysChange = (value: number) => {
    setDays(value);
    clearPrediction();
  };

  const handlePredict = () => {
    if (!selectedCoin || isLoading) return;
    predict(selectedCoin, days);
  };

  useEffect(() => {
    if (result && resultRef.current) {
      resultRef.current.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  }, [result]);

  return (
    <section className="space-y-6">
      <div className="glass-card rounded-xl p-6 glow-cyan animate-slide-up">
        <h2 className="font-display text-lg font-bold text-gradient-neon glow-text-cyan tracking-wider mb-1">
          AI Price Predictor
        </h2>
        <p className="text-muted-foreground text-sm">
          Deep learning analysis of historical crypto price behavior
        </p>
      </div>

      <PredictionForm
        coins={coins}
        selectedCoin={selectedCoin}
        onCoinChange={handleCoinChange}
        days={days}
        onDaysChange={handleDaysChange}
        onPredict={handlePredict}
        isLoading={isLoading}
      />

      {error && <ErrorState message={error} />}

      {result && (
        <div ref={resultRef}>
          <PredictionResult result={result} />
        </div>
      )}

      {!result && !error && <PredictorFeatures />}
    </section>
  );
};

const PredictorFeatures = () => (
  <div className="glass-card rounded-xl p-6 animate-slide-up">
    <h3 className="font-display text-sm font-bold text-gradient-neon tracking-wider mb-4">
      Predictor Features
    </h3>

    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      {FEATURES.map((feature) => (
        <div
          key={feature.title}
          className="flex items-start gap-3 p-3 rounded-lg bg-muted/30 border border-border/40"
        >
          <span className="w-10 text-center text-xs font-bold text-primary">
            {feature.label}
          </span>
          <div>
            <p className="text-foreground text-sm font-semibold">
              {feature.title}
            </p>
            <p className="text-muted-foreground text-xs mt-0.5">
              {feature.desc}
            </p>
          </div>
        </div>
      ))}
    </div>

    <p className="text-muted-foreground text-xs mt-4 border-t border-border/40 pt-3">
      <span className="font-semibold">Disclaimer:</span> AI predictions are
      statistical estimates based on historical data and should not be
      considered financial advice.
    </p>
  </div>
);

export default PredictionPanel;
