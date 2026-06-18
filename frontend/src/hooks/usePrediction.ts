import { useState } from "react";

import { postApi } from "@/services/api";
import { Coin } from "@/services/coingecko";

export interface PredictionResult {
  prediction: {
    predictedPrice: number;
    changePercent: number;
    confidence: number;
    trend: "bullish" | "bearish" | "neutral";
    dailyPrices: number[];
    analysis: string;
    supportLevel: number;
    resistanceLevel: number;
  };
  currentPrice: number;
  coinId: string;
  coinName: string;
  symbol: string;
  days: number;
}

export function usePrediction() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);

  const clearPrediction = () => {
    setError(null);
    setResult(null);
  };

  const predict = async (coin: Coin, days: number) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await postApi<PredictionResult>("/predict", {
        coinId: coin.id,
        days,
      });
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setIsLoading(false);
    }
  };

  return { predict, clearPrediction, isLoading, error, result };
}
