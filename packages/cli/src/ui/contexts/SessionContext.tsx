/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type React from 'react';
import {
  createContext,
  useCallback,
  useContext,
  useState,
  useMemo,
  useEffect,
} from 'react';

import type { SessionMetrics, ModelMetrics } from '@google/gemini-cli-core';
import { uiTelemetryService, sessionId } from '@google/gemini-cli-core';
import * as fs from 'node:fs';

export enum ToolCallDecision {
  ACCEPT = 'accept',
  REJECT = 'reject',
  MODIFY = 'modify',
  AUTO_ACCEPT = 'auto_accept',
}

export type { SessionMetrics, ModelMetrics };

export interface SessionStatsState {
  sessionId: string;
  sessionStartTime: Date;
  metrics: SessionMetrics;
  lastPromptTokenCount: number;
  promptCount: number;
  totalTokens: number;
  costEstimation: number;
}

export interface ComputedSessionStats {
  totalApiTime: number;
  totalToolTime: number;
  agentActiveTime: number;
  apiTimePercent: number;
  toolTimePercent: number;
  cacheEfficiency: number;
  totalDecisions: number;
  successRate: number;
  agreementRate: number;
  totalCachedTokens: number;
  totalPromptTokens: number;
  totalLinesAdded: number;
  totalLinesRemoved: number;
}

// Defines the final "value" of our context, including the state
// and the functions to update it.
interface SessionStatsContextValue {
  stats: SessionStatsState;
  startNewPrompt: () => void;
  getPromptCount: () => number;
}

// --- Context Definition ---

const SessionStatsContext = createContext<SessionStatsContextValue | undefined>(
  undefined,
);

// --- Provider Component ---

export const SessionStatsProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [stats, setStats] = useState<SessionStatsState>({
    sessionId,
    sessionStartTime: new Date(),
    metrics: uiTelemetryService.getMetrics(),
    lastPromptTokenCount: 0,
    promptCount: 0,
    totalTokens: 0,
    costEstimation: 0,
  });

  const MODEL_PRICING_PER_MILLION_TOKENS = useMemo(
    () => ({
      'gemini-2.5-pro': 1.25, // $1.25 per million input tokens
      'gemini-2.5-flash': 0.3, // $0.30 per million input tokens
    }),
    [],
  );

  useEffect(() => {
    const handleUpdate = ({
      metrics,
      lastPromptTokenCount,
    }: {
      metrics: SessionMetrics;
      lastPromptTokenCount: number;
    }) => {
      setStats((prevState) => {
        const totalTokens = Object.values(metrics.models || {}).reduce(
          (acc, modelMetrics) => acc + (modelMetrics.tokens?.total || 0),
          0,
        );
        const costEstimation = Object.entries(metrics.models || {}).reduce(
          (acc, [modelName, modelMetrics]) => {
            const costPerMillionTokens =
              MODEL_PRICING_PER_MILLION_TOKENS[
                modelName as keyof typeof MODEL_PRICING_PER_MILLION_TOKENS
              ] || 0;
            const inputCost =
              ((modelMetrics.tokens?.prompt || 0) / 1_000_000) *
              costPerMillionTokens;
            const outputCost =
              ((modelMetrics.tokens?.candidates || 0) / 1_000_000) *
              costPerMillionTokens; // Assuming output cost is same as input for now
            return acc + inputCost + outputCost;
          },
          0.0,
        );
        fs.appendFileSync(
          '/tmp/gemini_session_debug.log',
          `\n--- handleUpdate ---\nMetrics: ${JSON.stringify(metrics)}\nTotal Tokens: ${totalTokens}\nCost Estimation: ${costEstimation}`,
        );
        return {
          ...prevState,
          metrics,
          lastPromptTokenCount,
          totalTokens,
          costEstimation,
        };
      });
    };

    uiTelemetryService.on('update', handleUpdate);
    // Initial state is set by the first 'update' event from uiTelemetryService
    // or remains at 0 if no events occur.

    return () => {
      uiTelemetryService.off('update', handleUpdate);
    };
  }, [MODEL_PRICING_PER_MILLION_TOKENS]);

  const startNewPrompt = useCallback(() => {
    setStats((prevState) => ({
      ...prevState,
      promptCount: prevState.promptCount + 1,
    }));
  }, []);

  const getPromptCount = useCallback(
    () => stats.promptCount,
    [stats.promptCount],
  );

  const value = useMemo(
    () => ({
      stats,
      startNewPrompt,
      getPromptCount,
    }),
    [stats, startNewPrompt, getPromptCount],
  );

  return (
    <SessionStatsContext.Provider value={value}>
      {children}
    </SessionStatsContext.Provider>
  );
};

// --- Consumer Hook ---

export const useSessionStats = () => {
  const context = useContext(SessionStatsContext);
  if (context === undefined) {
    throw new Error(
      'useSessionStats must be used within a SessionStatsProvider',
    );
  }
  return context;
};
