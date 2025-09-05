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

import type {
  SessionMetrics,
  ModelMetrics,
  ToolCallStats,
} from '@google/gemini-cli-core';
import { uiTelemetryService, sessionId } from '@google/gemini-cli-core';
import * as fs from 'node:fs';

export enum ToolCallDecision {
  ACCEPT = 'accept',
  REJECT = 'reject',
  MODIFY = 'modify',
  AUTO_ACCEPT = 'auto_accept',
}

function areModelMetricsEqual(a: ModelMetrics, b: ModelMetrics): boolean {
  if (
    a.api.totalRequests !== b.api.totalRequests ||
    a.api.totalErrors !== b.api.totalErrors ||
    a.api.totalLatencyMs !== b.api.totalLatencyMs
  ) {
    return false;
  }
  if (
    a.tokens.prompt !== b.tokens.prompt ||
    a.tokens.candidates !== b.tokens.candidates ||
    a.tokens.total !== b.tokens.total ||
    a.tokens.cached !== b.tokens.cached ||
    a.tokens.thoughts !== b.tokens.thoughts ||
    a.tokens.tool !== b.tokens.tool
  ) {
    return false;
  }
  return true;
}

function areToolCallStatsEqual(a: ToolCallStats, b: ToolCallStats): boolean {
  if (
    a.count !== b.count ||
    a.success !== b.success ||
    a.fail !== b.fail ||
    a.durationMs !== b.durationMs
  ) {
    return false;
  }
  if (
    a.decisions[ToolCallDecision.ACCEPT] !==
      b.decisions[ToolCallDecision.ACCEPT] ||
    a.decisions[ToolCallDecision.REJECT] !==
      b.decisions[ToolCallDecision.REJECT] ||
    a.decisions[ToolCallDecision.MODIFY] !==
      b.decisions[ToolCallDecision.MODIFY] ||
    a.decisions[ToolCallDecision.AUTO_ACCEPT] !==
      b.decisions[ToolCallDecision.AUTO_ACCEPT]
  ) {
    return false;
  }
  return true;
}

function areMetricsEqual(a: SessionMetrics, b: SessionMetrics): boolean {
  if (a === b) return true;
  if (!a || !b) return false;

  // Compare files
  if (
    a.files.totalLinesAdded !== b.files.totalLinesAdded ||
    a.files.totalLinesRemoved !== b.files.totalLinesRemoved
  ) {
    return false;
  }

  // Compare tools
  const toolsA = a.tools;
  const toolsB = b.tools;
  if (
    toolsA.totalCalls !== toolsB.totalCalls ||
    toolsA.totalSuccess !== toolsB.totalSuccess ||
    toolsA.totalFail !== toolsB.totalFail ||
    toolsA.totalDurationMs !== toolsB.totalDurationMs
  ) {
    return false;
  }

  // Compare tool decisions
  const totalDecisionsA = toolsA.totalDecisions;
  const totalDecisionsB = toolsB.totalDecisions;
  if (
    totalDecisionsA[ToolCallDecision.ACCEPT] !==
      totalDecisionsB[ToolCallDecision.ACCEPT] ||
    totalDecisionsA[ToolCallDecision.REJECT] !==
      totalDecisionsB[ToolCallDecision.REJECT] ||
    totalDecisionsA[ToolCallDecision.MODIFY] !==
      totalDecisionsB[ToolCallDecision.MODIFY] ||
    totalDecisionsA[ToolCallDecision.AUTO_ACCEPT] !==
      totalDecisionsB[ToolCallDecision.AUTO_ACCEPT]
  ) {
    return false;
  }

  // Compare tools.byName
  const toolsByNameAKeys = Object.keys(toolsA.byName);
  const toolsByNameBKeys = Object.keys(toolsB.byName);
  if (toolsByNameAKeys.length !== toolsByNameBKeys.length) return false;

  for (const key of toolsByNameAKeys) {
    const toolA = toolsA.byName[key];
    const toolB = toolsB.byName[key];
    if (!toolB || !areToolCallStatsEqual(toolA, toolB)) {
      return false;
    }
  }

  // Compare models
  const modelsAKeys = Object.keys(a.models);
  const modelsBKeys = Object.keys(b.models);
  if (modelsAKeys.length !== modelsBKeys.length) return false;

  for (const key of modelsAKeys) {
    if (!b.models[key] || !areModelMetricsEqual(a.models[key], b.models[key])) {
      return false;
    }
  }

  return true;
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

  const MODEL_PRICING_PER_MILLION_TOKENS: Record<string, number> = {
    'gemini-2.5-pro': 1.25, // $1.25 per million input tokens
    'gemini-2.5-flash': 0.3, // $0.30 per million input tokens
  };

  useEffect(() => {
    const handleUpdate = ({
      metrics,
      lastPromptTokenCount,
    }: { metrics: SessionMetrics; lastPromptTokenCount: number }) => {
      setStats((prevState) => {
        const totalTokens = Object.values(metrics.models || {}).reduce(
          (acc, modelMetrics) => acc + (modelMetrics.tokens?.total || 0),
          0,
        );
        const costEstimation = Object.entries(metrics.models || {}).reduce(
          (acc, [modelName, modelMetrics]) => {
            const costPerMillionTokens = MODEL_PRICING_PER_MILLION_TOKENS[modelName] || 0;
            const inputCost = ((modelMetrics.tokens?.prompt || 0) / 1_000_000) * costPerMillionTokens;
            const outputCost = ((modelMetrics.tokens?.candidates || 0) / 1_000_000) * costPerMillionTokens; // Assuming output cost is same as input for now
            return acc + inputCost + outputCost;
          },
          0.0,
        );
        fs.appendFileSync('/tmp/gemini_session_debug.log', `\n--- handleUpdate ---\nMetrics: ${JSON.stringify(metrics)}\nTotal Tokens: ${totalTokens}\nCost Estimation: ${costEstimation}`);
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
  }, []);

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
