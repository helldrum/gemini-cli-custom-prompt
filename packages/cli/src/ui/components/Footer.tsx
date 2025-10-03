/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type React from 'react';
import { Box, Text } from 'ink';
import { theme } from '../semantic-colors.js';
import { shortenPath, tildeifyPath } from '@google/gemini-cli-core';
import { ConsoleSummaryDisplay } from './ConsoleSummaryDisplay.js';
import process from 'node:process';
import path from 'node:path';
import Gradient from 'ink-gradient';
import { MemoryUsageDisplay } from './MemoryUsageDisplay.js';
import { ContextUsageDisplay } from './ContextUsageDisplay.js';
import { DebugProfiler } from './DebugProfiler.js';

import { useTerminalSize } from '../hooks/useTerminalSize.js';
import { isNarrowWidth } from '../utils/isNarrowWidth.js';

const MODEL_PRICING_PER_MILLION_TOKENS: Record<string, number> = {
  'gemini-2.5-pro': 1.25, // $1.25 per million input tokens
  'gemini-2.0-flash': 0.3, // $0.30 per million input tokens
};

export interface FooterProps {
  model: string;
  targetDir: string;
  branchName?: string;
  debugMode: boolean;
  debugMessage: string;
  corgiMode: boolean;
  errorCount: number;
  showErrorDetails: boolean;
  showMemoryUsage?: boolean;
  promptTokenCount: number;
  nightly: boolean;
  vimMode?: string;
  isTrustedFolder?: boolean;
  hideCWD?: boolean;
  hideSandboxStatus?: boolean;
  hideModelInfo?: boolean;
  // costEstimation?: number; // No longer needed as a prop, will be calculated
  consumedTokens?: number;
}

export const Footer: React.FC<FooterProps> = ({
  model,
  targetDir,
  branchName,
  debugMode,
  debugMessage,
  corgiMode,
  errorCount,
  showErrorDetails,
  showMemoryUsage,
  promptTokenCount,
  nightly,
  vimMode,
  isTrustedFolder,
  hideCWD = false,
  hideSandboxStatus = false,
  hideModelInfo = false,
  // costEstimation = 0, // No longer needed as a prop, will be calculated
  consumedTokens = 0,
}) => {
  const { columns: terminalWidth } = useTerminalSize();

  const isNarrow = isNarrowWidth(terminalWidth);

  const costPerMillionTokens = MODEL_PRICING_PER_MILLION_TOKENS[model] || 0;
  const calculatedCostEstimation =
    (consumedTokens / 1_000_000) * costPerMillionTokens;

  // Adjust path length based on terminal width
  const pathLength = Math.max(20, Math.floor(terminalWidth * 0.4));
  const displayPath = isNarrow
    ? path.basename(tildeifyPath(targetDir))
    : shortenPath(tildeifyPath(targetDir), pathLength);

  const justifyContent = hideCWD && hideModelInfo ? 'center' : 'space-between';

  return (
    <Box flexDirection="column" width="100%">
      <Box
        justifyContent={justifyContent}
        width="100%"
        flexDirection={isNarrow ? 'column' : 'row'}
        alignItems={isNarrow ? 'flex-start' : 'center'}
      >
        {(debugMode || vimMode || !hideCWD) && (
          <Box>
            {debugMode && <DebugProfiler />}
            {vimMode && <Text color={theme.text.secondary}>[{vimMode}] </Text>}
            {!hideCWD &&
              (nightly ? (
                <Gradient colors={theme.ui.gradient}>
                  <Text>
                    {displayPath}
                    {branchName && <Text> ({branchName}*)</Text>}
                  </Text>
                </Gradient>
              ) : (
                <Text color={theme.text.link}>
                  {displayPath}
                  {branchName && (
                    <Text color={theme.text.secondary}> ({branchName}*)</Text>
                  )}
                </Text>
              ))}
            {debugMode && (
              <Text color={theme.status.error}>
                {' ' + (debugMessage || '--debug')}
              </Text>
            )}
          </Box>
        )}

        {/* Middle Section: Centered Trust/Sandbox Info */}
        {!hideSandboxStatus && (
          <Box
            flexGrow={isNarrow || hideCWD || hideModelInfo ? 0 : 1}
            alignItems="center"
            justifyContent={isNarrow || hideCWD ? 'flex-start' : 'center'}
            display="flex"
            paddingX={isNarrow ? 0 : 1}
            paddingTop={isNarrow ? 1 : 0}
          >
            {isTrustedFolder === false ? (
              <Text color={theme.status.warning}>untrusted</Text>
            ) : process.env['SANDBOX'] &&
              process.env['SANDBOX'] !== 'sandbox-exec' ? (
              <Text color="green">
                {process.env['SANDBOX'].replace(/^gemini-(?:cli-)?/, '')}
              </Text>
            ) : process.env['SANDBOX'] === 'sandbox-exec' ? (
              <Text color={theme.status.warning}>
                macOS Seatbelt{' '}
                <Text color={theme.text.secondary}>
                  ({process.env['SEATBELT_PROFILE']})
                </Text>
              </Text>
            ) : (
              <Text color={theme.status.error}>
                no sandbox <Text color={theme.text.secondary}>(see /docs)</Text>
              </Text>
            )}
          </Box>
        )}

        {/* Right Section: Gemini Label and Console Summary */}
        {(!hideModelInfo ||
          showMemoryUsage ||
          corgiMode ||
          (!showErrorDetails && errorCount > 0)) && (
          <Box alignItems="center" paddingTop={isNarrow ? 1 : 0}>
            {!hideModelInfo && (
              <Box alignItems="center">
                <Text color={theme.text.accent}>
                  {isNarrow ? '' : ' '}
                  {model}{' '}
                  <ContextUsageDisplay
                    promptTokenCount={promptTokenCount}
                    model={model}
                  />
                </Text>
                {showMemoryUsage && <MemoryUsageDisplay />}
              </Box>
            )}
            <Box alignItems="center" paddingLeft={2}>
              {corgiMode && (
                <Text>
                  {!hideModelInfo && <Text color={theme.ui.comment}>| </Text>}
                  <Text color={theme.status.error}>▼</Text>
                  <Text color={theme.text.primary}>(´</Text>
                  <Text color={theme.status.error}>ᴥ</Text>
                  <Text color={theme.text.primary}>`)</Text>
                  <Text color={theme.status.error}>▼ </Text>
                </Text>
              )}
              {!showErrorDetails && errorCount > 0 && (
                <Box>
                  {!hideModelInfo && <Text color={theme.ui.comment}>| </Text>}
                  <ConsoleSummaryDisplay errorCount={errorCount} />
                </Box>
              )}
            </Box>
          </Box>
        )}
      </Box>
      <Box width="100%" justifyContent="flex-start">
        <Text color={theme.text.secondary}>
          Cost: {calculatedCostEstimation.toFixed(6)} $
        </Text>
        <Text color={theme.text.secondary}> | Tokens: {consumedTokens}</Text>
      </Box>
    </Box>
  );
};
