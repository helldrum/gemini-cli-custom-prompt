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
import Gradient from 'ink-gradient';
import { MemoryUsageDisplay } from './MemoryUsageDisplay.js';
import { ContextUsageDisplay } from './ContextUsageDisplay.js';
import { DebugProfiler } from './DebugProfiler.js';
import { isDevelopment } from '../../utils/installationInfo.js';
import { useUIState } from '../contexts/UIStateContext.js';
import { useConfig } from '../contexts/ConfigContext.js';
import { useSettings } from '../contexts/SettingsContext.js';
import { useVimMode } from '../contexts/VimModeContext.js';
import { useSessionStats } from '../contexts/SessionContext.js'; // Import useSessionStats
import { uiTelemetryService } from '@google/gemini-cli-core'; // Import uiTelemetryService
import { useTerminalSize } from '../hooks/useTerminalSize.js'; // Import useTerminalSize
import { isNarrowWidth } from '../utils/isNarrowWidth.js'; // Import isNarrowWidth

export const Footer: React.FC = () => {
  const uiState = useUIState();
  const config = useConfig();
  const settings = useSettings();
  const { vimEnabled, vimMode } = useVimMode();
  const { stats } = useSessionStats(); // Use useSessionStats to get stats
  const { columns: terminalWidth } = useTerminalSize(); // Use useTerminalSize to get terminalWidth

  const { 
    model,
    targetDir,
    debugMode,
    branchName,
    debugMessage,
    corgiMode,
    errorCount,
    showErrorDetails,
    promptTokenCount,
    nightly,
    isTrustedFolder,
    mainAreaWidth,
  } = {
    model: config.getModel(),
    targetDir: config.getTargetDir(),
    debugMode: config.getDebugMode(),
    branchName: uiState.branchName,
    debugMessage: uiState.debugMessage,
    corgiMode: uiState.corgiMode,
    errorCount: uiState.errorCount,
    showErrorDetails: uiState.showErrorDetails,
    promptTokenCount: uiState.sessionStats.lastPromptTokenCount,
    nightly: uiState.nightly,
    isTrustedFolder: uiState.isTrustedFolder,
    mainAreaWidth: uiState.mainAreaWidth,
  };

  const showMemoryUsage =
    config.getDebugMode() || settings.merged.ui?.showMemoryUsage || false;
  const hideCWD = settings.merged.ui?.footer?.hideCWD || false;
  const hideSandboxStatus =
    settings.merged.ui?.footer?.hideSandboxStatus || false;
  const hideModelInfo = settings.merged.ui?.footer?.hideModelInfo || false;

  const isNarrow = isNarrowWidth(terminalWidth);

  const pathLength = Math.max(20, Math.floor(mainAreaWidth * 0.25));
  const displayPath = shortenPath(tildeifyPath(targetDir), pathLength);

  const justifyContent = hideCWD && hideModelInfo ? 'center' : 'space-between';
  const displayVimMode = vimEnabled ? vimMode : undefined;

  const showDebugProfiler = debugMode || isDevelopment;

    const totalTokens = stats.totalTokens; // Get totalTokens from stats
    const costEstimation = stats.costEstimation; // Get costEstimation from stats
  
    return (
      <Box
        justifyContent={justifyContent}
        width={mainAreaWidth}
        flexDirection="row"
        alignItems="center"
        paddingX={1}
      >
        {(showDebugProfiler || displayVimMode || !hideCWD) && (
          <Box>
            {showDebugProfiler && <DebugProfiler />}
            {displayVimMode && (
              <Text color={theme.text.secondary}>[{displayVimMode}] </Text>
            )}
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
        {/* New Box for Cost and Tokens */}
        <Box
          width="100%"
          justifyContent="flex-end" // Align to the right
          paddingX={1}
        >
          <Text color={theme.text.secondary}>
            Cost: {costEstimation.toFixed(6)} $
          </Text>
          <Text color={theme.text.secondary}> | Tokens: {totalTokens}</Text>
        </Box>
      </Box>
    );
  };