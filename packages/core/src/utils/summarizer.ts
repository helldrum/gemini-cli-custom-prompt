/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ToolResult } from '../tools/tools.js';
import type {
  Content,
  GenerateContentConfig,
  GenerateContentResponse,
} from '@google/genai';
import type { GeminiClient } from '../core/client.js';
import { DEFAULT_GEMINI_FLASH_LITE_MODEL } from '../config/models.js';
import { getResponseText, partToString } from './partUtils.js';
import path from 'node:path';
import fs from 'node:fs';

export type Summarizer = (
  result: ToolResult,
  geminiClient: GeminiClient,
  abortSignal: AbortSignal,
) => Promise<string>;
export const SUMMARIZE_TOOL_OUTPUT_PROMPT = fs.readFileSync(path.join(globalThis.__dirname, 'hacked_prompts_source/SUMMARIZE_TOOL_OUTPUT_PROMPT.txt'), 'utf8').trim();

export const llmSummarizer: Summarizer = (result, geminiClient, abortSignal) =>
  summarizeToolOutput(
    partToString(result.llmContent),
    geminiClient,
    abortSignal,
  );

export async function summarizeToolOutput(
  textToSummarize: string,
  geminiClient: GeminiClient,
  abortSignal: AbortSignal,
  maxOutputTokens: number = 2000,
): Promise<string> {
  // There is going to be a slight difference here since we are comparing length of string with maxOutputTokens.
  // This is meant to be a ballpark estimation of if we need to summarize the tool output.
  if (!textToSummarize || textToSummarize.length < maxOutputTokens) {
    return textToSummarize;
  }
  const prompt = SUMMARIZE_TOOL_OUTPUT_PROMPT.replace(
    '{maxOutputTokens}',
    String(maxOutputTokens),
  ).replace('{textToSummarize}', textToSummarize);

  const contents: Content[] = [{ role: 'user', parts: [{ text: prompt }] }];
  const toolOutputSummarizerConfig: GenerateContentConfig = {
    maxOutputTokens,
  };
  try {
    const parsedResponse = (await geminiClient.generateContent(
      contents,
      toolOutputSummarizerConfig,
      abortSignal,
      DEFAULT_GEMINI_FLASH_LITE_MODEL,
    )) as unknown as GenerateContentResponse;
    return getResponseText(parsedResponse) || textToSummarize;
  } catch (error) {
    console.error('Failed to summarize tool output.', error);
    return textToSummarize;
  }
}
