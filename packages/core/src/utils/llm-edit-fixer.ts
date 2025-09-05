/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { createHash } from 'node:crypto';
import { type Content, Type } from '@google/genai';
import { type BaseLlmClient } from '../core/baseLlmClient.js';
import { LruCache } from './LruCache.js';
import { DEFAULT_GEMINI_FLASH_MODEL } from '../config/models.js';
import { promptIdContext } from './promptIdContext.js';
import path from 'node:path';
import fs from 'node:fs';
const MAX_CACHE_SIZE = 50;
export const EDIT_SYS_PROMPT = fs.readFileSync(path.join(globalThis.__dirname, 'hacked_prompts_source/EDIT_SYS_PROMPT.txt'), 'utf8').trim();
export const EDIT_USER_PROMPT = fs.readFileSync(path.join(globalThis.__dirname, 'hacked_prompts_source/EDIT_USER_PROMPT.txt'), 'utf8').trim();

export interface SearchReplaceEdit {
  search: string;
  replace: string;
  noChangesRequired: boolean;
  explanation: string;
}

const SearchReplaceEditSchema = {
  type: Type.OBJECT,
  properties: {
    explanation: { type: Type.STRING },
    search: { type: Type.STRING },
    replace: { type: Type.STRING },
    noChangesRequired: { type: Type.BOOLEAN },
  },
  required: ['search', 'replace', 'explanation'],
};

const editCorrectionWithInstructionCache = new LruCache<
  string,
  SearchReplaceEdit
>(MAX_CACHE_SIZE);

/**
 * Attempts to fix a failed edit by using an LLM to generate a new search and replace pair.
 * @param instruction The instruction for what needs to be done.
 * @param old_string The original string to be replaced.
 * @param new_string The original replacement string.
 * @param error The error that occurred during the initial edit.
 * @param current_content The current content of the file.
 * @param baseLlmClient The BaseLlmClient to use for the LLM call.
 * @param abortSignal An abort signal to cancel the operation.
 * @param promptId A unique ID for the prompt.
 * @returns A new search and replace pair.
 */
export async function FixLLMEditWithInstruction(
  instruction: string,
  old_string: string,
  new_string: string,
  error: string,
  current_content: string,
  baseLlmClient: BaseLlmClient,
  abortSignal: AbortSignal,
): Promise<SearchReplaceEdit> {
  let promptId = promptIdContext.getStore();
  if (!promptId) {
    promptId = `llm-fixer-fallback-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    console.warn(
      `Could not find promptId in context. This is unexpected. Using a fallback ID: ${promptId}`,
    );
  }

  const cacheKey = createHash('sha256')
    .update(
      JSON.stringify([
        current_content,
        old_string,
        new_string,
        instruction,
        error,
      ]),
    )
    .digest('hex');
  const cachedResult = editCorrectionWithInstructionCache.get(cacheKey);
  if (cachedResult) {
    return cachedResult;
  }
  const userPrompt = EDIT_USER_PROMPT.replace('{instruction}', instruction)
    .replace('{old_string}', old_string)
    .replace('{new_string}', new_string)
    .replace('{error}', error)
    .replace('{current_content}', current_content);

  const contents: Content[] = [
    {
      role: 'user',
      parts: [{ text: userPrompt }],
    },
  ];

  const result = (await baseLlmClient.generateJson({
    contents,
    schema: SearchReplaceEditSchema,
    abortSignal,
    model: DEFAULT_GEMINI_FLASH_MODEL,
    systemInstruction: EDIT_SYS_PROMPT,
    promptId,
    maxAttempts: 1,
  })) as unknown as SearchReplaceEdit;

  editCorrectionWithInstructionCache.set(cacheKey, result);
  return result;
}

export function resetLlmEditFixerCaches_TEST_ONLY() {
  editCorrectionWithInstructionCache.clear();
}
