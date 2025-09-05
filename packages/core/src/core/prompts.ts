/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';


import process from 'node:process';
import { isGitRepository } from '../utils/gitUtils.js';
import { GEMINI_CONFIG_DIR } from '../tools/memoryTool.js';

export function resolvePathFromEnv(envVar?: string): {
  isSwitch: boolean;
  value: string | null;
  isDisabled: boolean;
} {
  // Handle the case where the environment variable is not set, empty, or just whitespace.
  const trimmedEnvVar = envVar?.trim();
  if (!trimmedEnvVar) {
    return { isSwitch: false, value: null, isDisabled: false };
  }

  const lowerEnvVar = trimmedEnvVar.toLowerCase();
  // Check if the input is a common boolean-like string.
  if (['0', 'false', '1', 'true'].includes(lowerEnvVar)) {
    // If so, identify it as a "switch" and return its value.
    const isDisabled = ['0', 'false'].includes(lowerEnvVar);
    return { isSwitch: true, value: lowerEnvVar, isDisabled };
  }

  // If it's not a switch, treat it as a potential file path.
  let customPath = trimmedEnvVar;

  // Safely expand the tilde (~) character to the user's home directory.
  if (customPath.startsWith('~/') || customPath === '~') {
    try {
      const home = os.homedir(); // This is the call that can throw an error.
      if (customPath === '~') {
        customPath = home;
      } else {
        customPath = path.join(home, customPath.slice(2));
      }
    } catch (error) {
      // If os.homedir() fails, we catch the error instead of crashing.
      console.warn(
        `Could not resolve home directory for path: ${trimmedEnvVar}`,
        error,
      );
      // Return null to indicate the path resolution failed.
      return { isSwitch: false, value: null, isDisabled: false };
    }
  }

  // Return it as a non-switch with the fully resolved absolute path.
  return {
    isSwitch: false,
    value: path.resolve(customPath),
    isDisabled: false,
  };
}

export const SANDBOX_PROMPT_SEATBELT = `
# macOS Seatbelt
You are running under macos seatbelt with limited access to files outside the project directory or system temp directory, and with limited access to host system resources such as ports. If you encounter failures that could be due to macOS Seatbelt (e.g. if a command fails with 'Operation not permitted' or similar error), as you report the error to the user, also explain why you think it could be due to macOS Seatbelt, and how the user may need to adjust their Seatbelt profile.
`;

export const SANDBOX_PROMPT_GENERIC = `
# Sandbox
You are running in a sandbox container with limited access to files outside the project directory or system temp directory, and with limited access to host system resources such as ports. If you encounter failures that could be due to sandboxing (e.g. if a command fails with 'Operation not permitted' or similar error), when you report the error to the user, also explain why you think it could be due to sandboxing, and how the user may need to adjust their sandbox configuration.
`;

export const SANDBOX_PROMPT_NONE = `
# Outside of Sandbox
You are running outside of a sandbox container, directly on the user's system. For critical commands that are particularly likely to modify the user's system outside of the project directory or system temp directory, as you explain the command to the user (per the Explain Critical Commands rule above), also remind the user to consider enabling sandboxing.
`;

export const GIT_REPOSITORY_INFO_PROMPT = `
# Git Repository
- The current working (project) directory is being managed by a git repository.
- When asked to commit changes or prepare a commit, always start by gathering information using shell commands:
  - 
git status
 to ensure that all relevant files are tracked and staged, using 

git add ...

 as needed.
  - 

git diff HEAD

 to review all changes (including unstaged changes) to tracked files in work tree since last commit.
    - 

git diff --staged

 to review only staged changes when a partial commit makes sense or was requested by the user.
  - 

git log -n 3

 to review recent commit messages and match their style (verbosity, formatting, signature line, etc.)
- Combine shell commands whenever possible to save time/steps, e.g. 

git status && git diff HEAD && git log -n 3

.
- Always propose a draft commit message. Never just ask the user to give you the full commit message.
- Prefer commit messages that are clear, concise, and focused more on "why" and less on "what".
- Keep the user informed and ask for clarification or confirmation where needed.
- After each commit, confirm that it was successful by running 

git status

.
- If a commit fails, never attempt to work around the issues without being asked to do so.
- Never push changes to a remote repository without being asked explicitly by the user.
`;

export function getCoreSystemPrompt(userMemory?: string): string {
  // A flag to indicate whether the system prompt override is active.
  let systemMdEnabled = false;
  // The default path for the system prompt file. This can be overridden.
  let systemMdPath = path.resolve(path.join(GEMINI_CONFIG_DIR, 'system.md'));
  // Resolve the environment variable to get either a path or a switch value.
  const systemMdResolution = resolvePathFromEnv(
    process.env['GEMINI_SYSTEM_MD'],
  );

  // Proceed only if the environment variable is set and is not disabled.
  if (systemMdResolution.value && !systemMdResolution.isDisabled) {
    systemMdEnabled = true;

    // We update systemMdPath to this new custom path.
    if (!systemMdResolution.isSwitch) {
      systemMdPath = systemMdResolution.value;
    }

    // require file to exist when override is enabled
    if (!fs.existsSync(systemMdPath)) {
      throw new Error(`missing system prompt file '${systemMdPath}'`);
    }
  }
  
  const basePrompt = systemMdEnabled
    ? fs.readFileSync(systemMdPath, 'utf8')
    : fs.readFileSync(path.join(globalThis.__dirname, 'hacked_prompts_source/CORE_SYSTEM_PROMPT.txt'), 'utf8').trim();

  const sandboxStatus = (() => {
    const isSandboxExec = process.env['SANDBOX'] === 'sandbox-exec';
    const isGenericSandbox = !!process.env['SANDBOX'];

    if (isSandboxExec) {
      return SANDBOX_PROMPT_SEATBELT;
    } else if (isGenericSandbox) {
      return SANDBOX_PROMPT_GENERIC;
    } else {
      return SANDBOX_PROMPT_NONE;
    }
  })();

  const gitRepositoryInfo = (() => {
    if (isGitRepository(process.cwd())) {
      return GIT_REPOSITORY_INFO_PROMPT;
    }
    return '';
  })();

  // if GEMINI_WRITE_SYSTEM_MD is set (and not 0|false), write base system prompt to file
  const writeSystemMdResolution = resolvePathFromEnv(
    process.env['GEMINI_WRITE_SYSTEM_MD'],
  );

  // Check if the feature is enabled. This proceeds only if the environment
  // variable is set and is not explicitly '0' or 'false'.
  if (writeSystemMdResolution.value && !writeSystemMdResolution.isDisabled) {
    const writePath = writeSystemMdResolution.isSwitch
      ? systemMdPath
      : writeSystemMdResolution.value;

    fs.mkdirSync(path.dirname(writePath), { recursive: true });
    fs.writeFileSync(writePath, basePrompt);
  }

  const memorySuffix =
    userMemory && userMemory.trim().length > 0
      ? `\n\n---\n\n${userMemory.trim()}`
      : '';

  return `${sandboxStatus}${gitRepositoryInfo}${basePrompt}${memorySuffix}`;
}


/**
 * Provides the system prompt for the history compression process.
 * This prompt instructs the model to act as a specialized state manager,
 * think in a scratchpad, and produce a structured XML summary.
 */
export function getCompressionPrompt(): string {
  return fs.readFileSync(path.join(globalThis.__dirname, 'hacked_prompts_source/COMPRESSION_PROMPT.txt'), 'utf8').trim();
}
