#!/usr/bin/env node

import { customRules } from './packages/core/src/pii/gitleaksFilters.ts';
import fs from 'node:fs';
import path from 'node:path';
import { glob } from 'glob';

// --- Helper function for sequential redaction ---
function redact(text) {
  let processedText = text;
  for (const rule of customRules) {
    const pattern = new RegExp(rule.pattern(), 'g');
    processedText = processedText.replace(pattern, `[REDACTED]`);
  }
  return processedText;
}

// --- Main test runner logic ---
async function main() {
  console.log('Running PII redaction snapshot tests...');
  console.log('='.repeat(40));

  const testDir = 'redacted_test';
  const clearFiles = glob.sync(`${testDir}/*-clear.txt`);

  if (clearFiles.length === 0) {
    console.log('No test files found in `redacted_test/` directory. (e.g., `my-test-clear.txt`) ');
    return;
  }

  let passed = 0;
  let failed = 0;

  for (const clearFile of clearFiles) {
    const redactedFile = clearFile.replace('-clear.txt', '-redacted.txt');

    if (!fs.existsSync(redactedFile)) {
      console.log(`[SKIP] Missing redacted file for ${clearFile}`);
      continue;
    }

    const inputFile = fs.readFileSync(clearFile, 'utf-8');
    const expectedFile = fs.readFileSync(redactedFile, 'utf-8');

    const actualOutput = redact(inputFile);

    const normalize = (str) => str.replace(/\r\n/g, '\n').trim();

    if (normalize(actualOutput) === normalize(expectedFile)) {
      console.log(`[PASS] ${path.basename(clearFile)}`);
      passed++;
    } else {
      console.log(`[FAIL] ${path.basename(clearFile)}`);
      // Write to temp file for debugging
      fs.writeFileSync('actual.tmp', normalize(actualOutput));
      console.log('--- EXPECTED ---\n' + normalize(expectedFile));
      console.log('--- ACTUAL ---\n' + normalize(actualOutput));
      console.log('------------------');
      failed++;
    }
  }

  console.log('='.repeat(40));
  console.log(`Test summary: Passed: ${passed}, Failed: ${failed}`);

  // Exit with a non-zero code if any tests failed, for CI/automation
  if (failed > 0) {
    process.exit(1);
  }
}

main().catch(console.error);