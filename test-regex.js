import { promises as fs } from 'node:fs';
import * as hacked from './hacked_prompts.js';

// This is the function from the esbuild plugin
const escapeForTemplate = (str) => {
  if (!str) return '';
  return str.replace(/\\/g, '\\').replace(/`/g, '\`').replace(/\${/g, '\${');
};

async function test() {
  // Read the file content
  let contents = await fs.readFile('./packages/core/dist/src/utils/llm-edit-fixer.js', 'utf8');

  // This is the replacement logic from the plugin
  contents = contents.replace(
    /export const EDIT_SYS_PROMPT = `[\s\S]+?`;/,
    'export const EDIT_SYS_PROMPT = `' + escapeForTemplate(hacked.EDIT_SYS_PROMPT) + '`;'
  );
  contents = contents.replace(
    /export const EDIT_USER_PROMPT = `[\s\S]+?`;/,
    'export const EDIT_USER_PROMPT = `' + escapeForTemplate(hacked.EDIT_USER_PROMPT) + '`;'
  );

  // Print the result to the console
  console.log('--- REPLACED CONTENT ---');
  console.log(contents);
}

test().catch(console.error);