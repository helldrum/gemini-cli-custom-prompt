import { getCoreSystemPrompt, getCompressionPrompt } from './packages/core/dist/src/core/prompts.js';
import { SUMMARIZE_TOOL_OUTPUT_PROMPT } from './packages/core/dist/src/utils/summarizer.js';
import { EDIT_SYS_PROMPT, EDIT_USER_PROMPT } from './packages/core/dist/src/utils/llm-edit-fixer.js';
import fs from 'fs';
import path from 'path';

async function main() {
    const outputDir = 'original_prompts';
    console.log(`Exporting original prompts to ./${outputDir}/`);

    // Create directory if it doesn't exist
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir);
    }

    const prompts = {
        EDIT_SYS_PROMPT: EDIT_SYS_PROMPT,
        EDIT_USER_PROMPT: EDIT_USER_PROMPT,
        SUMMARIZE_TOOL_OUTPUT_PROMPT: SUMMARIZE_TOOL_OUTPUT_PROMPT,
        COMPRESSION_PROMPT: getCompressionPrompt(),
        CORE_SYSTEM_PROMPT: getCoreSystemPrompt()
    };

    for (const [key, value] of Object.entries(prompts)) {
        const filePath = path.join(outputDir, `${key}.txt`);
        // Use trim() to remove leading/trailing whitespace from prompts
        fs.writeFileSync(filePath, value.trim(), 'utf8');
        console.log(`- Wrote ${filePath}`);
    }

    console.log('\nExport complete.');
}

main().catch(console.error);
