import path from 'path';
import { pathToFileURL } from 'url';

async function main() {
  // Get the file path from the command-line arguments
  const filePath = process.argv[2];

  if (!filePath) {
    console.error('Error: Please provide a file path as an argument.');
    console.log('Usage: node verify_prompts.js <path-to-js-module>');
    process.exit(1);
  }

  try {
    // Resolve the relative path to an absolute path and convert to a file URL for dynamic import
    const absolutePath = path.resolve(filePath);
    const moduleUrl = pathToFileURL(absolutePath).href;

    // Use dynamic import() to load the specified module
    const prompts = await import(moduleUrl);

    console.log(`--- VERIFYING PROMPTS FROM: ${absolutePath} ---`);

    for (const [key, value] of Object.entries(prompts)) {
      console.log(`\n----- VARIABLE: ${key} -----\n`);
      // If the value is a string, print it directly. Otherwise, stringify it.
      if (typeof value === 'string') {
        console.log(value);
      } else {
        console.log(JSON.stringify(value, null, 2));
      }
      console.log(`\n----- END OF VARIABLE: ${key} -----\n`);
    }

    console.log('--- VERIFICATION COMPLETE ---');
  } catch (error) {
    console.error(`Error loading or processing file: ${filePath}`);
    console.error(error);
    process.exit(1);
  }
}

main();
