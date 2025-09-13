# Follow-up on removing line numbering feature

## Directives

- Remove the line numbering feature when displaying files in Markdown format.
- Keep the code colors.
- Remove the extra indentation.

## Milestones

- [x] Remove line numbering
- [x] Remove extra indentation
- [x] Fix JSX syntax errors
- [x] Fix indentation for list items
- [ ] Fix linting errors in the codebase
- [ ] Improve the `replace` tool to handle multiple occurrences of the same string in a file.

## What worked

- Identifying the relevant files (`packages/cli/src/ui/utils/MarkdownDisplay.tsx` and `packages/cli/src/ui/utils/CodeColorizer.tsx`).
- Removing the `showLineNumbers` setting from the settings schema, the settings migration, and the tests.
- Removing the line numbering and extra indentation from the `CodeColorizer.tsx` file.
- Refactoring the `RenderListItemInternal` component to use a single `Text` component.

## What didn't work

- The preflight checks failed due to existing linting issues in the codebase. I had to ignore these errors.
- The `replace` tool was a bit tricky to use when there were multiple occurrences of the same string in a file. I had to use it multiple times to get the desired result.
- I had some trouble with the JSX syntax when I was removing the `Box` component. I had to read the file again to identify the issue and fix it.
- The build failed because of a JSX error. I had to restore the `catch` block and fix the `map` function.
- The indentation issue for list items was caused by a `Box` component with a `width` prop. I had to remove it and use a single `Text` component.
