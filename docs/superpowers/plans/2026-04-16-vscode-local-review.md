# VSCode Local Review Extension - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a VSCode extension that lets users add review comments on code selections (in diff views and normal editors), then export all comments as a structured Markdown file for AI batch processing.

**Architecture:** Single-extension, no external dependencies. A `ReviewSession` held in memory accumulates `ReviewComment` objects. Context menu on text selection triggers comment creation. Decoration API renders visual markers. TreeView provides sidebar management. On submit, a Markdown renderer serializes the session to `.reviews/review_{id}.md`.

**Tech Stack:** TypeScript, VSCode Extension API (^1.85.0), esbuild bundler, child_process for git CLI.

**Spec:** `docs/superpowers/specs/2026-04-16-vscode-local-review-design.md`

---

## File Structure

```
vscode-local-review/
├── src/
│   ├── extension.ts        # Extension activate/deactivate, registers all providers
│   ├── session.ts          # ReviewSession class: state, add/edit/delete/clear comments
│   ├── git.ts              # Git helpers: getBranch(), getHeadRef(), detectBaseRef()
│   ├── commands.ts         # Command handlers: start, addComment, submit, discard
│   ├── decoration.ts       # DecorationManager: gutter icons + line highlights + hover
│   ├── treeview.ts         # ReviewTreeProvider: two-level tree (file → comments)
│   └── markdown.ts         # MarkdownRenderer: session → markdown string, write to disk
├── resources/
│   └── icons/
│       ├── comment.svg     # Gutter icon for commented lines
│       ├── comment-dark.svg
│       └── comment-light.svg
├── package.json            # Extension manifest: commands, menus, views, activation
├── tsconfig.json           # TypeScript config
├── esbuild.js              # Build script
├── .vscodeignore           # Marketplace packaging excludes
└── README.md               # Usage docs
```

---

### Task 1: Project Scaffold + Package Manifest

**Files:**
- Create: `vscode-local-review/package.json`
- Create: `vscode-local-review/tsconfig.json`
- Create: `vscode-local-review/esbuild.js`
- Create: `vscode-local-review/.vscodeignore`
- Create: `vscode-local-review/.gitignore`
- Create: `vscode-local-review/src/extension.ts` (minimal activate/deactivate)

- [ ] **Step 1: Create project directory and package.json**

```bash
mkdir -p ~/Work/code/vscode-local-review/src
mkdir -p ~/Work/code/vscode-local-review/resources/icons
cd ~/Work/code/vscode-local-review
git init
```

Create `package.json`:

```json
{
  "name": "vscode-local-review",
  "displayName": "Local Review",
  "description": "Add review comments on code selections, export as structured Markdown for AI batch processing",
  "version": "0.1.0",
  "publisher": "xianliang-li",
  "engines": {
    "vscode": "^1.85.0"
  },
  "categories": ["Other"],
  "activationEvents": [],
  "main": "./dist/extension.js",
  "contributes": {
    "commands": [
      {
        "command": "localReview.start",
        "title": "Start Review",
        "category": "Local Review"
      },
      {
        "command": "localReview.addComment",
        "title": "Add Review Comment",
        "category": "Local Review"
      },
      {
        "command": "localReview.editComment",
        "title": "Edit Comment",
        "category": "Local Review",
        "icon": "$(edit)"
      },
      {
        "command": "localReview.deleteComment",
        "title": "Delete Comment",
        "category": "Local Review",
        "icon": "$(trash)"
      },
      {
        "command": "localReview.submit",
        "title": "Submit Review",
        "category": "Local Review"
      },
      {
        "command": "localReview.discard",
        "title": "Discard Review",
        "category": "Local Review"
      }
    ],
    "menus": {
      "editor/context": [
        {
          "command": "localReview.addComment",
          "when": "editorHasSelection && localReview.active",
          "group": "localReview@1"
        }
      ],
      "view/item/context": [
        {
          "command": "localReview.editComment",
          "when": "view == localReview.commentsView && viewItem == comment",
          "group": "inline@1"
        },
        {
          "command": "localReview.deleteComment",
          "when": "view == localReview.commentsView && viewItem == comment",
          "group": "inline@2"
        }
      ]
    },
    "viewsContainers": {
      "activitybar": [
        {
          "id": "localReview",
          "title": "Local Review",
          "icon": "$(comment-discussion)"
        }
      ]
    },
    "views": {
      "localReview": [
        {
          "id": "localReview.commentsView",
          "name": "Review Comments",
          "when": "localReview.active"
        }
      ]
    }
  },
  "scripts": {
    "compile": "node esbuild.js",
    "watch": "node esbuild.js --watch",
    "check-types": "tsc --noEmit",
    "vscode:prepublish": "node esbuild.js --production",
    "package": "vsce package"
  },
  "devDependencies": {
    "@types/vscode": "^1.85.0",
    "@types/node": "^20.0.0",
    "esbuild": "^0.20.0",
    "typescript": "^5.3.0"
  }
}
```

- [ ] **Step 2: Create tsconfig.json**

```json
{
  "compilerOptions": {
    "module": "commonjs",
    "target": "ES2022",
    "lib": ["ES2022"],
    "outDir": "dist",
    "rootDir": "src",
    "sourceMap": true,
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

- [ ] **Step 3: Create esbuild.js**

```javascript
const esbuild = require('esbuild');

const production = process.argv.includes('--production');
const watch = process.argv.includes('--watch');

async function main() {
  const ctx = await esbuild.context({
    entryPoints: ['src/extension.ts'],
    bundle: true,
    format: 'cjs',
    minify: production,
    sourcemap: !production,
    sourcesContent: false,
    platform: 'node',
    outfile: 'dist/extension.js',
    external: ['vscode'],
    logLevel: 'silent',
  });
  if (watch) {
    await ctx.watch();
    console.log('Watching for changes...');
  } else {
    await ctx.rebuild();
    await ctx.dispose();
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
```

- [ ] **Step 4: Create .vscodeignore and .gitignore**

`.vscodeignore`:
```
.vscode/**
src/**
node_modules/**
tsconfig.json
esbuild.js
.gitignore
```

`.gitignore`:
```
node_modules/
dist/
*.vsix
```

- [ ] **Step 5: Create minimal extension.ts**

```typescript
import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
  console.log('Local Review extension activated');
}

export function deactivate() {}
```

- [ ] **Step 6: Install deps and verify build**

```bash
cd ~/Work/code/vscode-local-review
npm install
npm run compile
```

Expected: `dist/extension.js` exists, no errors.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: project scaffold with package manifest and esbuild"
```

---

### Task 2: ReviewSession State Management

**Files:**
- Create: `vscode-local-review/src/session.ts`

- [ ] **Step 1: Implement ReviewSession class**

```typescript
import * as vscode from 'vscode';
import { randomUUID } from 'crypto';

export interface ReviewComment {
  id: string;
  filePath: string;
  startLine: number;
  endLine: number;
  selectedCode: string;
  comment: string;
  source: 'diff' | 'editor';
  createdAt: string;
}

export interface ReviewSessionData {
  id: string;
  branch: string;
  baseRef: string;
  headRef: string;
  comments: ReviewComment[];
  createdAt: string;
}

export class ReviewSession {
  private _data: ReviewSessionData | null = null;
  private readonly _onDidChange = new vscode.EventEmitter<void>();
  readonly onDidChange = this._onDidChange.event;

  get active(): boolean {
    return this._data !== null;
  }

  get data(): ReviewSessionData | null {
    return this._data;
  }

  get comments(): ReviewComment[] {
    return this._data?.comments ?? [];
  }

  get commentCount(): number {
    return this._data?.comments.length ?? 0;
  }

  start(branch: string, baseRef: string, headRef: string): void {
    const now = new Date();
    const id = now.toISOString().replace(/[-:T]/g, '').slice(0, 15);
    this._data = {
      id,
      branch,
      baseRef,
      headRef,
      comments: [],
      createdAt: now.toISOString(),
    };
    this._onDidChange.fire();
  }

  addComment(
    filePath: string,
    startLine: number,
    endLine: number,
    selectedCode: string,
    comment: string,
    source: 'diff' | 'editor',
  ): ReviewComment {
    if (!this._data) {
      throw new Error('No active review session');
    }
    const entry: ReviewComment = {
      id: randomUUID(),
      filePath,
      startLine,
      endLine,
      selectedCode,
      comment,
      source,
      createdAt: new Date().toISOString(),
    };
    this._data.comments.push(entry);
    this._onDidChange.fire();
    return entry;
  }

  editComment(id: string, newComment: string): void {
    if (!this._data) { return; }
    const entry = this._data.comments.find(c => c.id === id);
    if (entry) {
      entry.comment = newComment;
      this._onDidChange.fire();
    }
  }

  deleteComment(id: string): void {
    if (!this._data) { return; }
    this._data.comments = this._data.comments.filter(c => c.id !== id);
    this._onDidChange.fire();
  }

  clear(): void {
    this._data = null;
    this._onDidChange.fire();
  }

  dispose(): void {
    this._onDidChange.dispose();
  }
}
```

- [ ] **Step 2: Verify build**

```bash
npm run check-types && npm run compile
```

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add src/session.ts
git commit -m "feat: ReviewSession state management with event emitter"
```

---

### Task 3: Git Helpers

**Files:**
- Create: `vscode-local-review/src/git.ts`

- [ ] **Step 1: Implement git helpers**

```typescript
import { execSync } from 'child_process';

function git(cwd: string, args: string): string {
  try {
    return execSync(`git ${args}`, { cwd, encoding: 'utf-8' }).trim();
  } catch {
    return '';
  }
}

export function getBranch(cwd: string): string {
  return git(cwd, 'rev-parse --abbrev-ref HEAD');
}

export function getHeadRef(cwd: string): string {
  return git(cwd, 'rev-parse --short HEAD');
}

export function detectBaseRef(cwd: string): string {
  // Try common base branches: main, master
  const branch = getBranch(cwd);
  if (!branch || branch === 'HEAD') { return ''; }

  for (const base of ['main', 'master']) {
    const mergeBase = git(cwd, `merge-base ${base} HEAD`);
    if (mergeBase) {
      return base;
    }
  }
  // Fallback: parent commit
  return git(cwd, 'rev-parse --short HEAD~1');
}

export function isGitRepo(cwd: string): boolean {
  return git(cwd, 'rev-parse --is-inside-work-tree') === 'true';
}
```

- [ ] **Step 2: Verify build**

```bash
npm run check-types && npm run compile
```

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add src/git.ts
git commit -m "feat: git helpers for branch/ref detection"
```

---

### Task 4: Markdown Renderer

**Files:**
- Create: `vscode-local-review/src/markdown.ts`

- [ ] **Step 1: Implement Markdown renderer**

```typescript
import * as vscode from 'vscode';
import * as path from 'path';
import { ReviewSessionData, ReviewComment } from './session';

function inferLanguage(filePath: string): string {
  const ext = path.extname(filePath).slice(1);
  const map: Record<string, string> = {
    ts: 'typescript', tsx: 'typescript', js: 'javascript', jsx: 'javascript',
    py: 'python', go: 'go', rs: 'rust', java: 'java', cpp: 'cpp', c: 'c',
    h: 'c', hpp: 'cpp', cs: 'csharp', rb: 'ruby', php: 'php', swift: 'swift',
    kt: 'kotlin', scala: 'scala', sh: 'bash', zsh: 'bash', yaml: 'yaml',
    yml: 'yaml', json: 'json', md: 'markdown', sql: 'sql', proto: 'protobuf',
  };
  return map[ext] ?? ext;
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  const pad = (n: number) => n.toString().padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

function formatLineRange(c: ReviewComment): string {
  return c.startLine === c.endLine ? `${c.startLine}` : `${c.startLine}-${c.endLine}`;
}

export function renderMarkdown(session: ReviewSessionData): string {
  const lines: string[] = [];

  // Header
  const shortRef = session.headRef ? ` (${session.headRef})` : '';
  const title = session.branch ? `${session.branch}${shortRef}` : `review${shortRef}`;
  lines.push(`# Code Review - ${title}`);
  if (session.branch) { lines.push(`Branch: ${session.branch}`); }
  if (session.baseRef) { lines.push(`Base: ${session.baseRef}`); }
  lines.push(`Date: ${formatDate(session.createdAt)}`);
  lines.push(`Comments: ${session.comments.length}`);
  lines.push('', '---', '');

  // Comments grouped by file, ordered by appearance
  for (const comment of session.comments) {
    const range = formatLineRange(comment);
    lines.push(`## ${comment.filePath}:${range}`);
    const lang = inferLanguage(comment.filePath);
    lines.push('```' + lang);
    lines.push(comment.selectedCode);
    lines.push('```');
    lines.push(comment.comment);
    lines.push('', '---', '');
  }

  return lines.join('\n');
}

export async function writeReviewFile(
  workspaceRoot: string,
  session: ReviewSessionData,
): Promise<vscode.Uri> {
  const reviewsDir = vscode.Uri.file(path.join(workspaceRoot, '.reviews'));
  try {
    await vscode.workspace.fs.stat(reviewsDir);
  } catch {
    await vscode.workspace.fs.createDirectory(reviewsDir);
  }

  const fileName = `review_${session.id}.md`;
  const fileUri = vscode.Uri.file(path.join(workspaceRoot, '.reviews', fileName));
  const content = renderMarkdown(session);
  await vscode.workspace.fs.writeFile(fileUri, Buffer.from(content, 'utf-8'));
  return fileUri;
}
```

- [ ] **Step 2: Verify build**

```bash
npm run check-types && npm run compile
```

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add src/markdown.ts
git commit -m "feat: Markdown renderer with language inference and file writer"
```

---

### Task 5: Decoration Manager

**Files:**
- Create: `vscode-local-review/src/decoration.ts`
- Create: `vscode-local-review/resources/icons/comment.svg`

- [ ] **Step 1: Create gutter icon SVGs**

`resources/icons/comment.svg` (used for both light and dark themes — a simple orange speech bubble):

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="#e2a52e">
  <path d="M2 2h12v8H6l-4 3V2z"/>
</svg>
```

- [ ] **Step 2: Implement DecorationManager**

```typescript
import * as vscode from 'vscode';
import * as path from 'path';
import { ReviewSession, ReviewComment } from './session';

export class DecorationManager {
  private readonly _gutterType: vscode.TextEditorDecorationType;
  private readonly _highlightType: vscode.TextEditorDecorationType;
  private readonly _session: ReviewSession;
  private _disposables: vscode.Disposable[] = [];

  constructor(context: vscode.ExtensionContext, session: ReviewSession) {
    this._session = session;

    this._gutterType = vscode.window.createTextEditorDecorationType({
      gutterIconPath: context.asAbsolutePath('resources/icons/comment.svg'),
      gutterIconSize: '80%',
    });

    this._highlightType = vscode.window.createTextEditorDecorationType({
      backgroundColor: 'rgba(255, 213, 79, 0.15)',
      isWholeLine: true,
    });

    this._disposables.push(
      session.onDidChange(() => this._refreshAll()),
      vscode.window.onDidChangeActiveTextEditor(() => this._refreshAll()),
    );
  }

  private _refreshAll(): void {
    for (const editor of vscode.window.visibleTextEditors) {
      this._refreshEditor(editor);
    }
  }

  private _refreshEditor(editor: vscode.TextEditor): void {
    if (!this._session.active) {
      editor.setDecorations(this._gutterType, []);
      editor.setDecorations(this._highlightType, []);
      return;
    }

    const filePath = this._getRelativePath(editor.document.uri);
    if (!filePath) {
      editor.setDecorations(this._gutterType, []);
      editor.setDecorations(this._highlightType, []);
      return;
    }

    const comments = this._session.comments.filter(c => c.filePath === filePath);
    const gutterDecorations: vscode.DecorationOptions[] = [];
    const highlightDecorations: vscode.DecorationOptions[] = [];

    for (const comment of comments) {
      const startPos = new vscode.Position(comment.startLine - 1, 0);
      const endPos = new vscode.Position(comment.endLine - 1, Number.MAX_SAFE_INTEGER);
      const range = new vscode.Range(startPos, endPos);

      const hoverMessage = new vscode.MarkdownString();
      hoverMessage.appendMarkdown(`**Review Comment**\n\n${comment.comment}`);

      gutterDecorations.push({
        range: new vscode.Range(startPos, startPos),
        hoverMessage,
      });

      highlightDecorations.push({ range });
    }

    editor.setDecorations(this._gutterType, gutterDecorations);
    editor.setDecorations(this._highlightType, highlightDecorations);
  }

  private _getRelativePath(uri: vscode.Uri): string | undefined {
    const folder = vscode.workspace.getWorkspaceFolder(uri);
    if (!folder) { return undefined; }
    return path.relative(folder.uri.fsPath, uri.fsPath).replace(/\\/g, '/');
  }

  dispose(): void {
    this._gutterType.dispose();
    this._highlightType.dispose();
    this._disposables.forEach(d => d.dispose());
  }
}
```

- [ ] **Step 3: Verify build**

```bash
npm run check-types && npm run compile
```

Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add src/decoration.ts resources/icons/
git commit -m "feat: decoration manager with gutter icons and line highlights"
```

---

### Task 6: TreeView Provider

**Files:**
- Create: `vscode-local-review/src/treeview.ts`

- [ ] **Step 1: Implement ReviewTreeProvider**

```typescript
import * as vscode from 'vscode';
import { ReviewSession, ReviewComment } from './session';

type TreeItem = FileItem | CommentItem;

class FileItem extends vscode.TreeItem {
  constructor(
    public readonly filePath: string,
    public readonly commentCount: number,
  ) {
    super(filePath, vscode.TreeItemCollapsibleState.Expanded);
    this.description = `${commentCount} comment${commentCount > 1 ? 's' : ''}`;
    this.iconPath = new vscode.ThemeIcon('file');
    this.contextValue = 'file';
  }
}

class CommentItem extends vscode.TreeItem {
  constructor(public readonly comment: ReviewComment) {
    const range = comment.startLine === comment.endLine
      ? `L${comment.startLine}`
      : `L${comment.startLine}-${comment.endLine}`;
    const preview = comment.comment.length > 40
      ? comment.comment.slice(0, 40) + '...'
      : comment.comment;

    super(`${range}: ${preview}`, vscode.TreeItemCollapsibleState.None);
    this.iconPath = new vscode.ThemeIcon('comment');
    this.contextValue = 'comment';
    this.tooltip = comment.comment;
    this.command = {
      command: 'localReview.goToComment',
      title: 'Go to Comment',
      arguments: [comment],
    };
  }
}

export class ReviewTreeProvider implements vscode.TreeDataProvider<TreeItem> {
  private readonly _onDidChangeTreeData = new vscode.EventEmitter<TreeItem | undefined>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;
  private readonly _session: ReviewSession;

  constructor(session: ReviewSession) {
    this._session = session;
    session.onDidChange(() => this._onDidChangeTreeData.fire(undefined));
  }

  getTreeItem(element: TreeItem): vscode.TreeItem {
    return element;
  }

  getChildren(element?: TreeItem): TreeItem[] {
    if (!this._session.active) { return []; }

    // Root level: group by file
    if (!element) {
      const fileMap = new Map<string, ReviewComment[]>();
      for (const c of this._session.comments) {
        const list = fileMap.get(c.filePath) ?? [];
        list.push(c);
        fileMap.set(c.filePath, list);
      }
      return Array.from(fileMap.entries()).map(
        ([filePath, comments]) => new FileItem(filePath, comments.length),
      );
    }

    // File level: show comments for that file
    if (element instanceof FileItem) {
      return this._session.comments
        .filter(c => c.filePath === element.filePath)
        .map(c => new CommentItem(c));
    }

    return [];
  }

  /** Get the ReviewComment from a TreeItem (used by edit/delete commands) */
  getComment(item: TreeItem): ReviewComment | undefined {
    return item instanceof CommentItem ? item.comment : undefined;
  }
}
```

- [ ] **Step 2: Verify build**

```bash
npm run check-types && npm run compile
```

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add src/treeview.ts
git commit -m "feat: TreeView provider with file-grouped comments and inline actions"
```

---

### Task 7: Commands Implementation

**Files:**
- Create: `vscode-local-review/src/commands.ts`

- [ ] **Step 1: Implement all command handlers**

```typescript
import * as vscode from 'vscode';
import * as path from 'path';
import { ReviewSession, ReviewComment } from './session';
import { ReviewTreeProvider } from './treeview';
import { writeReviewFile } from './markdown';
import { getBranch, getHeadRef, detectBaseRef, isGitRepo } from './git';

export function registerCommands(
  context: vscode.ExtensionContext,
  session: ReviewSession,
  treeProvider: ReviewTreeProvider,
): void {
  context.subscriptions.push(
    vscode.commands.registerCommand('localReview.start', () => startReview(session)),
    vscode.commands.registerCommand('localReview.addComment', () => addComment(session)),
    vscode.commands.registerCommand('localReview.editComment', (item) => editComment(session, treeProvider, item)),
    vscode.commands.registerCommand('localReview.deleteComment', (item) => deleteComment(session, treeProvider, item)),
    vscode.commands.registerCommand('localReview.submit', () => submitReview(session)),
    vscode.commands.registerCommand('localReview.discard', () => discardReview(session)),
    vscode.commands.registerCommand('localReview.goToComment', (comment: ReviewComment) => goToComment(comment)),
  );
}

async function startReview(session: ReviewSession): Promise<void> {
  if (session.active) {
    const choice = await vscode.window.showWarningMessage(
      `Review in progress (${session.commentCount} comments). Submit or discard?`,
      'Submit', 'Discard', 'Cancel',
    );
    if (choice === 'Submit') { return submitReview(session); }
    if (choice === 'Discard') { return discardReview(session); }
    return;
  }

  const workspaceRoot = getWorkspaceRoot();
  if (!workspaceRoot) {
    vscode.window.showErrorMessage('No workspace folder open');
    return;
  }

  let branch = '';
  let headRef = '';
  let baseRef = '';

  if (isGitRepo(workspaceRoot)) {
    branch = getBranch(workspaceRoot);
    headRef = getHeadRef(workspaceRoot);
    baseRef = detectBaseRef(workspaceRoot);
  }

  session.start(branch, baseRef, headRef);
  vscode.commands.executeCommand('setContext', 'localReview.active', true);
  vscode.window.showInformationMessage('Review started');
}

async function addComment(session: ReviewSession): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.selection.isEmpty) { return; }

  const selection = editor.selection;
  const filePath = getRelativePath(editor.document.uri);
  if (!filePath) { return; }

  const startLine = selection.start.line + 1;
  const endLine = selection.end.line + 1;
  const selectedCode = editor.document.getText(selection);
  const source: 'diff' | 'editor' = editor.document.uri.scheme === 'git' ? 'diff' : 'editor';

  const placeholder = `${path.basename(filePath)}:${startLine}-${endLine} — Enter your review comment`;
  const comment = await vscode.window.showInputBox({
    prompt: 'Review Comment',
    placeHolder: placeholder,
  });

  if (!comment) { return; }

  session.addComment(filePath, startLine, endLine, selectedCode, comment, source);
}

async function editComment(
  session: ReviewSession,
  treeProvider: ReviewTreeProvider,
  item: unknown,
): Promise<void> {
  const comment = treeProvider.getComment(item as any);
  if (!comment) { return; }

  const newComment = await vscode.window.showInputBox({
    prompt: 'Edit Review Comment',
    value: comment.comment,
  });

  if (newComment !== undefined && newComment !== comment.comment) {
    session.editComment(comment.id, newComment);
  }
}

async function deleteComment(
  session: ReviewSession,
  treeProvider: ReviewTreeProvider,
  item: unknown,
): Promise<void> {
  const comment = treeProvider.getComment(item as any);
  if (!comment) { return; }

  session.deleteComment(comment.id);
}

async function submitReview(session: ReviewSession): Promise<void> {
  if (!session.active || session.commentCount === 0) {
    vscode.window.showWarningMessage('No comments to submit');
    return;
  }

  const workspaceRoot = getWorkspaceRoot();
  if (!workspaceRoot) { return; }

  const fileUri = await writeReviewFile(workspaceRoot, session.data!);
  session.clear();
  vscode.commands.executeCommand('setContext', 'localReview.active', false);

  const doc = await vscode.workspace.openTextDocument(fileUri);
  await vscode.window.showTextDocument(doc, { preview: false });
  vscode.window.showInformationMessage(`Review saved to ${path.basename(fileUri.fsPath)}`);
}

async function discardReview(session: ReviewSession): Promise<void> {
  if (!session.active) { return; }

  const confirm = await vscode.window.showWarningMessage(
    `Discard review with ${session.commentCount} comments?`,
    { modal: true },
    'Discard',
  );

  if (confirm === 'Discard') {
    session.clear();
    vscode.commands.executeCommand('setContext', 'localReview.active', false);
    vscode.window.showInformationMessage('Review discarded');
  }
}

async function goToComment(comment: ReviewComment): Promise<void> {
  const workspaceRoot = getWorkspaceRoot();
  if (!workspaceRoot) { return; }

  const fileUri = vscode.Uri.file(path.join(workspaceRoot, comment.filePath));
  const doc = await vscode.workspace.openTextDocument(fileUri);
  const editor = await vscode.window.showTextDocument(doc);

  const startPos = new vscode.Position(comment.startLine - 1, 0);
  const endPos = new vscode.Position(comment.endLine - 1, Number.MAX_SAFE_INTEGER);
  editor.selection = new vscode.Selection(startPos, endPos);
  editor.revealRange(new vscode.Range(startPos, endPos), vscode.TextEditorRevealType.InCenter);
}

function getWorkspaceRoot(): string | undefined {
  return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
}

function getRelativePath(uri: vscode.Uri): string | undefined {
  const folder = vscode.workspace.getWorkspaceFolder(uri);
  if (!folder) { return undefined; }
  return path.relative(folder.uri.fsPath, uri.fsPath).replace(/\\/g, '/');
}
```

- [ ] **Step 2: Verify build**

```bash
npm run check-types && npm run compile
```

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add src/commands.ts
git commit -m "feat: command handlers for start/add/edit/delete/submit/discard"
```

---

### Task 8: StatusBar Integration

This is wired directly in `extension.ts` — no separate module needed.

**Files:**
- Modify: `vscode-local-review/src/extension.ts`

- [ ] **Step 1: Wire up all components in extension.ts**

Replace `src/extension.ts` with the full activation logic:

```typescript
import * as vscode from 'vscode';
import { ReviewSession } from './session';
import { DecorationManager } from './decoration';
import { ReviewTreeProvider } from './treeview';
import { registerCommands } from './commands';

export function activate(context: vscode.ExtensionContext) {
  const session = new ReviewSession();
  const decorationManager = new DecorationManager(context, session);
  const treeProvider = new ReviewTreeProvider(session);

  // Register TreeView
  const treeView = vscode.window.createTreeView('localReview.commentsView', {
    treeDataProvider: treeProvider,
    showCollapseAll: true,
  });

  // StatusBar
  const statusBarItem = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Left,
    100,
  );
  statusBarItem.command = 'localReview.start';
  statusBarItem.text = '$(comment-discussion) Start Review';
  statusBarItem.tooltip = 'Start a local code review';
  statusBarItem.show();

  session.onDidChange(() => {
    if (session.active) {
      const n = session.commentCount;
      statusBarItem.text = `$(comment-discussion) Review: ${n} comment${n !== 1 ? 's' : ''}`;
      statusBarItem.command = 'localReview.submit';
      statusBarItem.tooltip = 'Click to submit review';
    } else {
      statusBarItem.text = '$(comment-discussion) Start Review';
      statusBarItem.command = 'localReview.start';
      statusBarItem.tooltip = 'Start a local code review';
    }
  });

  // Register commands
  registerCommands(context, session, treeProvider);

  // Disposables
  context.subscriptions.push(
    session,
    decorationManager,
    treeView,
    statusBarItem,
  );
}

export function deactivate() {}
```

- [ ] **Step 2: Verify build**

```bash
npm run check-types && npm run compile
```

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add src/extension.ts
git commit -m "feat: wire up all components with statusbar integration"
```

---

### Task 9: Manual Smoke Test

**Files:** None (testing only)

- [ ] **Step 1: Launch Extension Development Host**

```bash
cd ~/Work/code/vscode-local-review
code .
```

Then press `F5` to launch the Extension Development Host window.

- [ ] **Step 2: Test Start Review**

Open Command Palette (`Ctrl+Shift+P`) → "Local Review: Start Review".
Expected: StatusBar changes to "Review: 0 comments". Sidebar shows "Review Comments" panel (empty).

- [ ] **Step 3: Test Add Comment**

Open any source file → select a code range → right-click → "Add Review Comment" → type a comment → Enter.
Expected: Gutter icon appears. Line highlighted in yellow. Hover shows comment text. StatusBar increments. TreeView shows the file + comment.

- [ ] **Step 4: Test Add Comment on multiple files**

Open a second file → select code → add comment.
Expected: TreeView now shows two files, each with their comments.

- [ ] **Step 5: Test Edit/Delete from TreeView**

In TreeView, click edit (pencil) on a comment → modify text → Enter.
Expected: Comment text updates in hover and TreeView.
Click delete (trash) on a comment → confirm.
Expected: Comment removed from TreeView, decoration removed.

- [ ] **Step 6: Test Submit Review**

Command Palette → "Local Review: Submit Review".
Expected: `.reviews/review_*.md` created and opened. Content has header (branch, base, date, count) + each comment with file:line, code block, and comment text. StatusBar reverts. Decorations cleared.

- [ ] **Step 7: Test in GitLens diff view**

Open GitLens → compare branches or commits → open a diff → select changed code → right-click → Add Review Comment.
Expected: Comment is added with `source: 'diff'`, file path is relative.

- [ ] **Step 8: Commit any fixes**

If any issues found during smoke testing, fix and commit:

```bash
git add -A
git commit -m "fix: smoke test fixes"
```

---

### Task 10: README and Final Polish

**Files:**
- Create: `vscode-local-review/README.md`

- [ ] **Step 1: Write README**

```markdown
# Local Review

Add review comments on code selections in VSCode — in diff views or normal editors — and export them as a structured Markdown file. Designed for reviewing AI-generated code and feeding review comments back to AI for batch processing.

## Usage

1. **Start Review**: Command Palette → `Local Review: Start Review` (or click StatusBar)
2. **Add Comments**: Select code → Right-click → `Add Review Comment` → Type your comment
3. **Manage Comments**: Use the "Review Comments" sidebar to view, edit, or delete comments
4. **Submit Review**: Command Palette → `Local Review: Submit Review` → Markdown file generated in `.reviews/`

## Features

- Works in **diff views** (GitLens, built-in git diff) and **normal editors**
- **Gutter icons** and **line highlights** mark commented code
- **Hover** to read comment text inline
- **Sidebar TreeView** groups comments by file with edit/delete actions
- **StatusBar** shows review progress
- Output format optimized for **AI consumption** (structured Markdown with file paths, line ranges, and code blocks)

## Output Example

The generated `.reviews/review_*.md` contains:

    # Code Review - feat/my-feature (a1b2c3d)
    Branch: feat/my-feature
    Base: main
    Date: 2026-04-16 14:30:22
    Comments: 3

    ---

    ## src/handler.go:42-58
    ```go
    func Handle(ctx context.Context) error { ... }
    ```
    Error handling should use merr.WrapErrXxx

    ---
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with usage and output example"
```

- [ ] **Step 3: Verify final build and package**

```bash
npm run compile
npm run check-types
```

Expected: Clean build, no errors.

