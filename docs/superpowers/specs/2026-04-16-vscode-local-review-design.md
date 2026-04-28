# VSCode Local Review Extension - Design Spec

## Problem

Claude Code 等 AI 工具批量生成/修改代码后，用户需要逐文件 review 并标注修改意见。当前只能：
- 在 GitHub PR 上 review（需要先 push）
- 手动记录 "文件:行号 + 意见" 到文本文件（低效且易出错）

需要一个 VSCode 插件，在本地 diff view 中提供类似 GitHub web review 的体验，将所有 review comments 汇总为结构化 Markdown 文件，供 AI 批量理解并整改。

## Approach

方案 C：Context Menu + QuickInput + Decoration

- 选中代码 → 右键 "Add Review Comment" → InputBox 输入意见
- Gutter icon + 行高亮标记已 comment 的行，hover 显示 comment 内容
- Sidebar TreeView 按文件分组展示所有 comments
- Submit Review 生成结构化 Markdown 文件

## User Workflow

```
Start Review (Command Palette / StatusBar)
    │
    ├─ 自动检测 branch + HEAD commit
    ├─ StatusBar 显示 "Review: 0 comments"
    │
    ▼
用户在 diff view 或普通编辑器中：
    选中代码 → 右键 "Add Review Comment" → InputBox 输入意见
    │
    ├─ Gutter icon + 行背景高亮标记
    ├─ Hover 显示 comment 内容
    ├─ TreeView 实时更新
    ├─ StatusBar 计数递增
    │
    ▼ (重复多个文件)
    │
Submit Review (Command Palette / StatusBar)
    │
    ├─ 生成 .reviews/review_{id}.md
    ├─ 自动打开 Markdown 预览
    ├─ Session 清空，decoration 移除
    └─ StatusBar 恢复初始状态
```

两个入口场景：
- **GitLens diff view**：comment 记录 diff 的 base ref，文件路径取相对路径
- **普通编辑器**：不带 diff 上下文，只记录文件 + 行号 + 选中代码

## Data Model

```typescript
interface ReviewSession {
  id: string;              // 时间戳，如 "20260416_143022"
  branch: string;          // 当前 git branch
  baseRef: string;         // diff 对比基准 (commit hash 或 branch name)
  headRef: string;         // HEAD commit hash
  comments: ReviewComment[];
  createdAt: string;       // ISO timestamp
}

interface ReviewComment {
  id: string;              // uuid
  filePath: string;        // 相对于 workspace root
  startLine: number;       // 1-based
  endLine: number;         // 1-based
  selectedCode: string;    // 选中的代码文本
  comment: string;         // review 意见
  source: 'diff' | 'editor';
  createdAt: string;
}
```

## Output Format

```markdown
# Code Review - feat/adaptive-search (a1b2c3d)
Branch: feat/adaptive-search
Base: main
Date: 2026-04-16 14:30:22
Comments: 5

---

## src/query/executor.go:42-58
```go
func (e *Executor) Run(ctx context.Context) error {
    results, err := e.search(ctx)
    if err != nil {
        return fmt.Errorf("search failed: %w", err)
    }
}
```
错误处理应该用 merr.WrapErrServiceInternal，不要用 fmt.Errorf

---

## src/query/orderer.go:15-15
```go
scores := make([]float64, len(segments))
```
预分配大小应该用 cap 参数，这里 len 会导致零值填充

---
```

设计要点：
- 文件头包含 branch/base/commit 元信息，AI 能理解 review 上下文
- 每条 comment 独立 section，`file:line-range` 作为标题，AI 可精确定位
- 代码块用语言标记的 fenced code block，AI 能匹配原始代码
- `---` 分隔符分割 comments

输出路径：`{workspaceRoot}/.reviews/review_{session.id}.md`

## Module Architecture

```
vscode-local-review/
├── src/
│   ├── extension.ts          # 激活入口，注册命令和 provider
│   ├── session.ts            # ReviewSession 状态管理
│   ├── commands.ts           # startReview, addComment, submitReview, discardReview
│   ├── decoration.ts         # Gutter icon + 行高亮 + hover message
│   ├── treeview.ts           # Sidebar TreeView (文件 → comments 两级结构)
│   ├── markdown.ts           # Session → Markdown 转换 + 文件写入
│   └── git.ts                # branch/commit/base ref 检测
├── resources/
│   └── icons/                # gutter icon 素材
├── package.json              # commands, menus, views, statusbar 注册
└── tsconfig.json
```

模块依赖关系：

| 模块 | 职责 | 依赖 |
|------|------|------|
| session.ts | 持有 review session，增删改 comment | 无 |
| commands.ts | 4 个命令的实现 | session, git, markdown |
| decoration.ts | 监听 session 变化，更新视觉标记 | session |
| treeview.ts | TreeView 数据源 + 交互操作 | session |
| markdown.ts | session → Markdown 转换，写入文件 | session |
| git.ts | child_process 调用 git 获取元信息 | 无 |

## VSCode Integration Points (package.json contributes)

### Commands
- `localReview.start` — Start Review
- `localReview.addComment` — Add Review Comment
- `localReview.editComment` — Edit Comment (from TreeView)
- `localReview.deleteComment` — Delete Comment (from TreeView)
- `localReview.submit` — Submit Review
- `localReview.discard` — Discard Review

### Menus
- `editor/context`: "Add Review Comment" (`when: editorHasSelection && localReview.active`)
- `view/item/context`: edit/delete actions on TreeView items

### Views
- `localReview.commentsView`: Sidebar TreeView in SCM viewContainer

### StatusBar
- 左侧 StatusBar item 显示 review 状态：
  - 未开始：无显示（或 `$(comment) Start Review`）
  - 进行中：`$(comment-discussion) Review: 5 comments`
  - 点击 StatusBar → 弹出 Start/Submit 快捷操作

## Interaction Details

### Start Review
- 已有进行中 session 时提示 "Review in progress (N comments). Submit or discard?"
- 自动检测 git info；非 git 仓库可用，branch/ref 字段留空

### Add Comment
- 未选中代码时右键菜单不显示（`when: editorHasSelection && localReview.active`）
- InputBox placeholder: `"file.go:42-58 — Enter your review comment"`
- 空输入不创建 comment
- 同一行范围可加多条 comment

### Decoration
- Gutter icon：对话气泡图标，标记有 comment 的行
- 行背景：半透明黄色高亮，覆盖 startLine 到 endLine
- Hover message：显示 comment 全文（参考 GitHub 内联 comment）
- 切换文件时自动刷新；数据在 session 内存中持有

### TreeView
- 两级结构：文件名 → comments (行号 + comment 前 30 字)
- 点击 comment → 打开文件跳转到对应行并选中范围
- Inline actions：edit (pencil icon), delete (trash icon)

### Submit Review
- 写入 `{workspaceRoot}/.reviews/review_{session.id}.md`
- `.reviews/` 不存在时自动创建
- 生成后自动打开 Markdown 文件
- Session 清空，decoration 移除，StatusBar 恢复

### Discard Review
- Confirm dialog 后清空 session，不生成文件

## GitHub Web Review Pattern Mapping

| GitHub Web | 本插件 |
|------------|--------|
| "Files changed" tab | GitLens diff view / 普通编辑器 |
| 点击行号选范围 → 内联 comment 框 | 选中代码 → 右键 → InputBox |
| "Start a review" 批量模式 | `Start Review` 命令 |
| 右上角 pending badge | StatusBar `Review: N comments` |
| "Finish your review" → Submit | `Submit Review` 命令 |
| 文件 "Viewed" 勾选 | TreeView 文件折叠 |
| 内联 comment 线程 | Gutter icon + hover 显示 comment |

## Tech Stack

- Language: TypeScript
- Build: esbuild (VSCode extension 标准)
- VSCode Engine: ^1.85.0
- Dependencies: 仅 vscode API，无第三方依赖
- Git info: child_process 调用 git CLI
