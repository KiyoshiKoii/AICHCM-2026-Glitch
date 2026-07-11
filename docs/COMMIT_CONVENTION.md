# 📌 Git Commit Convention

To keep the project professionally managed, easily track the commit history, and avoid conflicts during teamwork, the entire team agrees to follow a standard commit message convention (based on **Conventional Commits**).

## 1. Basic Syntax 🏗️

```text
<type>(<scope>): <short description>

[Optional body for detailed explanation]
```

### 1.1. `<type>` (Required) 🎯
Defines the primary purpose of the commit. The mandatory types are:
- ✨ `feat`: Adds a new feature (e.g., add search function, add image grid UI).
- 🐛 `fix`: Fixes a bug in the code.
- 📚 `docs`: Changes or updates documentation only (README, task.md, doc files...).
- 💎 `style`: Code formatting changes that do not affect logic (indentation, whitespace, removing draft comments...).
- ♻️ `refactor`: Rewrites code to optimize flow without changing features or fixing bugs.
- 🚀 `perf`: Performance optimization (e.g., making the code run faster, consuming less RAM).
- 🧪 `test`: Adds new tests or modifies existing Unit/UI Tests.
- 🔧 `chore`: Updates build configurations, adds libraries (like modifying `environment.yml`), updates `.gitignore`, etc.

### 1.2. `<scope>` (Optional) 🔍
Specifies the area affected by the commit (e.g., `api`, `ui`, `db`, `qdrant`, `bm25`). Helps other developers instantly know which component the commit alters.

### 1.3. `<description>` (Required) 📝
- Written in English.
- Start with an imperative, lowercase verb (e.g., add, fix, remove, update).
- Keep it concise and straight to the point (preferably under 50 characters).

---

## 2. Examples 💡

✅ **Correct Format (Recommended):**
- `feat(api): ✨ add /internal/search/visual endpoint`
- `fix(db): 🐛 resolve Qdrant payload missing metadata error`
- `docs(dev2): 📚 update semantic pipeline task documentation`
- `chore: 🔧 add sentence-transformers library to environment`
- `refactor(ui): ♻️ extract TimelineViewer component into a separate file`

❌ **Incorrect Format (STRICTLY AVOID):**
- `update code` (Unclear what feature is updated)
- `fix bug` (Unclear what bug is fixed, or in which file)
- `push code at the end of the day` (Makes it extremely difficult to track code history later)
- `finished task 1 guys` (Personal communication, should not be in a commit message)

---

## 3. Recommended Daily Workflow 🔄
To avoid losing code and minimize conflicts, strictly follow these 4 steps:
1. 📥 `git add <modified_file_name>` (Avoid using `git add .` if there are draft files you don't want to push).
2. 💾 `git commit -m "feat(module): ✨ brief description of the code changes"`
3. 🔄 `git pull origin <your_branch_name>` (Always pull the latest code to auto-sync before pushing).
4. 📤 `git push origin <your_branch_name>` (Push the code for backup).
