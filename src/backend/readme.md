find . -name "pyproject.toml"

python -m pip install fastapi uvicorn httpx pydantic pydantic-settings python-multipart
python -m pip install pytest pytest-asyncio pytest-cov ruff

git diff --name-only --diff-filter=U
git grep -n -E '^(<<<<<<<|=======|>>>>>>>)' -- src/backend

python -m pip install -e ".[dev]"
