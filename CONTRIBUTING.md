# Contributing to pragma

Thanks for considering a contribution! pragma is small, well-tested, and
deliberately easy to extend. This guide tells you exactly where to put your
code and which tests to add.

## Dev setup

```bash
git clone https://github.com/kbpr21/pragma-ai
cd pragma-ai
python -m venv .venv && .venv\Scripts\activate     # Windows
# source .venv/bin/activate                          # Linux/macOS
pip install -e ".[dev,pdf,docx,html]"
pre-commit install
pytest tests -q
ruff check pragma tests
mypy pragma                # warnings allowed; errors should not increase
```

We follow these conventions:

* **Style:** `ruff` (line length, import order, lints).
* **Tests:** every PR adds or updates a test under `tests/unit/`.
* **Commits:** [Conventional Commits](https://www.conventionalcommits.org/)
  (e.g. `feat(loader): add markdown front-matter parsing`).
* **No new runtime dependencies** without discussion in an issue first.
  Optional features (PDF, DOCX, HTML, etc.) live behind extras in
  `pyproject.toml`.

## Project layout

```text
pragma/
  ingestion/      # loaders, preprocessor, fact extractor
  graph/          # entity resolver, graph builder, traversal
  storage/        # SQLite schema + ORM
  query/          # decomposer, retriever, assembler, synthesizer
  llm/            # provider implementations
  prompts/        # editable .txt prompt files
  eval/           # offline evaluation harness
  cli/            # Typer CLI
tests/
  unit/           # fast, deterministic, LLM-free
  integration/    # slower end-to-end flows
  benchmarks/     # token / latency / accuracy harnesses
```

## Adding a document loader

Loaders live in `pragma/ingestion/loaders/<format>.py` and return a
`List[DocumentSegment]`.

```python
# pragma/ingestion/loaders/myformat.py
from pathlib import Path
from typing import List

from pragma.ingestion.loader import DocumentSegment


def load_myformat_file(path: Path) -> List[DocumentSegment]:
    """Load <format> files into one or more segments.

    Each segment becomes an independently-extracted unit. Add structural
    metadata (page, line, row, header) so downstream consumers can cite it.
    """
    text = path.read_text(encoding="utf-8")
    return [
        DocumentSegment(
            content=text,
            source=str(path),
            doc_type="myformat",
            metadata={"filename": path.name, "char_count": len(text)},
        )
    ]
```

Then wire it up in `pragma/ingestion/loader.py` (the `load_file` dispatcher
that maps file extensions to loader functions) and add at least:

* a unit test in `tests/unit/test_loaders.py`
* a fixture file under `tests/unit/fixtures/`
* a row in the **Supported document formats** table in `README.md`

Heavy parser dependencies (e.g. `pdfplumber`, `python-docx`) **must** be
imported inside the loader function and degrade gracefully with a warning
when the optional extra is not installed (see
`pragma/ingestion/loaders/pdf.py` for the pattern). Add the dependency to a
new `[project.optional-dependencies]` extra in `pyproject.toml`.

## Adding an LLM provider

Providers live in `pragma/llm/<name>.py` and implement the `LLMProvider`
Protocol from `pragma/llm/base.py`:

```python
class LLMProvider(Protocol):
    @property
    def model_name(self) -> str: ...
    def complete(self, messages: List[Dict[str, str]], **kwargs) -> str: ...
    async def acomplete(self, messages: List[Dict[str, str]], **kwargs) -> str: ...
    async def stream_complete(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncGenerator[str, None]: ...
```

Skeleton:

```python
# pragma/llm/myprovider.py
import asyncio
import functools
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from pragma.llm.base import LLMError


class MyProvider:
    BASE_URL = "https://api.example.com/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "default-model",
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("MYPROVIDER_API_KEY", "")
        if not self.api_key:
            raise LLMError("MYPROVIDER_API_KEY not set")
        self.model = model
        self.timeout = timeout

    @property
    def model_name(self) -> str:
        return self.model

    def complete(self, messages: List[Dict[str, str]], **kw: Any) -> str:
        last: Exception = LLMError("unknown")
        for attempt in range(3):
            try:
                # ... POST to BASE_URL, parse, return str ...
                return "..."
            except httpx.HTTPError as e:
                last = e
                if attempt < 2:
                    time.sleep(2 ** attempt)
        raise LLMError(f"MyProvider failed after 3 attempts: {last}")

    async def acomplete(self, messages: List[Dict[str, str]], **kw: Any) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, functools.partial(self.complete, messages, **kw)
        )

    async def stream_complete(
        self, messages: List[Dict[str, str]], **kw: Any
    ) -> AsyncGenerator[str, None]:
        # ... stream tokens via httpx.AsyncClient.stream(...) ...
        if False:
            yield ""  # pragma: no cover
```

Then:

1. Register in `pragma/llm/__init__.py`'s `get_provider(name)` dispatch.
2. Add to `pragma/cli/main.py`'s auto-detect (`get_llm()`).
3. Add a row to **Providers** in `README.md`.
4. Add a unit test that mocks `httpx` (see `tests/unit/test_groq.py` for the
   pattern using `respx`).

The required retry / async-via-executor / stream skeleton is identical
across all five built-in providers — copy from `pragma/llm/groq.py` if in
doubt.

## Adding or changing a prompt

Edit `pragma/prompts/<name>.txt` directly. The `_BUILTIN_*_PROMPT` constants
in extractor / decomposer / synthesizer are only fallbacks used if the file
is missing. Users can also override any prompt at runtime:

```bash
PRAGMA_PROMPT_FACT_EXTRACTION=/path/to/custom.txt python my_script.py
```

## Running the test suite

```bash
pytest tests/unit -q                       # fast, no network
pytest tests/integration -q                # slower
pytest tests/benchmarks -q                 # token/latency/accuracy
pytest tests -q --cov=pragma --cov-report=term-missing
```

CI runs the full matrix (Linux/macOS/Windows × Python 3.9–3.12) on every PR.
A PR is mergeable when:

* all unit + integration tests pass on all matrix legs;
* `ruff check pragma tests` is clean;
* mypy doesn't introduce new errors (existing baseline is allowed);
* `python -m build && twine check dist/*` succeeds.

## Reporting bugs

Open an issue with:

1. pragma version (`pip show pragma-ai`)
2. Python + OS
3. Minimum reproducible snippet
4. Expected vs actual behaviour
5. Full traceback if any

For LLM-related issues, please **redact API keys** from logs and stack
traces before posting.

## Code of conduct

Be kind. Disagree on the technical merits. We don't ship code authored under
duress, by harassment, or via plagiarism. PRs that violate this will be
closed.

## License

By submitting a contribution you agree that it will be released under the
[MIT License](LICENSE), the same license that covers the rest of pragma.
