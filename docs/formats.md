# Supported Formats

pragma supports multiple document formats for ingestion.

## Overview

| Format | Extension | Status | Dependencies |
|--------|----------|--------|--------------|
| Text | `.txt` | ✅ | None |
| Markdown | `.md` | ✅ | None |
| PDF | `.pdf` | ✅ | pdfplumber |
| CSV | `.csv` | ✅ | pandas |
| JSON | `.json`, `.jsonl` | ✅ | None |
| Word | `.docx` | ✅ | python-docx |
| HTML | `.html`, `.htm` | ✅ | beautifulsoup4 |

## Installation

For full format support:

```bash
pip install pragma[all]

# Or individually:
pip install pragma[pdf]
pip install pragma[docx]
pip install pragma[html]
```

## Text Files (.txt)

Simple text files, one fact per line or paragraph:

```
Apple is a technology company.
Founded in 1976.
Headquartered in Cupertino.
```

```python
kb.ingest("document.txt")
```

## Markdown (.md)

Headers are treated as sections:

```markdown
# Apple

Apple Inc. is a technology company.

## History

Founded in 1976 by Steve Jobs.
```

```python
kb.ingest("readme.md")
```

## PDF (.pdf)

Extracts text from each page:

```python
kb.ingest("document.pdf")
```

Requires: `pip install pdfplumber`

## CSV (.csv)

Each row is a document:

```csv
company,description
Apple,Technology company
Google,Search and cloud
```

```python
kb.ingest("companies.csv")
# Or specify columns:
kb.ingest("companies.csv", text_column="description")
```

Requires: `pip install pandas`

## JSON (.json)

```json
[
  {"text": "Apple is a company"},
  {"text": "Google is a company"}
]
```

```python
kb.ingest("facts.json")
```

## JSONL (.jsonl)

```json
{"text": "Apple is a company"}
{"text": "Google is a company"}
```

```python
kb.ingest("facts.jsonl")
```

## Word (.docx)

```python
kb.ingest("document.docx")
```

Requires: `pip install python-docx`

## HTML (.html)

```html
<html>
  <body>
    <p>Apple is a company.</p>
  </body>
</html>
```

```python
kb.ingest("page.html")
```

Requires: `pip install beautifulsoup4`

## Direct ingestion

Pass text directly:

```python
kb.ingest("Apple is a company founded in 1976.")
```

Or a list of strings:

```python
kb.ingest([
    "Apple is a company.",
    "Google is a company.",
    "Microsoft is a company.",
])
```
