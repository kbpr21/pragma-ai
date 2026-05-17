"""Built-in synonym and abbreviation dictionary for ontology normalization.

Provides a configurable lookup table that the ``EntityResolver`` consults
before falling back to fuzzy string matching.  This reduces graph
fragmentation caused by entity aliases like "ML" vs "Machine Learning"
or "Inc." vs "Incorporated".

Users can extend the built-in dictionary by pointing
``config.synonym_dict_path`` at a JSON file with the same structure::

    {
      "ml": "machine learning",
      "dl": "deep learning"
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Built-in abbreviation / synonym map.
# Keys are lowercase normalized forms; values are the canonical expansion.
# This is intentionally conservative — we only include unambiguous
# technical abbreviations.
# -----------------------------------------------------------------------
_BUILTIN_SYNONYMS: Dict[str, str] = {
    # AI / ML
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "llm": "large language model",
    "rl": "reinforcement learning",
    "cv": "computer vision",
    "gpt": "generative pre-trained transformer",
    "bert": "bidirectional encoder representations from transformers",
    "rag": "retrieval-augmented generation",
    "ai": "artificial intelligence",
    "ann": "artificial neural network",
    "cnn": "convolutional neural network",
    "rnn": "recurrent neural network",
    "gan": "generative adversarial network",
    "vae": "variational autoencoder",
    "svm": "support vector machine",
    "knn": "k-nearest neighbors",
    "pca": "principal component analysis",
    "svd": "singular value decomposition",
    # Software
    "api": "application programming interface",
    "sdk": "software development kit",
    "cli": "command-line interface",
    "gui": "graphical user interface",
    "ide": "integrated development environment",
    "orm": "object-relational mapping",
    "ci": "continuous integration",
    "cd": "continuous delivery",
    "cicd": "continuous integration and continuous delivery",
    "ui": "user interface",
    "ux": "user experience",
    "db": "database",
    "sql": "structured query language",
    "nosql": "non-relational database",
    "os": "operating system",
    "vm": "virtual machine",
    "k8s": "kubernetes",
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    # Corporate suffixes → empty (stripped during normalization)
    "inc.": "",
    "inc": "",
    "corp.": "",
    "corp": "",
    "ltd.": "",
    "ltd": "",
    "llc": "",
    "co.": "",
    "plc": "",
}


class SynonymDictionary:
    """Merged synonym dictionary: built-in + user-provided.

    Lookup is O(1) via dict; the dictionary is loaded once on
    construction and cached for the lifetime of the resolver.
    """

    def __init__(self, user_dict_path: Optional[str] = None) -> None:
        self._map: Dict[str, str] = dict(_BUILTIN_SYNONYMS)
        if user_dict_path:
            self._load_user_dict(user_dict_path)

    def _load_user_dict(self, path: str) -> None:
        """Load a user-provided JSON synonym file and merge it."""
        try:
            p = Path(path)
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        self._map[str(k).lower().strip()] = str(v).strip()
                    logger.info(
                        "SynonymDictionary: loaded %d user synonyms from %s",
                        len(data),
                        path,
                    )
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not load user synonym dict %s: %s", path, e)

    def expand(self, term: str) -> Optional[str]:
        """Return the canonical form if the term is a known synonym,
        or ``None`` if no match."""
        key = term.lower().strip()
        if key in self._map:
            return self._map[key]
        # Try without trailing period (e.g. "Inc." → "inc.")
        if key.endswith(".") and key[:-1] in self._map:
            return self._map[key[:-1]]
        return None

    @property
    def size(self) -> int:
        return len(self._map)
