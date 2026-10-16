"""Utilities for loading a local JEFP corpus and extracting 19 stylometric features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import re

import nltk
import numpy as np
import pandas as pd

from data_handling.stylometric_features import extract_19_features_ptb
from feature_selection import ALL_FEATURES

CATEGORY_COLUMN_ALIASES = [
    'category', 'genre', 'label', 'jefp_category', 'class', 'type',
]
PATH_COLUMN_ALIASES = [
    'path', 'file_path', 'filepath', 'text_path', 'relative_path', 'filename', 'file_name'
]
ID_COLUMN_ALIASES = [
    'id', 'doc_id', 'document_id', 'fileid', 'file_id', 'text_id', 'title'
]

CATEGORY_MAP = {
    'canonical': 'canonical_fiction',
    'canonical fiction': 'canonical_fiction',
    'canonical_fiction': 'canonical_fiction',
    'noncanonical': 'noncanonical_fiction',
    'non-canonical': 'noncanonical_fiction',
    'non canonical': 'noncanonical_fiction',
    'non-canonical fiction': 'noncanonical_fiction',
    'noncanonical fiction': 'noncanonical_fiction',
    'noncanonical_fiction': 'noncanonical_fiction',
    'fiction canonical': 'canonical_fiction',
    'fiction noncanonical': 'noncanonical_fiction',
    'fiction': 'canonical_fiction',
    'non-fiction': 'non_fiction',
    'non fiction': 'non_fiction',
    'non_fiction': 'non_fiction',
    'nonfiction': 'non_fiction',
}


@dataclass
class JefpValidationError(Exception):
    message: str

    def __str__(self) -> str:  # pragma: no cover - tiny helper
        return self.message


def _find_column(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    lower_lookup = {c.lower().strip(): c for c in df.columns}
    for alias in aliases:
        if alias in lower_lookup:
            return lower_lookup[alias]
    return None


def normalize_jefp_category(raw_category: object) -> str:
    if raw_category is None or (isinstance(raw_category, float) and np.isnan(raw_category)):
        return 'unknown'
    normalized = re.sub(r'\s+', ' ', str(raw_category).strip().lower())
    return CATEGORY_MAP.get(normalized, 'unknown')


def derive_binary_label(category: str) -> str:
    if category in {'canonical_fiction', 'noncanonical_fiction'}:
        return 'fiction'
    if category == 'non_fiction':
        return 'non_fiction'
    return 'unknown'


def _safe_read_text(path: Path, encodings: Iterable[str]) -> str:
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise JefpValidationError(
        f"Unable to decode file '{path}' with encodings {list(encodings)}"
    )


def discover_text_files(text_root: Path) -> Dict[str, Path]:
    if not text_root.exists():
        raise FileNotFoundError(f"JEFP text root does not exist: {text_root}")
    txt_files = sorted(text_root.rglob('*.txt'))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found under JEFP text root: {text_root}")

    discovered: Dict[str, Path] = {}
    for file_path in txt_files:
        stem_key = file_path.stem.lower()
        if stem_key not in discovered:
            discovered[stem_key] = file_path
    return discovered


def _split_sentences(text: str) -> List[List[str]]:
    chunks = [segment.strip() for segment in re.split(r'(?<=[.!?])\s+', text) if segment.strip()]
    if not chunks:
        chunks = [text.strip()] if text.strip() else []
    sentences = []
    for chunk in chunks:
        tokens = nltk.wordpunct_tokenize(chunk)
        if tokens:
            sentences.append(tokens)
    return sentences


def _resolve_path(row: pd.Series, root: Path, discovered: Dict[str, Path], path_col: Optional[str], id_col: Optional[str]) -> Optional[Path]:
    if path_col and pd.notna(row.get(path_col)):
        value = str(row[path_col]).strip()
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.exists() and candidate.is_file():
            return candidate

    id_candidates = []
    if id_col and pd.notna(row.get(id_col)):
        id_candidates.append(str(row[id_col]).strip())
    for col in ['filename', 'file_name', 'title']:
        if col in row.index and pd.notna(row[col]):
            id_candidates.append(str(row[col]).strip())

    for candidate_id in id_candidates:
        stem = Path(candidate_id).stem.lower()
        if stem in discovered:
            return discovered[stem]

    return None


def inspect_jefp_metadata(metadata_csv: Path, text_root: Path, sample_size: int = 5) -> Dict[str, object]:
    metadata = pd.read_csv(metadata_csv)
    category_col = _find_column(metadata, CATEGORY_COLUMN_ALIASES)
    path_col = _find_column(metadata, PATH_COLUMN_ALIASES)
    id_col = _find_column(metadata, ID_COLUMN_ALIASES)

    discovered = discover_text_files(text_root)
    resolved_count = 0
    for _, row in metadata.iterrows():
        if _resolve_path(row, text_root, discovered, path_col, id_col) is not None:
            resolved_count += 1

    category_preview = {}
    if category_col:
        normalized = metadata[category_col].map(normalize_jefp_category)
        category_preview = normalized.value_counts(dropna=False).to_dict()

    return {
        'metadata_rows': int(len(metadata)),
        'text_files_discovered': int(len(discovered)),
        'resolved_rows': int(resolved_count),
        'category_column': category_col,
        'path_column': path_col,
        'id_column': id_col,
        'normalized_category_counts': category_preview,
        'sample_rows': metadata.head(sample_size).to_dict(orient='records'),
    }


def preprocess_jefp_corpus(
    text_root: Path | str,
    metadata_csv: Path | str,
    encodings: Optional[List[str]] = None,
) -> pd.DataFrame:
    text_root = Path(text_root)
    metadata_csv = Path(metadata_csv)

    if not text_root.exists():
        raise FileNotFoundError(f"JEFP text root does not exist: {text_root}")
    if not metadata_csv.exists():
        raise FileNotFoundError(f"JEFP metadata CSV does not exist: {metadata_csv}")

    metadata = pd.read_csv(metadata_csv)
    category_col = _find_column(metadata, CATEGORY_COLUMN_ALIASES)
    if not category_col:
        raise JefpValidationError(
            f"Could not find a category column. Tried aliases: {CATEGORY_COLUMN_ALIASES}"
        )

    path_col = _find_column(metadata, PATH_COLUMN_ALIASES)
    id_col = _find_column(metadata, ID_COLUMN_ALIASES)
    discovered = discover_text_files(text_root)
    fallback_encodings = encodings or ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1']

    rows = []
    unresolved_rows = []

    for idx, row in metadata.iterrows():
        category = normalize_jefp_category(row[category_col])
        binary_label = derive_binary_label(category)
        file_path = _resolve_path(row, text_root, discovered, path_col, id_col)

        if file_path is None:
            unresolved_rows.append(idx)
            continue

        text = _safe_read_text(file_path, fallback_encodings)
        sentences = _split_sentences(text)
        words = [token for sent in sentences for token in sent]
        if not words:
            continue

        features = extract_19_features_ptb(words, sentences)
        doc_id = str(row[id_col]).strip() if id_col and pd.notna(row.get(id_col)) else file_path.stem

        item = {
            'DocumentID': doc_id,
            'FileID': file_path.stem,
            'FilePath': str(file_path),
            'Category': category,
            'BinaryLabel': binary_label,
            'Token_Count': len(words),
            'Sentence_Count': len(sentences),
            **features,
        }

        for col in metadata.columns:
            item[f"Meta_{col}"] = row[col]

        rows.append(item)

    if not rows:
        raise JefpValidationError(
            "No JEFP rows were processed. Check metadata categories, file paths, and text root."
        )

    df = pd.DataFrame(rows)

    missing_features = [f for f in ALL_FEATURES if f not in df.columns]
    if missing_features:
        raise JefpValidationError(f"Missing extracted feature columns: {missing_features}")

    if unresolved_rows:
        print(
            f"[JEFP] Warning: {len(unresolved_rows)} metadata rows had unresolved file paths "
            f"and were skipped. First unresolved row indices: {unresolved_rows[:10]}"
        )

    print(
        f"[JEFP] {len(df)} documents loaded. "
        f"Categories: {df['Category'].value_counts(dropna=False).to_dict()}"
    )

    return df
