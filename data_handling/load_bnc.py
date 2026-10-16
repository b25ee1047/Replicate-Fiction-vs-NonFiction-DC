"""
load_bnc.py
===========
Feature extraction from the BNC Baby corpus (XML format).

Implements the exact 19 stylometric features described in:
  Qureshi et al., "A Simple Approach to Classify Fictional and Non-Fictional Genres"
  ACL Storytelling Workshop, 2019.

BNC Baby partitioning (Section 3 of paper):
  Training/testing : fic (25 docs) + aca (30 docs)  = 55 documents
  External eval    : news (97 docs) — Table 5 of paper
  Excluded         : dem (30 spoken) — not used in paper

Tagging:
  The paper used a Stanford POS Tagger to produce Penn Treebank tags.
  Here we use NLTK's averaged perceptron tagger, which is the closest
  freely available equivalent (same 36-tag PTB tagset, minor differences
  on boundary cases).

  The Brown bigram/unigram N-gram tagger is NOT used for BNC — the paper
  never used it for BNC. Only PTB automatic tags are used.
"""

import pandas as pd
import nltk
import xml.etree.ElementTree as ET
from pathlib import Path
from data_handling.stylometric_features import (
    group_pos_counts,
    extract_19_features_ptb,
)

nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# ---------------------------------------------------------------------------
# XML parsing helpers
# ---------------------------------------------------------------------------

def _local_name(tag):
    """Strip XML namespace prefix from a tag string."""
    return tag.rsplit('}', 1)[-1]


def _read_bnc_document(xml_file):
    """Parse a BNC Baby XML file and return (words, sentences).

    BNC XML uses <w> for word tokens and <c> for punctuation tokens,
    grouped into <s> (sentence) elements.

    Returns:
        words : list of str   — all tokens in document order
        sents : list[list[str]] — tokens grouped by sentence boundary
    """
    words, sents, current_sentence = [], [], []

    for event, element in ET.iterparse(xml_file, events=('start', 'end')):
        tag = _local_name(element.tag)

        if event == 'end' and tag in {'w', 'c'}:
            # BNC tokens can have leading/trailing whitespace in XML text
            word = (element.text or '').strip()
            if word:
                words.append(word)
                current_sentence.append(word)
            element.clear()

        elif event == 'end' and tag == 's':
            if current_sentence:
                sents.append(current_sentence)
                current_sentence = []
            element.clear()

    # Flush any sentence not closed by an </s> tag
    if current_sentence:
        sents.append(current_sentence)

    # Fallback: treat entire document as one sentence
    if not sents and words:
        sents = [words]

    return words, sents


# ---------------------------------------------------------------------------
# POS grouping helper
# ---------------------------------------------------------------------------

def _group_pos_counts(tag_list, tag_groups):
    """Count tags by broad POS category using the given grouping."""
    return group_pos_counts(tag_list, tag_groups)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_document_features(xml_file, category):
    """Extract the 19 stylometric features from one BNC Baby XML document.

    Tagging uses NLTK averaged perceptron tagger (PTB tagset), matching
    the paper's use of a PTB-based automatic tagger on BNC data.

    Args:
        xml_file : path-like — path to BNC XML file
        category : str       — BNC category label ('fic', 'aca', 'news')

    Returns:
        dict of 19 feature values + 'FileID' and 'Category',
        or None if the document contains no tokens.
    """
    words, sents = _read_bnc_document(xml_file)
    if not words:
        return None

    features = extract_19_features_ptb(words, sents)

    return {
        'FileID':               Path(xml_file).stem,
        'Category':             category,
        **features,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_bnc_corpus(bnc_root):
    """Load BNC Baby fic + aca documents and return a 19-feature DataFrame.

    This is the training/testing split used in Table 4 of the paper:
      fic  : 25 documents (Fiction     → label 1)
      aca  : 30 documents (Non-fiction → label 0)
    Total  : 55 documents

    Spoken documents (dem) are excluded (not in paper).
    News documents (news) are reserved for Table 5 — use load_bnc_news().

    Args:
        bnc_root : str or Path — directory containing 'fic', 'aca', 'news', 'dem' subdirs.

    Returns:
        pd.DataFrame with columns [FileID, Category, <19 features>].
    """
    bnc_root = Path(bnc_root)
    data = []

    for category in ['fic', 'aca']:
        cat_dir = bnc_root / category
        if not cat_dir.exists():
            raise FileNotFoundError(
                f"Expected '{category}' directory not found under {bnc_root}"
            )
        for xml_file in sorted(cat_dir.rglob('*.xml')):
            feat = extract_document_features(xml_file, category)
            if feat is not None:
                data.append(feat)

    if not data:
        raise FileNotFoundError(
            f"No BNC Baby XML files found under {bnc_root}; "
            "expected 'fic' and 'aca' subdirectories."
        )

    df = pd.DataFrame(data)
    print(f"[BNC/ptb] {len(df)} documents, categories: "
          f"{df['Category'].value_counts().to_dict()}")
    return df


def load_bnc_news(bnc_root):
    """Load BNC Baby news documents for external (Table 5) evaluation.

    Returns 97 news documents, all unlabelled (no Target column).
    These are tagged with the same PTB tagger used for fic/aca.

    Args:
        bnc_root : str or Path — directory containing 'news' subdir.

    Returns:
        pd.DataFrame with columns [FileID, Category, <19 features>].
    """
    bnc_root = Path(bnc_root)
    news_dir = bnc_root / 'news'
    if not news_dir.exists():
        raise FileNotFoundError(f"'news' directory not found under {bnc_root}")

    data = []
    for xml_file in sorted(news_dir.rglob('*.xml')):
        feat = extract_document_features(xml_file, 'news')
        if feat is not None:
            data.append(feat)

    df = pd.DataFrame(data)
    print(f"[BNC/news/ptb] {len(df)} documents")
    return df