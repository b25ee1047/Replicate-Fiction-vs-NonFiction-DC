"""Shared stylometric feature utilities (PTB-tag based 19 features)."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List

import nltk
import numpy as np

PTB_TAG_GROUPS = {
    'ADJ': {'JJ', 'JJR', 'JJS'},
    'ADV': {'RB', 'RBR', 'RBS', 'WRB'},
    'NOUN': {'NN', 'NNS', 'NNP', 'NNPS'},
    'VERB': {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'},
    'PRON': {'PRP', 'WP'},
}

PUNCT_SET = {
    '`', '-', ',', ';', ':', '!', '?', '/', '.', "'",
    '(', ')', '[', ']', '{', '}', '$', '*', '&', '%', '+',
    '--', '``', "''",
    '\u2014',
    '\u2013',
    '\u201c',
    '\u201d',
    '\u2018',
    '\u2019',
}

HYPHEN_TOKENS = {'-', '--', '\u2014', '\u2013'}
QUOTE_TOKENS = {"'", '"', '``', "''", '\u201c', '\u201d', '\u2018', '\u2019'}
EXCLAIM_TOKENS = {'!'}
QUESTION_TOKENS = {'?'}


def group_pos_counts(tag_list: Iterable[str], tag_groups: Dict[str, set]) -> Dict[str, int]:
    grouped = {cat: 0 for cat in tag_groups}
    for tag in tag_list:
        for cat, tags in tag_groups.items():
            if tag in tags:
                grouped[cat] += 1
                break
    return grouped


def extract_19_features_ptb(words: List[str], sents: List[List[str]]) -> Dict[str, float]:
    sent_lengths = [len(s) for s in sents]
    word_lengths = [len(w) for w in words if w.isalnum()]

    chunk_size = 100
    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]

    ttr_list, punct_ttr_list = [], []
    for chunk in chunks:
        if not chunk:
            continue
        ttr_list.append(len(chunk) / len(set(chunk)))

        punct_chunk = [w for w in chunk if w in PUNCT_SET]
        if punct_chunk:
            punct_ttr_list.append(len(punct_chunk) / len(set(punct_chunk)))

    counts = Counter(words)
    hyphens = sum(counts.get(t, 0) for t in HYPHEN_TOKENS)
    quotes = sum(counts.get(t, 0) for t in QUOTE_TOKENS)
    exclaims = sum(counts.get(t, 0) for t in EXCLAIM_TOKENS)
    questions = sum(counts.get(t, 0) for t in QUESTION_TOKENS)

    tagged_raw = nltk.pos_tag(words)
    raw_tags = [t for _, t in tagged_raw]
    pos = group_pos_counts(raw_tags, PTB_TAG_GROUPS)

    def ratio(num: float, den: float) -> float:
        return num / den if den > 0 else 0.0

    return {
        'Avg_Sentence_Length': np.mean(sent_lengths),
        'Avg_Word_Length': np.mean(word_lengths) if word_lengths else 0.0,
        'Std_Sentence_Length': np.std(sent_lengths),
        'Std_Word_Length': np.std(word_lengths) if word_lengths else 0.0,
        'Avg_Token_Type': np.mean(ttr_list) if ttr_list else 0.0,
        'Std_Token_Type': np.std(ttr_list) if ttr_list else 0.0,
        'Avg_Punct_Token_Type': np.mean(punct_ttr_list) if punct_ttr_list else 0.0,
        'Std_Punct_Token_Type': np.std(punct_ttr_list) if punct_ttr_list else 0.0,
        'Hyphen_Quote': ratio(hyphens, quotes),
        'Hyphen_Exclamation': ratio(hyphens, exclaims),
        'Quote_Question': ratio(quotes, questions),
        'Adverb_Noun': ratio(pos['ADV'], pos['NOUN']),
        'Adverb_Pronoun': ratio(pos['ADV'], pos['PRON']),
        'Adjective_Verb': ratio(pos['ADJ'], pos['VERB']),
        'Noun_Verb': ratio(pos['NOUN'], pos['VERB']),
        'Verb_Pronoun': ratio(pos['VERB'], pos['PRON']),
        'Adverb_Adjective': ratio(pos['ADV'], pos['ADJ']),
        'Adjective_Pronoun': ratio(pos['ADJ'], pos['PRON']),
        'Noun_Pronoun': ratio(pos['NOUN'], pos['PRON']),
    }
