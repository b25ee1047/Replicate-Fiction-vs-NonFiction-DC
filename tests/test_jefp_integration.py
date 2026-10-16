import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_handling.load_jefp import (
    derive_binary_label,
    discover_text_files,
    normalize_jefp_category,
    preprocess_jefp_corpus,
)
from feature_selection import ALL_FEATURES
from jefp_analysis import validate_feature_columns


class TestJefpIntegrationHelpers(unittest.TestCase):

    def test_category_normalization_and_binary_mapping(self):
        self.assertEqual(normalize_jefp_category('Canonical Fiction'), 'canonical_fiction')
        self.assertEqual(normalize_jefp_category('non-canonical fiction'), 'noncanonical_fiction')
        self.assertEqual(normalize_jefp_category('nonfiction'), 'non_fiction')
        self.assertEqual(normalize_jefp_category('mystery'), 'unknown')

        self.assertEqual(derive_binary_label('canonical_fiction'), 'fiction')
        self.assertEqual(derive_binary_label('noncanonical_fiction'), 'fiction')
        self.assertEqual(derive_binary_label('non_fiction'), 'non_fiction')
        self.assertEqual(derive_binary_label('unknown'), 'unknown')

    def test_discover_text_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / 'canonical').mkdir(parents=True)
            (root / 'canonical' / 'A001.txt').write_text('One. Two.', encoding='utf-8')
            (root / 'nonfiction').mkdir(parents=True)
            (root / 'nonfiction' / 'B001.txt').write_text('Fact text.', encoding='utf-8')

            discovered = discover_text_files(root)
            self.assertIn('a001', discovered)
            self.assertIn('b001', discovered)

    def test_validate_feature_columns_reports_missing(self):
        df = pd.DataFrame([{'Category': 'fiction', 'Avg_Sentence_Length': 1.0}])
        missing = validate_feature_columns(df, ALL_FEATURES)
        self.assertIn('Avg_Word_Length', missing)
        self.assertNotIn('Avg_Sentence_Length', missing)

    def test_missing_root_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = Path(tmpdir) / 'metadata.csv'
            pd.DataFrame([{'id': 'doc1', 'category': 'canonical_fiction', 'path': 'doc1.txt'}]).to_csv(metadata_path, index=False)
            with self.assertRaises(FileNotFoundError):
                preprocess_jefp_corpus(Path(tmpdir) / 'missing-root', metadata_path)


if __name__ == '__main__':
    unittest.main()
