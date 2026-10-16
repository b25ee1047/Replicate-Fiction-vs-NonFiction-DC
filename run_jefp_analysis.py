"""CLI entry point for JEFP external analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from data_handling.load_jefp import inspect_jefp_metadata
from jefp_analysis import run_jefp_external_analysis


def _parse_feature_list(raw_value: Optional[str]) -> Optional[List[str]]:
    if not raw_value:
        return None
    parts = [item.strip() for item in raw_value.split(',') if item.strip()]
    return parts or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run JEFP external evaluation using Brown-trained classifier.')
    parser.add_argument('--jefp-root', required=True, help='Local JEFP text root directory')
    parser.add_argument('--jefp-metadata', required=True, help='JEFP metadata CSV path')
    parser.add_argument('--scratch-dir', default='scratch/jefp', help='Output directory for CSV/tables')
    parser.add_argument('--plots-dir', default='plots/jefp', help='Output directory for plots')
    parser.add_argument('--six-features', default=None, help='Comma-separated 6-feature list override')
    parser.add_argument('--two-features', default=None, help='Comma-separated 2-feature list override')
    parser.add_argument('--no-derive-feature-sets', action='store_true', help='Skip RFECV-derived 6/2 feature sets')
    parser.add_argument('--inspect-only', action='store_true', help='Only inspect metadata and discovered text paths')
    return parser


def main() -> None:
    args = build_parser().parse_args()

    jefp_root = Path(args.jefp_root)
    jefp_metadata = Path(args.jefp_metadata)

    if args.inspect_only:
        summary = inspect_jefp_metadata(jefp_metadata, jefp_root)
        print('[JEFP] Metadata inspection summary:')
        for key, value in summary.items():
            print(f'  - {key}: {value}')
        return

    run_jefp_external_analysis(
        jefp_root=jefp_root,
        jefp_metadata=jefp_metadata,
        scratch_dir=Path(args.scratch_dir),
        plots_dir=Path(args.plots_dir),
        six_features=_parse_feature_list(args.six_features),
        two_features=_parse_feature_list(args.two_features),
        derive_feature_sets=not args.no_derive_feature_sets,
    )


if __name__ == '__main__':
    main()
