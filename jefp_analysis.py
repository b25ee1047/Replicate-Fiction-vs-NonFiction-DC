"""JEFP external evaluation and reporting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from data_handling import preprocess_brown_corpus
from data_handling.load_jefp import preprocess_jefp_corpus
from feature_selection import ALL_FEATURES, NF_CATEGORIES, run_two_stage_rfecv


def _lr() -> LogisticRegression:
    return LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C=1.0,
        max_iter=1000,
        random_state=42,
    )


def _binary_target_from_brown(df_brown: pd.DataFrame) -> np.ndarray:
    return df_brown['Category'].map(lambda c: 0 if c in NF_CATEGORIES else 1).values


def _binary_target_from_jefp(df_jefp: pd.DataFrame) -> np.ndarray:
    mapping = {'non_fiction': 0, 'fiction': 1}
    return df_jefp['BinaryLabel'].map(mapping).values


def _sanitize_feature_name(name: str) -> str:
    return name.lower().replace(' ', '_').replace('/', '_')


def validate_feature_columns(df: pd.DataFrame, features: Iterable[str]) -> List[str]:
    return [f for f in features if f not in df.columns]


def build_feature_sets(
    six_features: Optional[List[str]] = None,
    two_features: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    feature_sets = {'all_19': ALL_FEATURES}
    if six_features:
        feature_sets['selected_6'] = six_features
    if two_features:
        feature_sets['selected_2'] = two_features
    return feature_sets


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_confusion_heatmap(cm: np.ndarray, title: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap='Blues')
    fig.colorbar(image, ax=ax)

    labels = ['non_fiction', 'fiction']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_category_distribution(df: pd.DataFrame, save_path: Path) -> None:
    counts = df['Category'].value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(counts))
    ax.bar(x, counts.values, color='slateblue')
    ax.set_xticks(x)
    ax.set_xticklabels(counts.index.astype(str), rotation=25, ha='right')
    ax.set_title('JEFP category distribution')
    ax.set_ylabel('Document count')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_feature_set_comparison(metrics_df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(metrics_df))
    width = 0.35

    ax.bar(x - width / 2, metrics_df['accuracy'], width, label='Accuracy', color='steelblue')
    ax.bar(x + width / 2, metrics_df['f1_fiction'], width, label='F1 (fiction)', color='tomato')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['feature_set'], rotation=20, ha='right')
    ax.set_ylim(0, 1.0)
    ax.set_title('JEFP performance by feature set')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_prediction_confidence(pred_df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    grouped = pred_df.groupby(['Category', 'PredictedLabel'])['Fiction_Probability']

    x_labels = []
    means = []
    stds = []
    for (category, pred_label), values in grouped:
        x_labels.append(f"{category}\n→ {pred_label}")
        means.append(values.mean())
        stds.append(values.std() if len(values) > 1 else 0.0)

    x = np.arange(len(x_labels))
    ax.bar(x, means, yerr=stds, color='seagreen', alpha=0.85, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha='right')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Mean fiction probability')
    ax.set_title('Prediction confidence by true category and predicted class')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_feature_distribution_by_category(df: pd.DataFrame, features: List[str], save_path: Path) -> None:
    categories = sorted(df['Category'].dropna().unique())
    n_features = len(features)
    fig, axes = plt.subplots(n_features, 1, figsize=(10, 3 * n_features), squeeze=False)

    for idx, feature in enumerate(features):
        ax = axes[idx, 0]
        means = [df.loc[df['Category'] == category, feature].mean() for category in categories]
        ax.bar(categories, means, color='mediumpurple')
        ax.set_title(f'{feature} by JEFP category')
        ax.tick_params(axis='x', rotation=20)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_canonical_breakdown(breakdown_df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(breakdown_df['group'], breakdown_df['accuracy'], color='darkorange')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Accuracy')
    ax.set_title('Canonical vs non-canonical fiction performance')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _canonical_breakdown(pred_df: pd.DataFrame, feature_set_name: str) -> pd.DataFrame:
    fiction_only = pred_df[pred_df['Category'].isin(['canonical_fiction', 'noncanonical_fiction'])].copy()
    if fiction_only.empty:
        return pd.DataFrame(columns=['feature_set', 'group', 'count', 'accuracy', 'pred_non_fiction_rate'])

    rows = []
    for group, group_df in fiction_only.groupby('Category'):
        rows.append({
            'feature_set': feature_set_name,
            'group': group,
            'count': int(len(group_df)),
            'accuracy': float((group_df['PredictedBinary'] == group_df['TrueBinary']).mean()),
            'pred_non_fiction_rate': float((group_df['PredictedBinary'] == 0).mean()),
        })
    return pd.DataFrame(rows)


def evaluate_brown_model_on_jefp(
    df_brown_train: pd.DataFrame,
    df_jefp: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    scratch_dir: Path,
    plots_dir: Path,
) -> Dict[str, pd.DataFrame]:
    _ensure_dir(scratch_dir)
    _ensure_dir(plots_dir)

    missing_labels = int((df_jefp['BinaryLabel'] == 'unknown').sum())
    if missing_labels:
        print(f"[JEFP] Warning: {missing_labels} documents have unknown binary label and are excluded from metric computation.")

    eval_df = df_jefp[df_jefp['BinaryLabel'].isin({'fiction', 'non_fiction'})].copy()
    predict_df = df_jefp.copy()
    if eval_df.empty:
        raise ValueError('JEFP evaluation requires at least one row with binary label fiction/non_fiction.')

    y_train = _binary_target_from_brown(df_brown_train)
    y_eval = _binary_target_from_jefp(eval_df)
    unique_eval_classes = sorted(set(int(v) for v in y_eval.tolist()))
    if len(unique_eval_classes) < 2:
        print(
            f"[JEFP] Warning: evaluation labels contain a single class {unique_eval_classes}; "
            "precision/recall/F1 for the absent class will be 0."
        )

    metrics_rows = []
    all_predictions = []
    all_breakdowns = []

    for feature_set_name, features in feature_sets.items():
        missing_brown = validate_feature_columns(df_brown_train, features)
        missing_jefp = validate_feature_columns(predict_df, features)
        if missing_brown or missing_jefp:
            raise ValueError(
                f"Feature set '{feature_set_name}' is missing columns. "
                f"Brown missing={missing_brown}, JEFP missing={missing_jefp}"
            )

        X_train = df_brown_train[features].values
        X_eval = eval_df[features].values
        X_predict_all = predict_df[features].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_eval_scaled = scaler.transform(X_eval)

        model = _lr()
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_eval_scaled)
        y_prob = model.predict_proba(X_eval_scaled)[:, 1]
        y_pred_all = model.predict(scaler.transform(X_predict_all))
        y_prob_all = model.predict_proba(scaler.transform(X_predict_all))[:, 1]

        cm = confusion_matrix(y_eval, y_pred, labels=[0, 1])
        metrics_rows.append({
            'feature_set': feature_set_name,
            'n_docs_evaluated': int(len(eval_df)),
            'support_non_fiction': int(np.sum(y_eval == 0)),
            'support_fiction': int(np.sum(y_eval == 1)),
            'accuracy': float(accuracy_score(y_eval, y_pred)),
            'precision_fiction': float(precision_score(y_eval, y_pred, pos_label=1, zero_division=0)),
            'recall_fiction': float(recall_score(y_eval, y_pred, pos_label=1, zero_division=0)),
            'f1_fiction': float(f1_score(y_eval, y_pred, pos_label=1, zero_division=0)),
            'precision_non_fiction': float(precision_score(y_eval, y_pred, pos_label=0, zero_division=0)),
            'recall_non_fiction': float(recall_score(y_eval, y_pred, pos_label=0, zero_division=0)),
            'f1_non_fiction': float(f1_score(y_eval, y_pred, pos_label=0, zero_division=0)),
            'cm_true_non_fiction_pred_non_fiction': int(cm[0, 0]),
            'cm_true_non_fiction_pred_fiction': int(cm[0, 1]),
            'cm_true_fiction_pred_non_fiction': int(cm[1, 0]),
            'cm_true_fiction_pred_fiction': int(cm[1, 1]),
        })

        pred_df = predict_df[['DocumentID', 'FileID', 'FilePath', 'Category', 'BinaryLabel', 'Token_Count', 'Sentence_Count']].copy()
        pred_df['feature_set'] = feature_set_name
        pred_df['TrueBinary'] = _binary_target_from_jefp(predict_df)
        pred_df['PredictedBinary'] = y_pred_all
        pred_df['PredictedLabel'] = np.where(y_pred_all == 1, 'fiction', 'non_fiction')
        pred_df['Fiction_Probability'] = y_prob_all
        pred_df['NonFiction_Probability'] = 1.0 - y_prob_all
        pred_df['Correct'] = pred_df['TrueBinary'].notna() & (pred_df['PredictedBinary'] == pred_df['TrueBinary'])
        all_predictions.append(pred_df)

        canonical_df = _canonical_breakdown(pred_df, feature_set_name)
        if not canonical_df.empty:
            all_breakdowns.append(canonical_df)

        cm_df = pd.DataFrame(
            cm,
            index=['true_non_fiction', 'true_fiction'],
            columns=['pred_non_fiction', 'pred_fiction'],
        )
        cm_df.to_csv(scratch_dir / f"confusion_matrix_{_sanitize_feature_name(feature_set_name)}.csv")
        _save_confusion_heatmap(
            cm,
            f"Confusion matrix — {feature_set_name}",
            plots_dir / f"confusion_matrix_{_sanitize_feature_name(feature_set_name)}.png",
        )

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    breakdown_df = pd.concat(all_breakdowns, ignore_index=True) if all_breakdowns else pd.DataFrame()

    metrics_df.to_csv(scratch_dir / 'metrics_by_feature_set.csv', index=False)
    predictions_df.to_csv(scratch_dir / 'document_predictions.csv', index=False)
    df_jefp['Category'].value_counts(dropna=False).rename_axis('category').reset_index(name='count').to_csv(
        scratch_dir / 'category_distribution.csv',
        index=False,
    )
    if not breakdown_df.empty:
        breakdown_df.to_csv(scratch_dir / 'canonical_noncanonical_breakdown.csv', index=False)

    _plot_category_distribution(df_jefp, plots_dir / 'category_distribution.png')
    _plot_feature_set_comparison(metrics_df, plots_dir / 'feature_set_accuracy_f1.png')

    for feature_set_name, pred_subset in predictions_df.groupby('feature_set'):
        _plot_prediction_confidence(
            pred_subset,
            plots_dir / f"prediction_confidence_{_sanitize_feature_name(feature_set_name)}.png",
        )

    feature_plot_candidates = []
    for feature_key in ['selected_6', 'selected_2', 'all_19']:
        if feature_key in feature_sets:
            feature_plot_candidates = feature_sets[feature_key]
            break
    _plot_feature_distribution_by_category(
        eval_df,
        feature_plot_candidates,
        plots_dir / 'selected_feature_distribution_by_category.png',
    )

    if not breakdown_df.empty:
        _plot_canonical_breakdown(breakdown_df, plots_dir / 'canonical_noncanonical_performance.png')

    return {
        'metrics': metrics_df,
        'predictions': predictions_df,
        'canonical_breakdown': breakdown_df,
    }


def run_jefp_external_analysis(
    jefp_root: Path | str,
    jefp_metadata: Path | str,
    scratch_dir: Path | str = Path('scratch/jefp'),
    plots_dir: Path | str = Path('plots/jefp'),
    six_features: Optional[List[str]] = None,
    two_features: Optional[List[str]] = None,
    derive_feature_sets: bool = True,
) -> Dict[str, pd.DataFrame]:
    scratch_dir = Path(scratch_dir)
    plots_dir = Path(plots_dir)

    df_jefp = preprocess_jefp_corpus(jefp_root, jefp_metadata)
    df_brown_ptb = preprocess_brown_corpus(tagset='ptb')

    selected_six = six_features
    selected_two = two_features

    if derive_feature_sets and (selected_six is None or selected_two is None):
        df_brown_gold = preprocess_brown_corpus(tagset='brown')
        rfecv_output = run_two_stage_rfecv(df_brown_gold, df_brown_ptb)
        selected_six = selected_six or rfecv_output['six_features']
        selected_two = selected_two or rfecv_output['two_features']

    feature_sets = build_feature_sets(selected_six, selected_two)
    print(f"[JEFP] Evaluating feature sets: {list(feature_sets.keys())}")

    return evaluate_brown_model_on_jefp(
        df_brown_train=df_brown_ptb,
        df_jefp=df_jefp,
        feature_sets=feature_sets,
        scratch_dir=scratch_dir,
        plots_dir=plots_dir,
    )
