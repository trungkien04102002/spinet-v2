#!/usr/bin/env python3
"""
Analyze test results and calculate metrics.

Usage:
    python3 analyze_results.py quick_check.txt
"""

import re
import sys
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np


def parse_results(filename):
    """Parse results from test output file."""
    with open(filename, 'r') as f:
        content = f.read()

    # Map severity to numeric
    severity_to_num = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}

    y_true = []
    y_pred = []

    # Extract all Ground Truth and Predictions
    pattern = r'Ground Truth:\s+Spinal Canal:\s+(\w+/?\w*)\s+Left Foraminal:\s+(\w+/?\w*)\s+Right Foraminal:\s+(\w+/?\w*)\s+Predictions:\s+Spinal Canal:\s+(\w+/?\w*)\s+Left Foraminal:\s+(\w+/?\w*)\s+Right Foraminal:\s+(\w+/?\w*)'

    matches = re.findall(pattern, content)

    for match in matches:
        gt_spinal, gt_left, gt_right, pred_spinal, pred_left, pred_right = match

        # Add ground truth
        y_true.extend([
            severity_to_num[gt_spinal],
            severity_to_num[gt_left],
            severity_to_num[gt_right]
        ])

        # Add predictions
        y_pred.extend([
            severity_to_num[pred_spinal],
            severity_to_num[pred_left],
            severity_to_num[pred_right]
        ])

    return np.array(y_true), np.array(y_pred)


def main():
    if len(sys.argv) < 2:
        filename = 'quick_check.txt'
    else:
        filename = sys.argv[1]

    print("="*70)
    print("Metrics Analysis")
    print("="*70)

    # Parse results
    y_true, y_pred = parse_results(filename)

    print(f"\nTotal predictions: {len(y_pred)}")

    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class metrics (macro average)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    # Per-class metrics (for detailed view)
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Display results
    print("\n" + "="*70)
    print("OVERALL METRICS")
    print("="*70)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} (macro average)")
    print(f"Recall:    {recall:.4f} (macro average)")
    print(f"F1 Score:  {f1:.4f} (macro average)")

    print("\n" + "="*70)
    print("PER-CLASS METRICS")
    print("="*70)

    class_names = ['Normal/Mild', 'Moderate', 'Severe']
    print(f"\n{'Class':<15} {'Support':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*70)

    for i, name in enumerate(class_names):
        print(f"{name:<15} {support_per_class[i]:<10} {precision_per_class[i]:<12.4f} {recall_per_class[i]:<12.4f} {f1_per_class[i]:<12.4f}")

    print("\n" + "="*70)
    print("CONFUSION MATRIX")
    print("="*70)
    print("\nRows: Ground Truth | Columns: Predictions\n")
    print(f"{'':>15} {'Normal/Mild':<15} {'Moderate':<15} {'Severe':<15}")
    print("-"*70)
    for i, name in enumerate(class_names):
        row = cm[i]
        print(f"{name:>15} {row[0]:<15} {row[1]:<15} {row[2]:<15}")

    print("\n" + "="*70)
    print("CLASS DISTRIBUTION")
    print("="*70)
    unique, counts = np.unique(y_true, return_counts=True)
    print("\nGround Truth:")
    for cls, count in zip(unique, counts):
        print(f"  {class_names[cls]}: {count} ({count/len(y_true)*100:.1f}%)")

    unique, counts = np.unique(y_pred, return_counts=True)
    print("\nPredictions:")
    for cls, count in zip(unique, counts):
        print(f"  {class_names[cls]}: {count} ({count/len(y_pred)*100:.1f}%)")

    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    # Check if model is biased
    if len(np.unique(y_pred)) == 1:
        print("\n⚠️  WARNING: Model predicts only ONE class!")
        print(f"   All predictions are: {class_names[y_pred[0]]}")
        print("\n   This indicates:")
        print("   - Model has NOT been trained (using random weights)")
        print("   - OR model collapsed during training")
        print("\n   ✅ This is EXPECTED for untrained/random model")

    # Accuracy compared to majority baseline
    majority_class = np.bincount(y_true).argmax()
    majority_baseline = np.sum(y_true == majority_class) / len(y_true)
    print(f"\n📊 Majority class baseline: {majority_baseline:.4f} ({majority_baseline*100:.2f}%)")
    print(f"   (Always predicting '{class_names[majority_class]}')")

    if accuracy < majority_baseline:
        print(f"\n   Model accuracy ({accuracy:.2%}) is BELOW majority baseline ({majority_baseline:.2%})")
    elif accuracy == majority_baseline:
        print(f"\n   Model accuracy equals majority baseline (not learning)")
    else:
        print(f"\n   Model accuracy ({accuracy:.2%}) is ABOVE majority baseline ({majority_baseline:.2%})")

    print()


if __name__ == "__main__":
    main()
