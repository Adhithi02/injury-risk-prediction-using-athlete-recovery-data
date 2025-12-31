#!/usr/bin/env python3
"""
Diagnostic script to show the difference between old and new feature sets.
This helps understand what features were causing the data leakage issue.
"""

import pandas as pd
import json
import os

def main():
    print("🔍 Diagnosing Feature Selection Issue")
    print("=" * 50)
    
    # Load the processed data
    data_path = "data/processed/master_dataset.csv"
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    print(f"📊 Dataset shape: {df.shape}")
    print()
    
    # Show all columns
    print("📋 All columns in dataset:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    print()
    
    # Old feature selection (includes injury)
    old_features = [c for c in df.columns if c not in ["date", "label_next_10_15"]]
    old_features = [c for c in old_features if pd.api.types.is_numeric_dtype(df[c])]
    
    # New feature selection (excludes injury)
    new_features = [c for c in df.columns if c not in ["date", "label_next_10_15", "injury"]]
    new_features = [c for c in new_features if pd.api.types.is_numeric_dtype(df[c])]
    
    print("🔴 OLD Feature Selection (WITH data leakage):")
    print(f"   Count: {len(old_features)}")
    print("   Features:", old_features)
    print()
    
    print("🟢 NEW Feature Selection (FIXED - no data leakage):")
    print(f"   Count: {len(new_features)}")
    print("   Features:", new_features)
    print()
    
    # Show the difference
    removed_features = set(old_features) - set(new_features)
    print("🚫 Removed features (data leakage sources):")
    for feat in removed_features:
        print(f"   - {feat}")
    print()
    
    # Check injury column values
    if "injury" in df.columns:
        injury_stats = df["injury"].describe()
        print("📈 Injury column statistics:")
        print(f"   - Non-null values: {df['injury'].notna().sum()}")
        print(f"   - Unique values: {df['injury'].nunique()}")
        print(f"   - Value counts: {df['injury'].value_counts().to_dict()}")
        print()
    
    # Check label column
    if "label_next_10_15" in df.columns:
        label_stats = df["label_next_10_15"].value_counts()
        print("🎯 Target variable (label_next_10_15) distribution:")
        print(f"   - 0 (No injury): {label_stats.get(0, 0)}")
        print(f"   - 1 (Injury): {label_stats.get(1, 0)}")
        print(f"   - Injury rate: {label_stats.get(1, 0) / len(df):.1%}")
        print()
    
    print("✅ Diagnosis complete!")
    print("💡 The 'injury' column was causing data leakage because:")
    print("   1. It contains actual injury events (1.0 when injury occurs)")
    print("   2. Using it as a feature means the model sees future injury info")
    print("   3. This creates backwards correlations (injury today → lower future risk)")
    print("   4. Removing it should fix the negative correlation issue")

if __name__ == "__main__":
    main()

