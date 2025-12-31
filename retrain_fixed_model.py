#!/usr/bin/env python3
"""
Script to retrain the model with the fixed feature selection (excluding injury column).
This fixes the data leakage issue that was causing negative correlations.
"""

import os
import sys
import subprocess

def main():
    print("🔧 Retraining model with fixed feature selection...")
    print("   - Excluding 'injury' column to prevent data leakage")
    print("   - This should fix the negative correlations issue")
    print()
    
    try:
        # Run the training script
        result = subprocess.run([sys.executable, "train_model.py"], 
                              capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✅ Model retraining completed successfully!")
            print("📊 Updated metadata should now exclude the 'injury' column")
            print("🎯 Feature correlations should now show expected positive values")
            print()
            print("Next steps:")
            print("1. Run the Streamlit app: streamlit run app.py")
            print("2. Check the Risk Dashboard tab")
            print("3. Verify that correlations are now positive for risk factors")
        else:
            print("❌ Error during model retraining:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running training script: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

