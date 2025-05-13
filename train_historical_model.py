#!/usr/bin/env python
"""
Enhanced Options Trading Model Trainer

This script trains a machine learning model on real market data to predict
profitable options trading patterns. It uses the processed data from the 
data_collector.py and prepare_training_data.py scripts.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Directories
DATA_DIR = "training_data"
PROCESSED_DIR = f"{DATA_DIR}/processed"
MODELS_DIR = "models"

def load_training_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the processed training and testing data
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Load train and test data
    train_df = pd.read_csv(f"{PROCESSED_DIR}/option_train_data.csv")
    test_df = pd.read_csv(f"{PROCESSED_DIR}/option_test_data.csv")
    
    # Create target variable (profitable trade or not)
    # For this demo, we'll use a simple rule: call options with delta > 0.5 or put options with delta < -0.5
    # are considered favorable trading opportunities
    train_df['profitable'] = ((train_df['option_type_C'] & (train_df['delta'] > 0.5)) | 
                             (train_df['option_type_P'] & (train_df['delta'] < -0.5))).astype(int)
    test_df['profitable'] = ((test_df['option_type_C'] & (test_df['delta'] > 0.5)) | 
                            (test_df['option_type_P'] & (test_df['delta'] < -0.5))).astype(int)
    
    # Split into features and target
    feature_cols = [
        'strike', 'underlying_price', 'days_to_expiry', 'implied_volatility',
        'delta', 'gamma', 'theta', 'vega', 'moneyness', 'relative_strike',
        'bid_ask_spread', 'bid_ask_spread_percent', 'time_value',
        'option_type_C', 'option_type_P'
    ]
    
    X_train = train_df[feature_cols]
    y_train = train_df['profitable']
    X_test = test_df[feature_cols]
    y_test = test_df['profitable']
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = [col for col in feature_cols if col not in ['option_type_C', 'option_type_P']]
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_enhanced_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> Dict:
    """
    Train an enhanced model using GridSearchCV to find optimal hyperparameters
    
    Args:
        X_train: Training features
        y_train: Training targets
        
    Returns:
        Dict with the trained model and training metrics
    """
    print("Training enhanced Random Forest model...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced']
    }
    
    # Create base model
    rf = RandomForestClassifier(random_state=42)
    
    # Use GridSearchCV to find the best parameters
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    
    # Train the model
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Get cross-validation scores
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1')
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Average CV F1 score: {cv_scores.mean():.4f}")
    
    # Get feature importances
    feature_importances = pd.DataFrame(
        best_model.feature_importances_,
        index=X_train.columns,
        columns=['importance']
    ).sort_values('importance', ascending=False)
    
    # Create result dictionary
    result = {
        'model': best_model,
        'best_params': grid_search.best_params_,
        'cv_scores': cv_scores,
        'feature_importances': feature_importances
    }
    
    return result

def evaluate_model(model: object, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict:
    """
    Evaluate the trained model on test data
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dict with evaluation metrics
    """
    print("Evaluating model on test data...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    
    # Create evaluation result
    evaluation = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }
    
    return evaluation

def plot_results(model_result: Dict, evaluation: Dict, X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, output_dir: str = "models/visualizations"):
    """
    Create visualizations of model performance and feature importance
    
    Args:
        model_result: Result dictionary from train_enhanced_model
        evaluation: Evaluation dictionary from evaluate_model
        X_train: Training features
        X_test: Test features
        y_test: Test targets
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Plot feature importances
    plt.figure(figsize=(10, 8))
    feature_importances = model_result['feature_importances']
    sns.barplot(x='importance', y=feature_importances.index, data=feature_importances)
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importances.png")
    plt.close()
    
    # 2. Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = evaluation['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()
    
    # 3. Plot probability distribution by class
    plt.figure(figsize=(10, 6))
    y_proba = model_result['model'].predict_proba(X_test)[:, 1]
    
    for label in [0, 1]:
        mask = y_test == label
        plt.hist(y_proba[mask], alpha=0.7, bins=20, 
                 label=f"Class {label} ({'Unprofitable' if label == 0 else 'Profitable'})")
    
    plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
    plt.xlabel('Predicted Probability of Profitable Trade')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Probability Distribution by Class')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/probability_distribution.png")
    plt.close()
    
    # 4. Plot ROC curve
    from sklearn.metrics import roc_curve, auc
    plt.figure(figsize=(8, 8))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f"{output_dir}/roc_curve.png")
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def save_model(model: object, metrics: Dict, output_dir: str = "models"):
    """
    Save the trained model and evaluation metrics
    
    Args:
        model: Trained model
        metrics: Evaluation metrics
        output_dir: Output directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = f"{output_dir}/options_model_{timestamp}.joblib"
    joblib.dump(model, model_path)
    
    # Helper function to convert numpy values to Python native types
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        else:
            return obj
    
    # Save metrics
    metrics_path = f"{output_dir}/options_model_metrics_{timestamp}.json"
    with open(metrics_path, 'w') as f:
        # Convert all numpy values to Python native types
        serializable_metrics = convert_to_serializable(metrics)
        
        import json
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"Model saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")
    
    return model_path, metrics_path

def main():
    """Main function to train the enhanced trading model"""
    print("Enhanced Options Trading Model Trainer")
    print("=====================================")
    
    # Check if processed data exists
    if not os.path.exists(f"{PROCESSED_DIR}/option_train_data.csv") or not os.path.exists(f"{PROCESSED_DIR}/option_test_data.csv"):
        print("Error: Processed data files not found. Please run prepare_training_data.py first.")
        sys.exit(1)
    
    # Load data
    X_train, X_test, y_train, y_test = load_training_data()
    
    # Train model
    model_result = train_enhanced_model(X_train, y_train)
    
    # Evaluate model
    evaluation = evaluate_model(model_result['model'], X_test, y_test)
    
    # Plot results
    plot_results(model_result, evaluation, X_train, X_test, y_test)
    
    # Save model
    model_path, metrics_path = save_model(model_result['model'], {
        'evaluation': evaluation,
        'feature_importances': model_result['feature_importances'].to_dict(),
        'best_params': model_result['best_params'],
        'cv_scores': model_result['cv_scores']
    })
    
    print("\nModel Training Complete!")
    print(f"You can now use this model to improve your options trading strategy.")
    print(f"To use this model in your trading system, update your code to load the model from {model_path}")

if __name__ == "__main__":
    main() 