# Cancer Classification Machine Learning and Deep Learning Models
# This script implements various ML and DL models for cancer classification

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve)
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_preprocessed_data():
    """
    Load the preprocessed data from the previous step
    """
    print("Loading preprocessed data...")
    X_train = np.load('/home/ubuntu/X_train.npy')
    X_test = np.load('/home/ubuntu/X_test.npy')
    y_train = np.load('/home/ubuntu/y_train.npy')
    y_test = np.load('/home/ubuntu/y_test.npy')
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def initialize_models():
    """
    Initialize various machine learning models
    """
    print("\nInitializing machine learning models...")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine': SVC(kernel='rbf', random_state=42, probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
    }
    
    print(f"Initialized {len(models)} models:")
    for name in models.keys():
        print(f"  - {name}")
    
    return models

def train_models(models, X_train, y_train):
    """
    Train all models
    """
    print("\nTraining models...")
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    print("All models trained successfully!")
    return trained_models

def evaluate_models(trained_models, X_train, X_test, y_train, y_test):
    """
    Evaluate all trained models
    """
    print("\nEvaluating models...")
    
    results = {}
    
    for name, model in trained_models.items():
        print(f"Evaluating {name}...")
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        
        # ROC-AUC (if probability predictions available)
        roc_auc = roc_auc_score(y_test, y_test_proba) if y_test_proba is not None else None
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results[name] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'y_test_pred': y_test_pred,
            'y_test_proba': y_test_proba
        }
    
    return results

def create_results_dataframe(results):
    """
    Create a DataFrame with all results for easy comparison
    """
    print("\nCreating results summary...")
    
    data = []
    for model_name, metrics in results.items():
        data.append({
            'Model': model_name,
            'Train Accuracy': metrics['train_accuracy'],
            'Test Accuracy': metrics['test_accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc'],
            'CV Mean': metrics['cv_mean'],
            'CV Std': metrics['cv_std']
        })
    
    results_df = pd.DataFrame(data)
    results_df = results_df.round(4)
    
    print("Results Summary:")
    print(results_df.to_string(index=False))
    
    return results_df

def visualize_model_comparison(results_df, save_path="/home/ubuntu/"):
    """
    Create comprehensive visualizations comparing model performance
    """
    print("\nCreating model comparison visualizations...")
    
    # Set up the figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Cancer Classification Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. Test Accuracy Comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(results_df['Model'], results_df['Test Accuracy'], color='skyblue', alpha=0.8)
    ax1.set_title('Test Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0.8, 1.0)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Precision vs Recall
    ax2 = axes[0, 1]
    scatter = ax2.scatter(results_df['Recall'], results_df['Precision'], 
                         c=results_df['F1-Score'], cmap='viridis', s=100, alpha=0.8)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision vs Recall (colored by F1-Score)')
    
    # Add model labels
    for i, model in enumerate(results_df['Model']):
        ax2.annotate(model.split()[0], (results_df['Recall'].iloc[i], results_df['Precision'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.colorbar(scatter, ax=ax2, label='F1-Score')
    
    # 3. F1-Score Comparison
    ax3 = axes[0, 2]
    bars3 = ax3.bar(results_df['Model'], results_df['F1-Score'], color='lightcoral', alpha=0.8)
    ax3.set_title('F1-Score Comparison')
    ax3.set_ylabel('F1-Score')
    ax3.set_ylim(0.8, 1.0)
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 4. Cross-Validation Scores with Error Bars
    ax4 = axes[1, 0]
    bars4 = ax4.bar(results_df['Model'], results_df['CV Mean'], 
                   yerr=results_df['CV Std'], capsize=5, color='lightgreen', alpha=0.8)
    ax4.set_title('Cross-Validation Accuracy (with std)')
    ax4.set_ylabel('CV Accuracy')
    ax4.set_ylim(0.8, 1.0)
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. ROC-AUC Comparison (excluding models without probability predictions)
    ax5 = axes[1, 1]
    roc_data = results_df[results_df['ROC-AUC'].notna()]
    bars5 = ax5.bar(roc_data['Model'], roc_data['ROC-AUC'], color='gold', alpha=0.8)
    ax5.set_title('ROC-AUC Comparison')
    ax5.set_ylabel('ROC-AUC')
    ax5.set_ylim(0.8, 1.0)
    ax5.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars5:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 6. Overall Performance Radar Chart (using top 5 models)
    ax6 = axes[1, 2]
    top_models = results_df.nlargest(5, 'Test Accuracy')
    
    # Create a simple performance score combining multiple metrics
    performance_scores = []
    for _, row in top_models.iterrows():
        score = (row['Test Accuracy'] + row['Precision'] + row['Recall'] + row['F1-Score']) / 4
        performance_scores.append(score)
    
    bars6 = ax6.bar(top_models['Model'], performance_scores, color='mediumpurple', alpha=0.8)
    ax6.set_title('Overall Performance Score (Top 5)')
    ax6.set_ylabel('Average Score')
    ax6.set_ylim(0.8, 1.0)
    ax6.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars6:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}model_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def create_confusion_matrices(results, y_test, save_path="/home/ubuntu/"):
    """
    Create confusion matrices for top performing models
    """
    print("\nCreating confusion matrices...")
    
    # Select top 4 models based on test accuracy
    top_models = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Confusion Matrices - Top 4 Models', fontsize=16, fontweight='bold')
    
    axes = axes.ravel()
    
    for i, (model_name, metrics) in enumerate(top_models):
        cm = confusion_matrix(y_test, metrics['y_test_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=['Malignant', 'Benign'],
                   yticklabels=['Malignant', 'Benign'])
        axes[i].set_title(f'{model_name}\nAccuracy: {metrics["test_accuracy"]:.3f}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.show()

def create_roc_curves(results, y_test, save_path="/home/ubuntu/"):
    """
    Create ROC curves for models with probability predictions
    """
    print("\nCreating ROC curves...")
    
    plt.figure(figsize=(10, 8))
    
    for model_name, metrics in results.items():
        if metrics['y_test_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, metrics['y_test_proba'])
            auc = metrics['roc_auc']
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Cancer Classification Models')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}roc_curves.png", dpi=300, bbox_inches='tight')
    plt.show()

def hyperparameter_tuning(X_train, y_train, save_path="/home/ubuntu/"):
    """
    Perform hyperparameter tuning for the best performing models
    """
    print("\nPerforming hyperparameter tuning...")
    
    # Define parameter grids for top models
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1, 1],
            'kernel': ['rbf', 'linear']
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    }
    
    tuned_results = {}
    
    for model_name, param_grid in param_grids.items():
        print(f"Tuning {model_name}...")
        
        if model_name == 'Random Forest':
            model = RandomForestClassifier(random_state=42)
        elif model_name == 'SVM':
            model = SVC(random_state=42, probability=True)
        elif model_name == 'Gradient Boosting':
            model = GradientBoostingClassifier(random_state=42)
        
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        tuned_results[model_name] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }
        
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return tuned_results

def save_model_results(results_df, tuned_results, save_path="/home/ubuntu/"):
    """
    Save all results to files
    """
    print("\nSaving results...")
    
    # Save results DataFrame
    results_df.to_csv(f"{save_path}model_results.csv", index=False)
    
    # Save tuned results
    with open(f"{save_path}tuned_results.txt", 'w') as f:
        f.write("Hyperparameter Tuning Results\n")
        f.write("=" * 40 + "\n\n")
        
        for model_name, results in tuned_results.items():
            f.write(f"{model_name}:\n")
            f.write(f"Best Parameters: {results['best_params']}\n")
            f.write(f"Best CV Score: {results['best_score']:.4f}\n\n")
    
    print("Results saved successfully!")

def main():
    """
    Main function to execute the machine learning pipeline
    """
    print("=== Cancer Classification Machine Learning Pipeline ===")
    
    # Load preprocessed data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Initialize models
    models = initialize_models()
    
    # Train models
    trained_models = train_models(models, X_train, y_train)
    
    # Evaluate models
    results = evaluate_models(trained_models, X_train, X_test, y_train, y_test)
    
    # Create results DataFrame
    results_df = create_results_dataframe(results)
    
    # Create visualizations
    visualize_model_comparison(results_df)
    create_confusion_matrices(results, y_test)
    create_roc_curves(results, y_test)
    
    # Hyperparameter tuning
    tuned_results = hyperparameter_tuning(X_train, y_train)
    
    # Save results
    save_model_results(results_df, tuned_results)
    
    print("\nMachine Learning pipeline completed successfully!")
    return results_df, tuned_results

if __name__ == "__main__":
    results_df, tuned_results = main()

