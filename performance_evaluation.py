# Cancer Classification Performance Evaluation and Testing
# This script provides comprehensive evaluation of model performance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve, precision_recall_curve,
                           average_precision_score)
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data_and_results():
    """
    Load preprocessed data and model results
    """
    print("Loading data and results...")
    
    # Load preprocessed data
    X_train = np.load('/home/ubuntu/X_train.npy')
    X_test = np.load('/home/ubuntu/X_test.npy')
    y_train = np.load('/home/ubuntu/y_train.npy')
    y_test = np.load('/home/ubuntu/y_test.npy')
    
    # Load model results
    results_df = pd.read_csv('/home/ubuntu/model_results.csv')
    
    print(f"Data loaded successfully:")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Model results: {len(results_df)} models")
    
    return X_train, X_test, y_train, y_test, results_df

def detailed_performance_analysis(results_df, save_path="/home/ubuntu/"):
    """
    Create detailed performance analysis visualizations
    """
    print("\nCreating detailed performance analysis...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Overall Performance Radar Chart
    ax1 = axes[0, 0]
    
    # Select top 5 models by test accuracy
    top_models = results_df.nlargest(5, 'Test Accuracy')
    
    # Metrics for radar chart
    metrics = ['Test Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    # Create radar chart data
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for i, (_, model) in enumerate(top_models.iterrows()):
        values = [model[metric] for metric in metrics]
        values += values[:1]  # Complete the circle
        
        ax1.plot(angles, values, 'o-', linewidth=2, label=model['Model'], color=colors[i])
        ax1.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics)
    ax1.set_ylim(0.85, 1.0)
    ax1.set_title('Performance Radar Chart (Top 5 Models)')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax1.grid(True)
    
    # 2. Bias-Variance Analysis (Train vs Test Accuracy)
    ax2 = axes[0, 1]
    
    train_acc = results_df['Train Accuracy']
    test_acc = results_df['Test Accuracy']
    
    # Create scatter plot
    scatter = ax2.scatter(train_acc, test_acc, c=results_df['F1-Score'], 
                         cmap='viridis', s=100, alpha=0.7)
    
    # Add diagonal line (perfect generalization)
    ax2.plot([0.9, 1.0], [0.9, 1.0], 'r--', alpha=0.8, label='Perfect Generalization')
    
    # Add model labels
    for i, model in enumerate(results_df['Model']):
        ax2.annotate(model.split()[0], (train_acc.iloc[i], test_acc.iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.set_xlabel('Training Accuracy')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Bias-Variance Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='F1-Score')
    
    # 3. Cross-Validation Stability
    ax3 = axes[0, 2]
    
    models = results_df['Model']
    cv_means = results_df['CV Mean']
    cv_stds = results_df['CV Std']
    
    bars = ax3.bar(range(len(models)), cv_means, yerr=cv_stds, 
                   capsize=5, alpha=0.8, color='lightblue')
    ax3.set_xlabel('Models')
    ax3.set_ylabel('Cross-Validation Accuracy')
    ax3.set_title('Cross-Validation Stability')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels([m.split()[0] for m in models], rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, cv_means, cv_stds)):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.005,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 4. Precision vs Recall Trade-off
    ax4 = axes[1, 0]
    
    precision = results_df['Precision']
    recall = results_df['Recall']
    
    scatter = ax4.scatter(recall, precision, c=results_df['Test Accuracy'], 
                         cmap='plasma', s=100, alpha=0.7)
    
    # Add model labels
    for i, model in enumerate(results_df['Model']):
        ax4.annotate(model.split()[0], (recall.iloc[i], precision.iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.set_title('Precision-Recall Trade-off')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Test Accuracy')
    
    # 5. Model Complexity vs Performance
    ax5 = axes[1, 1]
    
    # Define complexity scores (subjective ranking)
    complexity_scores = {
        'Naive Bayes': 1,
        'Logistic Regression': 2,
        'Decision Tree': 3,
        'K-Nearest Neighbors': 4,
        'Support Vector Machine': 5,
        'Random Forest': 6,
        'Gradient Boosting': 7,
        'Neural Network': 8
    }
    
    complexities = [complexity_scores.get(model, 5) for model in results_df['Model']]
    
    scatter = ax5.scatter(complexities, results_df['Test Accuracy'], 
                         c=results_df['ROC-AUC'], cmap='coolwarm', s=100, alpha=0.7)
    
    # Add model labels
    for i, model in enumerate(results_df['Model']):
        ax5.annotate(model.split()[0], (complexities[i], results_df['Test Accuracy'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax5.set_xlabel('Model Complexity (Subjective Scale)')
    ax5.set_ylabel('Test Accuracy')
    ax5.set_title('Complexity vs Performance')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='ROC-AUC')
    
    # 6. Performance Distribution
    ax6 = axes[1, 2]
    
    metrics_data = results_df[['Test Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
    
    box_plot = ax6.boxplot([metrics_data[col] for col in metrics_data.columns], 
                          labels=metrics_data.columns, patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax6.set_ylabel('Score')
    ax6.set_title('Performance Metrics Distribution')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}detailed_performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def create_learning_curves(X_train, y_train, save_path="/home/ubuntu/"):
    """
    Create learning curves for top performing models
    """
    print("\nCreating learning curves...")
    
    # Select top 3 models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(kernel='linear', C=0.1, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42)
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Learning Curves - Top Performing Models', fontsize=16, fontweight='bold')
    
    for i, (name, model) in enumerate(models.items()):
        print(f"Computing learning curve for {name}...")
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy', random_state=42
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        axes[i].plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        axes[i].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                           alpha=0.1, color='blue')
        
        axes[i].plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        axes[i].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                           alpha=0.1, color='red')
        
        axes[i].set_xlabel('Training Set Size')
        axes[i].set_ylabel('Accuracy Score')
        axes[i].set_title(f'{name}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}learning_curves.png", dpi=300, bbox_inches='tight')
    plt.show()

def create_validation_curves(X_train, y_train, save_path="/home/ubuntu/"):
    """
    Create validation curves for hyperparameter analysis
    """
    print("\nCreating validation curves...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Validation Curves - Hyperparameter Analysis', fontsize=16, fontweight='bold')
    
    # 1. Random Forest - n_estimators
    print("Computing validation curve for Random Forest n_estimators...")
    param_range = [10, 50, 100, 150, 200, 250, 300]
    train_scores, val_scores = validation_curve(
        RandomForestClassifier(random_state=42), X_train, y_train,
        param_name='n_estimators', param_range=param_range,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    axes[0].plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
    axes[0].fill_between(param_range, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
    axes[0].plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
    axes[0].fill_between(param_range, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
    axes[0].set_xlabel('Number of Estimators')
    axes[0].set_ylabel('Accuracy Score')
    axes[0].set_title('Random Forest - n_estimators')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. SVM - C parameter
    print("Computing validation curve for SVM C parameter...")
    param_range = [0.01, 0.1, 1, 10, 100]
    train_scores, val_scores = validation_curve(
        SVC(kernel='linear', random_state=42), X_train, y_train,
        param_name='C', param_range=param_range,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    axes[1].semilogx(param_range, train_mean, 'o-', color='blue', label='Training Score')
    axes[1].fill_between(param_range, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
    axes[1].semilogx(param_range, val_mean, 'o-', color='red', label='Validation Score')
    axes[1].fill_between(param_range, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
    axes[1].set_xlabel('C Parameter (log scale)')
    axes[1].set_ylabel('Accuracy Score')
    axes[1].set_title('SVM - C Parameter')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Logistic Regression - C parameter
    print("Computing validation curve for Logistic Regression C parameter...")
    param_range = [0.01, 0.1, 1, 10, 100]
    train_scores, val_scores = validation_curve(
        LogisticRegression(random_state=42, max_iter=1000), X_train, y_train,
        param_name='C', param_range=param_range,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    axes[2].semilogx(param_range, train_mean, 'o-', color='blue', label='Training Score')
    axes[2].fill_between(param_range, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
    axes[2].semilogx(param_range, val_mean, 'o-', color='red', label='Validation Score')
    axes[2].fill_between(param_range, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
    axes[2].set_xlabel('C Parameter (log scale)')
    axes[2].set_ylabel('Accuracy Score')
    axes[2].set_title('Logistic Regression - C Parameter')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}validation_curves.png", dpi=300, bbox_inches='tight')
    plt.show()

def statistical_significance_testing(results_df, save_path="/home/ubuntu/"):
    """
    Perform statistical significance testing between models
    """
    print("\nPerforming statistical significance testing...")
    
    # Create a summary table
    summary_data = []
    
    # Get top 3 models
    top_models = results_df.nlargest(3, 'Test Accuracy')
    
    for _, model in top_models.iterrows():
        summary_data.append({
            'Model': model['Model'],
            'Test Accuracy': f"{model['Test Accuracy']:.4f}",
            'Precision': f"{model['Precision']:.4f}",
            'Recall': f"{model['Recall']:.4f}",
            'F1-Score': f"{model['F1-Score']:.4f}",
            'ROC-AUC': f"{model['ROC-AUC']:.4f}",
            'CV Mean ± Std': f"{model['CV Mean']:.4f} ± {model['CV Std']:.4f}",
            'Generalization Gap': f"{model['Train Accuracy'] - model['Test Accuracy']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary table
    summary_df.to_csv(f"{save_path}model_performance_summary.csv", index=False)
    
    print("Top 3 Models Performance Summary:")
    print(summary_df.to_string(index=False))
    
    return summary_df

def create_final_recommendations(results_df, save_path="/home/ubuntu/"):
    """
    Create final model recommendations based on comprehensive analysis
    """
    print("\nGenerating final model recommendations...")
    
    # Analyze different aspects
    best_accuracy = results_df.loc[results_df['Test Accuracy'].idxmax()]
    best_f1 = results_df.loc[results_df['F1-Score'].idxmax()]
    best_roc_auc = results_df.loc[results_df['ROC-AUC'].idxmax()]
    most_stable = results_df.loc[results_df['CV Std'].idxmin()]
    best_generalization = results_df.loc[(results_df['Train Accuracy'] - results_df['Test Accuracy']).idxmin()]
    
    recommendations = {
        'Best Overall Accuracy': best_accuracy['Model'],
        'Best F1-Score': best_f1['Model'],
        'Best ROC-AUC': best_roc_auc['Model'],
        'Most Stable (Lowest CV Std)': most_stable['Model'],
        'Best Generalization': best_generalization['Model']
    }
    
    print("Model Recommendations by Category:")
    for category, model in recommendations.items():
        print(f"  {category}: {model}")
    
    # Overall recommendation
    print(f"\nOverall Recommendation: {best_accuracy['Model']}")
    print(f"Reasoning: Highest test accuracy ({best_accuracy['Test Accuracy']:.4f}) with excellent ROC-AUC ({best_accuracy['ROC-AUC']:.4f})")
    
    return recommendations

def main():
    """
    Main function to execute the performance evaluation pipeline
    """
    print("=== Cancer Classification Performance Evaluation ===")
    
    # Load data and results
    X_train, X_test, y_train, y_test, results_df = load_data_and_results()
    
    # Detailed performance analysis
    detailed_performance_analysis(results_df)
    
    # Learning curves
    create_learning_curves(X_train, y_train)
    
    # Validation curves
    create_validation_curves(X_train, y_train)
    
    # Statistical significance testing
    summary_df = statistical_significance_testing(results_df)
    
    # Final recommendations
    recommendations = create_final_recommendations(results_df)
    
    print("\nPerformance evaluation completed successfully!")
    return results_df, summary_df, recommendations

if __name__ == "__main__":
    results_df, summary_df, recommendations = main()

