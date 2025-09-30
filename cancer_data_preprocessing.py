# Cancer Classification Data Collection and Preprocessing
# This script demonstrates data collection and preprocessing for cancer classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_cancer_dataset():
    """
    Load the breast cancer Wisconsin dataset from sklearn
    """
    print("Loading Breast Cancer Wisconsin Dataset...")
    data = load_breast_cancer()
    
    # Create DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df['target_names'] = df['target'].map({0: 'malignant', 1: 'benign'})
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {len(data.feature_names)}")
    print(f"Classes: {data.target_names}")
    
    return df, data

def explore_dataset(df):
    """
    Perform exploratory data analysis
    """
    print("\n=== Dataset Exploration ===")
    print(f"Dataset Info:")
    print(f"- Total samples: {len(df)}")
    print(f"- Features: {df.shape[1] - 2}")  # Excluding target and target_names
    print(f"- Missing values: {df.isnull().sum().sum()}")
    
    # Class distribution
    print(f"\nClass Distribution:")
    class_counts = df['target_names'].value_counts()
    print(class_counts)
    
    # Statistical summary
    print(f"\nStatistical Summary:")
    print(df.describe())
    
    return class_counts

def visualize_data(df, save_path="/home/ubuntu/"):
    """
    Create visualizations for the dataset
    """
    print("\n=== Creating Visualizations ===")
    
    # 1. Class distribution
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    df['target_names'].value_counts().plot(kind='bar', color=['#FF6B6B', '#4ECDC4'])
    plt.title('Cancer Type Distribution')
    plt.xlabel('Cancer Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # 2. Feature correlation heatmap (top 10 features)
    plt.subplot(2, 2, 2)
    top_features = df.columns[:10]  # First 10 features
    corr_matrix = df[top_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix (Top 10)')
    
    # 3. Feature distribution for key features
    plt.subplot(2, 2, 3)
    key_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']
    for feature in key_features:
        plt.hist(df[feature], alpha=0.7, label=feature, bins=20)
    plt.title('Distribution of Key Features')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 4. Box plot for target comparison
    plt.subplot(2, 2, 4)
    df.boxplot(column='mean radius', by='target_names', ax=plt.gca())
    plt.title('Mean Radius by Cancer Type')
    plt.suptitle('')  # Remove default title
    
    plt.tight_layout()
    plt.savefig(f"{save_path}cancer_data_exploration.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional detailed visualization
    plt.figure(figsize=(15, 10))
    
    # Feature importance visualization
    plt.subplot(2, 3, 1)
    feature_means = df.groupby('target_names').mean()
    top_diff_features = (feature_means.loc['malignant'] - feature_means.loc['benign']).abs().nlargest(10)
    top_diff_features.plot(kind='bar')
    plt.title('Top 10 Discriminative Features')
    plt.xlabel('Features')
    plt.ylabel('Absolute Difference')
    plt.xticks(rotation=45)
    
    # Scatter plot of two important features
    plt.subplot(2, 3, 2)
    colors = {'malignant': '#FF6B6B', 'benign': '#4ECDC4'}
    for target in df['target_names'].unique():
        subset = df[df['target_names'] == target]
        plt.scatter(subset['mean radius'], subset['mean texture'], 
                   c=colors[target], label=target, alpha=0.7)
    plt.xlabel('Mean Radius')
    plt.ylabel('Mean Texture')
    plt.title('Mean Radius vs Mean Texture')
    plt.legend()
    
    # Distribution comparison
    plt.subplot(2, 3, 3)
    for target in df['target_names'].unique():
        subset = df[df['target_names'] == target]
        plt.hist(subset['mean area'], alpha=0.7, label=target, bins=20, color=colors[target])
    plt.xlabel('Mean Area')
    plt.ylabel('Frequency')
    plt.title('Mean Area Distribution by Cancer Type')
    plt.legend()
    
    # Feature scaling comparison
    plt.subplot(2, 3, 4)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['mean radius', 'mean texture', 'mean area']])
    scaled_df = pd.DataFrame(scaled_features, columns=['mean radius', 'mean texture', 'mean area'])
    scaled_df.boxplot()
    plt.title('Scaled Features Distribution')
    plt.ylabel('Scaled Values')
    
    # Correlation with target
    plt.subplot(2, 3, 5)
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()['target'].drop('target').abs().sort_values(ascending=False)[:10]
    correlations.plot(kind='bar')
    plt.title('Top 10 Features Correlated with Target')
    plt.xlabel('Features')
    plt.ylabel('Absolute Correlation')
    plt.xticks(rotation=45)
    
    # Feature variance
    plt.subplot(2, 3, 6)
    feature_variance = df.drop(['target', 'target_names'], axis=1).var().sort_values(ascending=False)[:10]
    feature_variance.plot(kind='bar')
    plt.title('Top 10 Features by Variance')
    plt.xlabel('Features')
    plt.ylabel('Variance')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}cancer_detailed_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def preprocess_data(df):
    """
    Preprocess the data for machine learning
    """
    print("\n=== Data Preprocessing ===")
    
    # Separate features and target
    X = df.drop(['target', 'target_names'], axis=1)
    y = df['target']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Check for missing values
    print(f"Missing values in features: {X.isnull().sum().sum()}")
    print(f"Missing values in target: {y.isnull().sum()}")
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    print("Data scaling completed.")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Training set class distribution: {np.bincount(y_train)}")
    print(f"Test set class distribution: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test, scaler, X_scaled_df

def save_preprocessing_results(X_train, X_test, y_train, y_test, scaler, save_path="/home/ubuntu/"):
    """
    Save preprocessing results for later use
    """
    print("\n=== Saving Preprocessing Results ===")
    
    # Save datasets
    np.save(f"{save_path}X_train.npy", X_train)
    np.save(f"{save_path}X_test.npy", X_test)
    np.save(f"{save_path}y_train.npy", y_train)
    np.save(f"{save_path}y_test.npy", y_test)
    
    print("Preprocessed datasets saved successfully.")
    
    # Save preprocessing summary
    summary = {
        'train_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'features': X_train.shape[1],
        'classes': len(np.unique(y_train)),
        'train_class_0': np.sum(y_train == 0),
        'train_class_1': np.sum(y_train == 1),
        'test_class_0': np.sum(y_test == 0),
        'test_class_1': np.sum(y_test == 1)
    }
    
    return summary

def main():
    """
    Main function to execute the data preprocessing pipeline
    """
    print("=== Cancer Classification Data Preprocessing Pipeline ===")
    
    # Load dataset
    df, data = load_cancer_dataset()
    
    # Explore dataset
    class_counts = explore_dataset(df)
    
    # Visualize data
    visualize_data(df)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler, X_scaled_df = preprocess_data(df)
    
    # Save results
    summary = save_preprocessing_results(X_train, X_test, y_train, y_test, scaler)
    
    print("\n=== Preprocessing Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\nData preprocessing completed successfully!")
    return df, X_train, X_test, y_train, y_test, scaler, summary

if __name__ == "__main__":
    df, X_train, X_test, y_train, y_test, scaler, summary = main()

