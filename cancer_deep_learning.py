# Cancer Classification Deep Learning Models
# This script implements deep learning models using TensorFlow/Keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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

def create_simple_dnn(input_dim):
    """
    Create a simple Deep Neural Network
    """
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def create_deep_dnn(input_dim):
    """
    Create a deeper Deep Neural Network
    """
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def create_regularized_dnn(input_dim):
    """
    Create a regularized Deep Neural Network with L1/L2 regularization
    """
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,),
                    kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu',
                    kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu',
                    kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def create_wide_dnn(input_dim):
    """
    Create a wide Deep Neural Network
    """
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test, epochs=100):
    """
    Train and evaluate a deep learning model
    """
    print(f"\nTraining {model_name}...")
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    # Predictions
    y_train_pred_proba = model.predict(X_train, verbose=0)
    y_test_pred_proba = model.predict(X_test, verbose=0)
    
    y_train_pred = (y_train_pred_proba > 0.5).astype(int).flatten()
    y_test_pred = (y_test_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    results = {
        'model_name': model_name,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'history': history,
        'y_test_pred': y_test_pred,
        'y_test_pred_proba': y_test_pred_proba.flatten()
    }
    
    print(f"{model_name} - Test Accuracy: {test_accuracy:.4f}, F1-Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
    
    return results

def cross_validate_model(model_creator, model_name, X_train, y_train, cv_folds=5):
    """
    Perform cross-validation for a deep learning model
    """
    print(f"\nPerforming cross-validation for {model_name}...")
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"Fold {fold + 1}/{cv_folds}")
        
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        # Create and train model
        model = model_creator(X_train.shape[1])
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        model.fit(
            X_train_fold, y_train_fold,
            epochs=100,
            batch_size=32,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate
        y_val_pred_proba = model.predict(X_val_fold, verbose=0)
        y_val_pred = (y_val_pred_proba > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(y_val_fold, y_val_pred)
        cv_scores.append(accuracy)
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    print(f"{model_name} CV Results - Mean: {cv_mean:.4f} ± {cv_std:.4f}")
    
    return cv_mean, cv_std

def visualize_training_history(results_list, save_path="/home/ubuntu/"):
    """
    Visualize training history for all models
    """
    print("\nCreating training history visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Deep Learning Models Training History', fontsize=16, fontweight='bold')
    
    # Training and Validation Loss
    ax1 = axes[0, 0]
    for result in results_list:
        history = result['history']
        ax1.plot(history.history['loss'], label=f"{result['model_name']} - Train")
        ax1.plot(history.history['val_loss'], label=f"{result['model_name']} - Val", linestyle='--')
    
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Training and Validation Accuracy
    ax2 = axes[0, 1]
    for result in results_list:
        history = result['history']
        ax2.plot(history.history['accuracy'], label=f"{result['model_name']} - Train")
        ax2.plot(history.history['val_accuracy'], label=f"{result['model_name']} - Val", linestyle='--')
    
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Model Performance Comparison
    ax3 = axes[1, 0]
    model_names = [result['model_name'] for result in results_list]
    test_accuracies = [result['test_accuracy'] for result in results_list]
    f1_scores = [result['f1_score'] for result in results_list]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, test_accuracies, width, label='Test Accuracy', alpha=0.8)
    bars2 = ax3.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax3.set_title('Model Performance Comparison')
    ax3.set_xlabel('Models')
    ax3.set_ylabel('Score')
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # ROC-AUC Comparison
    ax4 = axes[1, 1]
    roc_aucs = [result['roc_auc'] for result in results_list]
    bars4 = ax4.bar(model_names, roc_aucs, color='gold', alpha=0.8)
    ax4.set_title('ROC-AUC Comparison')
    ax4.set_xlabel('Models')
    ax4.set_ylabel('ROC-AUC')
    ax4.set_xticklabels(model_names, rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}deep_learning_results.png", dpi=300, bbox_inches='tight')
    plt.show()

def create_dl_results_dataframe(results_list, cv_results):
    """
    Create a DataFrame with deep learning results
    """
    print("\nCreating deep learning results summary...")
    
    data = []
    for i, result in enumerate(results_list):
        cv_mean, cv_std = cv_results[i]
        data.append({
            'Model': result['model_name'],
            'Train Accuracy': result['train_accuracy'],
            'Test Accuracy': result['test_accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1_score'],
            'ROC-AUC': result['roc_auc'],
            'CV Mean': cv_mean,
            'CV Std': cv_std
        })
    
    results_df = pd.DataFrame(data)
    results_df = results_df.round(4)
    
    print("Deep Learning Results Summary:")
    print(results_df.to_string(index=False))
    
    return results_df

def save_dl_results(results_df, save_path="/home/ubuntu/"):
    """
    Save deep learning results
    """
    print("\nSaving deep learning results...")
    
    # Save results DataFrame
    results_df.to_csv(f"{save_path}deep_learning_results.csv", index=False)
    
    print("Deep learning results saved successfully!")

def main():
    """
    Main function to execute the deep learning pipeline
    """
    print("=== Cancer Classification Deep Learning Pipeline ===")
    
    # Load preprocessed data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Define model creators
    model_creators = {
        'Simple DNN': create_simple_dnn,
        'Deep DNN': create_deep_dnn,
        'Regularized DNN': create_regularized_dnn,
        'Wide DNN': create_wide_dnn
    }
    
    # Train and evaluate models
    results_list = []
    cv_results = []
    
    for model_name, model_creator in model_creators.items():
        # Create and train model
        model = model_creator(X_train.shape[1])
        result = train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test)
        results_list.append(result)
        
        # Cross-validation
        cv_mean, cv_std = cross_validate_model(model_creator, model_name, X_train, y_train)
        cv_results.append((cv_mean, cv_std))
    
    # Create visualizations
    visualize_training_history(results_list)
    
    # Create results DataFrame
    results_df = create_dl_results_dataframe(results_list, cv_results)
    
    # Save results
    save_dl_results(results_df)
    
    print("\nDeep Learning pipeline completed successfully!")
    return results_df, results_list

if __name__ == "__main__":
    results_df, results_list = main()

