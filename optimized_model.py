import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load dataset from a CSV file."""
    data = pd.read_csv('/Users/remimomo/Documents/heart_disease_project/data/heart-disease.csv')
    return data

def preprocess_data(data):
    """Preprocess the data: split into features and target, scale features."""
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def build_models():
    """Define a list of models to evaluate."""
    models = [
        ('Logistic Regression', LogisticRegression(max_iter=1000)),
        ('Random Forest', RandomForestClassifier(random_state=42)),
        ('Support Vector Machine', SVC(kernel='linear', random_state=42)),
        ('K-Nearest Neighbors', KNeighborsClassifier()),
        ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
        ('AdaBoost', AdaBoostClassifier(random_state=42)),
        ('LightGBM', lgb.LGBMClassifier(random_state=42))
    ]
    return models

def evaluate_models(models, X_train, y_train):
    """Evaluate multiple models using cross-validation and return a DataFrame of results."""
    results = []

    for name, model in models:
        pipeline = Pipeline([('model', model)])
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        results.append({
            'Model': name,
            'Mean Accuracy': np.mean(cv_scores),
            'Std Accuracy': np.std(cv_scores)
        })
    
    return pd.DataFrame(results)

def plot_model_performance(results_df):
    """Plot the mean accuracy scores of the models with error bars for standard deviation."""
    plt.figure(figsize=(12, 8))
    
    # Plot mean accuracy with error bars for standard deviation
    plt.barh(results_df['Model'], results_df['Mean Accuracy'], xerr=results_df['Std Accuracy'], color='skyblue')
    
    plt.xlabel('Accuracy Score')
    plt.ylabel('Model')
    plt.title('Model Comparison: Mean Accuracy with Standard Deviation')
    plt.xlim(0, 1)
    plt.gca().invert_yaxis()  # Invert y-axis for better readability
    
    plt.show()

def hyperparameter_tuning(X_train, y_train, model, param_grid):
    """Perform hyperparameter tuning using GridSearchCV."""
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

def evaluate_optimized_model(model, X_train, y_train, X_test, y_test):
    """Evaluate the optimized model using cross-validation and test set."""
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Optimized Model Cross-Validation Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Set Accuracy: {test_accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

def plot_feature_importance(model, feature_names):
    """Plot the feature importance for a given model."""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]

        # Create a plot
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importance")
        plt.bar(range(len(importance)), importance[indices], align="center")
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=90)
        plt.xlabel("Features")
        plt.ylabel("Importance Score")
        plt.tight_layout()
        plt.show()
    else:
        print("The model does not have feature_importances_ attribute.")

def save_model(model, file_path):
    """Save the trained model as a pickle file."""
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    # Load data
    data = load_data('/Users/remimomo/Documents/heart_disease_project/data/heart-disease.csv')
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Build models
    models = build_models()
    
    # Evaluate models
    results_df = evaluate_models(models, X_train, y_train)
    
    # Display the results
    print("Model Evaluation Results:")
    print(results_df.sort_values(by='Mean Accuracy', ascending=False))

    # Plot model performance
    plot_model_performance(results_df)
    
    # Selecting the best model based on cross-validation score
    best_model_name = results_df.loc[results_df['Mean Accuracy'].idxmax(), 'Model']
    print(f"\nBest Model: {best_model_name}")

    # Assuming the best model is LightGBM (as an example)
    if best_model_name == 'LightGBM':
        param_grid = {
            'num_leaves': [31, 50, 100],
            'learning_rate': [0.01, 0.1, 0.05],
            'n_estimators': [100, 200, 300],
            'max_depth': [-1, 10, 20, 30]
        }
        optimized_model = hyperparameter_tuning(X_train, y_train, lgb.LGBMClassifier(random_state=42), param_grid)
    # Save the optimized model
    save_model(optimized_model, 'models/optimized_model.pkl')
    # Evaluate optimized model
    evaluate_optimized_model(optimized_model, X_train, y_train, X_test, y_test)
    
    # Save the optimized model
    save_model(optimized_model, 'models/optimized_model.pkl')
    
    # Feature importance
    feature_names = data.columns[:-1]  # Assuming the last column is the target
    plot_feature_importance(optimized_model, feature_names)
