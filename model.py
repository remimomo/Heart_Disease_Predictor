import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import pickle
import lightgbm as lgb
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load dataset from a CSV file."""
    data = pd.read_csv('/Users/remimomo/Documents/heart_disease_project/data/heart-disease.csv')
    return data

def preprocess_data(data):
    """Preprocess the data: split into features and target, scale features."""
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8,  random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def build_models():
    """Define a list of models to evaluate."""
    models = [
        ('Logistic Regression', LogisticRegression()),
        ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
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

def save_model(model, file_path):
    """Save the trained model as a pickle file."""
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)

def load_model(file_path):
    """Load a saved model from a pickle file."""
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

if __name__ == "__main__":
    # Load data
    data = load_data('data/heart-disease.csv')
    
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

    # Train the best model on the entire training set
    best_model = dict(models)[best_model_name]
    best_model.fit(X_train, y_train)
    
    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f'\nTest Set Accuracy of {best_model_name}: {test_accuracy:.2f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    
    # Save the best model
    save_model(best_model, 'models/model.pkl')







