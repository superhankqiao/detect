import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score


def load_data(file_path):
    """
    Load Excel data. Assumes first column is Y and the rest are X.
    """
    data = pd.read_excel(file_path)
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    return X, y


def main():
    # File path
    file_path = 'data.xlsx'  # Replace with your Excel file path

    # Load data
    X, y = load_data(file_path)

    # Split data into training and testing sets (70:30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Define GBDT classifier
    gbdt = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
        # Other parameters are default; scikit-learn's GBDT includes anti-overfitting measures
    )

    # Ten-fold cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(gbdt, X_train, y_train, cv=skf, scoring='accuracy')
    print(f'10-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

    # Train model on the entire training set
    gbdt.fit(X_train, y_train)

    # Predict on the test set
    y_pred = gbdt.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Set Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()