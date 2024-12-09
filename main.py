import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def compute_feature_importance(X_train, y_train):
    """
    Compute t-statistics for each feature to measure importance
    """
    n_features = X_train.shape[1]
    t_stats = np.zeros(n_features)
    # Calculate t-statistic for each feature
    for j in range(n_features):
        pos_samples = X_train[y_train == 1, j]
        neg_samples = X_train[y_train == 0, j]
        t_stat, _ = stats.ttest_ind(pos_samples, neg_samples, equal_var=False)
        t_stats[j] = abs(t_stat) if not np.isnan(t_stat) else 0
    return t_stats

# Load training data
print("Loading training data...")
train_df = pd.read_csv('train.csv')

# Extract features and target
X_train = train_df.iloc[:, 3:].values  # Skip id, sentiment, and review columns
y_train = train_df['sentiment'].values

# Compute feature importance using t-statistics
print("Computing feature importance...")
t_stats = compute_feature_importance(X_train, y_train)

# Select top features
print("Selecting top features...")
top_n = 1400
selected_features = np.argsort(t_stats)[-top_n:]
X_train_selected = X_train[:, selected_features]

# Train logistic regression model with elastic net
print("Training model...")
model = LogisticRegression(
    max_iter=1000,
    C=5,  # Adjust regularization strength
    penalty='elasticnet',
    solver='saga',  # Required for elastic net
    l1_ratio=0.5    # Mix of L1 and L2 (0.5 means equal mix)
)
model.fit(X_train_selected, y_train)

# Load and process test data
print("Processing test data...")
test_df = pd.read_csv('test.csv')
X_test = test_df.iloc[:, 2:].values  # Skip id and review columns
X_test_selected = X_test[:, selected_features]

# Generate predictions
print("Generating predictions...")
test_probs = model.predict_proba(X_test_selected)[:, 1]

# Create and save submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'prob': test_probs
})
submission.to_csv('mysubmission.csv', index=False)

# Calculate and print AUC
print("Calculating AUC...")
true_labels_df = pd.read_csv('test_y.csv')
true_sentiment = true_labels_df['sentiment']
predicted_probabilities = test_probs
auc_score = roc_auc_score(true_sentiment, predicted_probabilities)
print(f"AUC Score: {auc_score:.5f}")
