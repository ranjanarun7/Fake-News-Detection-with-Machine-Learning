# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Step 2: Load true.csv and fake.csv
true_news = pd.read_csv('true.csv')  # Ensure the path is correct
fake_news = pd.read_csv('fake.csv')  # Ensure the path is correct

# Step 3: Add Labels
# Add a 'label' column: 1 for True (Real) news, 0 for Fake news
true_news['label'] = 1
fake_news['label'] = 0

# Step 4: Combine the DataFrames
# Combine true and fake datasets
combined_data = pd.concat([true_news, fake_news], ignore_index=True)

# Step 5: Shuffle the data to mix the true and fake news
combined_data = combined_data.sample(frac=1).reset_index(drop=True)

# Optional: Display the first few rows of the combined dataset
print("Combined Data (first few rows):")
print(combined_data.head())

# Step 6: Check for missing values
print("\nChecking for missing values:")
print(combined_data.isnull().sum())  # If there are missing values, handle them as needed

# Step 7: Data Preprocessing
# Assuming the news text is in a column named 'text'
X = combined_data['text']  # Features (news articles)
y = combined_data['label']  # Target (labels: 0 for Fake, 1 for Real)

# Step 8: Split the Data into Training and Testing Sets
# 80% of the data for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Text Vectorization using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)  # Removing English stopwords
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)  # Fit and transform training data
X_test_tfidf = tfidf_vectorizer.transform(X_test)  # Transform testing data (without fitting again)

# Step 10: Build and Train the Model (Logistic Regression)
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)  # Train the model

# Step 11: Make Predictions on the Test Data
y_pred = model.predict(X_test_tfidf)

# Step 12: Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'\nModel Accuracy: {accuracy * 100:.2f}%')

# Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Display classification report (precision, recall, F1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

# Step 13: Save the Model and Vectorizer (Optional)
# Save the trained Logistic Regression model
joblib.dump(model, 'fake_news_model.pkl')

# Save the TF-IDF vectorizer
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

print("\nModel and vectorizer saved as 'fake_news_model.pkl' and 'tfidf_vectorizer.pkl'.")
