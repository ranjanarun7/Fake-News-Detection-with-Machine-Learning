import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

true_news = pd.read_csv('true.csv')  
fake_news = pd.read_csv('fake.csv') 

true_news['label'] = 1
fake_news['label'] = 0

combined_data = pd.concat([true_news, fake_news], ignore_index=True)

combined_data = combined_data.sample(frac=1).reset_index(drop=True)

print("Combined Data (first few rows):")
print(combined_data.head())

print("\nChecking for missing values:")
print(combined_data.isnull().sum())

X = combined_data['text'] 
y = combined_data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)  
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f'\nModel Accuracy: {accuracy * 100:.2f}%')

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

joblib.dump(model, 'fake_news_model.pkl')

joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

print("\nModel and vectorizer saved as 'fake_news_model.pkl' and 'tfidf_vectorizer.pkl'.")
