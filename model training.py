import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Load cleaned dataset
df = pd.read_csv("preprocessed_spam.csv")

# Check if necessary columns exist
if "content" not in df.columns or "category" not in df.columns:
    raise ValueError("Dataset doesnt contain the necessary columns")


# Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove punctuation and numbers
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)

# Apply preprocessing to messages
df["content"] = df["content"].apply(preprocess_text)

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df["content"])  
y = df["category"]

# Split data into training and test set (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Train Naïve Bayes Model
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train, y_train)

# Train Decision Tree Model
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)

# Save Models & Vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("logistic.pkl", "wb") as f:
    pickle.dump(logistic_model, f)

with open("naive_bayes.pkl", "wb") as f:
    pickle.dump(naive_bayes_model, f)

with open("decision_tree.pkl", "wb") as f:
    pickle.dump(decision_tree_model, f)

print("Models trained and saved successfully!")

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"--- {model_name} Evaluation ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, pos_label=1):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, pos_label=1):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred, pos_label=1):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    print("-" * 40 + "\n")

# Evaluate all models
evaluate_model(logistic_model, X_test, y_test, "Logistic Regression")
evaluate_model(naive_bayes_model, X_test, y_test, "Naive Bayes")
evaluate_model(decision_tree_model, X_test, y_test, "Decision Tree")