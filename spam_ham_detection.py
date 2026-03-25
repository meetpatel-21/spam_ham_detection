import matplotlib.pyplot as plt
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, 
    classification_report, roc_curve, auc,
    precision_recall_fscore_support
)
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SpamDetector:
    """Professional spam detection pipeline with comprehensive evaluation."""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stopwords = set(stopwords.words('english'))
        self.vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1
        )
        self.is_fitted = False
    
    def preprocess_text(self, text):
        """Clean and preprocess text data."""
        text = text.lower()
        text = text.replace('\\r\\n', ' ')
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        stemmed_words = [self.stemmer.stem(word) for word in words 
                        if word not in self.stopwords]
        return ' '.join(stemmed_words)
    
    def load_and_prepare_data(self, filepath):
        """Load dataset and prepare features/labels."""
        print("Loading dataset...")
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}")
        print(f"Class distribution:\n{df['label_num'].value_counts()}")
        
        # Preprocess text
        print("Preprocessing text...")
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Vectorize
        X = self.vectorizer.fit_transform(df['processed_text']).toarray()
        y = df['label_num'].values
        
        print(f"Feature matrix shape: {X.shape}")
        return X, y, df
    
    def train(self, X, y):
        """Train the spam detection model."""
        print("Training Random Forest classifier...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_comprehensive(self, X_test, y_test):
        """Generate comprehensive evaluation metrics and visualizations."""
        if not self.is_fitted:
            raise ValueError("Model must be trained first!")
        
        y_pred = self.model.predict(X_test)
        y_scores = self.model.predict_proba(X_test)[:, 1]
        
        # 1. Classification Report
        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
        
        # 2. Confusion Matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        # 3. Precision-Recall Bar Chart
        self.plot_precision_recall(y_test, y_pred)
        
        # 4. ROC Curve
        self.plot_roc_curve(y_test, y_scores)
        
        return {
            'predictions': y_pred,
            'probabilities': y_scores,
            'test_accuracy': self.model.score(X_test, y_test)
        }
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['ham', 'spam'], 
                   yticklabels=['ham', 'spam'])
        plt.title('Confusion Matrix: Spam vs Ham', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall(self, y_true, y_pred):
        """Plot precision and recall per class."""
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0,1])
        
        x = np.arange(2)
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width/2, prec, width, label='Precision', alpha=0.8)
        ax.bar(x + width/2, rec, width, label='Recall', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(['ham', 'spam'])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score')
        ax.set_title('Precision and Recall per Class', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, y_true, y_scores):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Spam Detection', fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def predict_single_email(self, email_text):
        """Predict spam/ham for a single email."""
        if not self.is_fitted:
            raise ValueError("Model must be trained first!")
        
        processed_email = self.preprocess_text(email_text)
        email_vector = self.vectorizer.transform([processed_email]).toarray()
        prediction = self.model.predict(email_vector)[0]
        probability = self.model.predict_proba(email_vector)[0]
        
        label = 'spam' if prediction == 1 else 'ham'
        confidence = max(probability)
        
        return {
            'prediction': label,
            'confidence': confidence,
            'probabilities': {'ham': probability[0], 'spam': probability[1]}
        }

# Main execution
def main():
    """Main function to run the spam detection pipeline."""
    # Initialize detector
    detector = SpamDetector()
    
    # Load and prepare data
    X, y, df = detector.load_and_prepare_data('spam_ham_dataset.csv')
    
    # Train model
    X_train, X_test, y_train, y_test = detector.train(X, y)
    
    # Comprehensive evaluation
    results = detector.evaluate_comprehensive(X_test, y_test)
    
    # Example prediction
    print("\n" + "="*50)
    print("EXAMPLE PREDICTION")
    print("="*50)
    sample_email = df['text'].iloc[10]
    print(f"Sample email preview: {sample_email[:100]}...")
    prediction = detector.predict_single_email(sample_email)
    print(f"Prediction: {prediction['prediction']} (confidence: {prediction['confidence']:.3f})")
    print(f"Actual label: {'spam' if df['label_num'].iloc[10] == 1 else 'ham'}")
    
    return detector

if __name__ == "__main__":
    detector = main()
