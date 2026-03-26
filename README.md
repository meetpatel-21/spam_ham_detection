# Spam_Ham_Detection
## ✨ Overview
A production-ready spam detection pipeline that processes SMS/email text data using advanced NLP preprocessing (stemming, stopword removal, n-grams) and trains a Random Forest classifier. Features comprehensive evaluation with confusion matrices, ROC curves, precision-recall charts, and single-email prediction capabilities.

## Key Capabilities:
- 99%+ accuracy on spam/ham classification
- Professional visualizations for model evaluation
- Real-time single email prediction
- Scalable feature extraction (5000 features + bigrams)

## 🚀 Features
- Text Preprocessing: Lowercasing, punctuation removal, stemming, stopword filtering
- Advanced Vectorization: Bag-of-words with unigrams + bigrams (5000 features)
- Robust ML Model: Random Forest with hyper-optimized parameters
- Comprehensive Evaluation:
- Classification report (precision, recall, F1-score)
- Confusion matrix heatmap
- Precision-Recall bar charts
- ROC curve with AUC score
- Production Ready: Single email prediction with confidence scores
- Interactive CLI: Command-line interface for testin

## 🛠️ Technologies & Tools
- Programming:	Python 3.8+
- Data Processing:	pandas, numpy
- NLP	NLTK (PorterStemmer, stopwords)
- Machine Learning:	scikit-learn (RandomForest, metrics)
- Feature Extraction:	CountVectorizer (ngrams)
- Visualization:	matplotlib, seaborn
- Development:	warnings filter

## 📋 Standard CLI Input/Output
### Basic Usage
##### python spam_detector.py --data spam_ham_dataset.csv

### Sample output
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/7403da25-b176-4c6e-a5c8-ddb22831526f" />

## 🧪 Testing Instructions
### 1. Prerequisites
#### i) Clone repository
- ##### git clone <your-repo-url>
- ##### cd spam-detector

#### ii) Create virtual environment
- ##### python -m venv venv
- ##### source venv/bin/activate  ##### Linux/Mac
- ##### venv\Scripts\activate     #####Windows

 #### iii) Install dependencies
- ##### pip install -r requirements.txt

### 2. Download Dataset
- #### Get spam_ham_dataset.csv from:
- ##### https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- ##### Or use the provided sample dataset in /data/

### 3. Quick Test
#### i)Full pipeline test
- python spam_detector.py

#### ii)Single email prediction test
- python predict.py "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005"

### 4. Expected Results

- ✅ Test accuracy: >98%
- ✅ ROC-AUC: >0.99
- ✅ F1-Score (Spam): >0.90
- ✅ All visualizations generated

### 5. Custom Testing
#### i)Test your own email
- python predict.py "Your test email text here"

#### ii)Expected output:
- ##### {'prediction': 'spam', 'confidence': 0.95, 'probabilities': {'ham': 0.05, 'spam': 0.95}}

### 🔍 Demo Screenshots
#### Confusion Matrix | ROC Curve | Precision-Recall

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/53ced339-8f62-4ee7-a90b-c5f6778cdbba" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/011f9b95-9a32-4163-a475-9fd957327d02" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/78d99b8b-b769-457c-ae4d-1d7cdccd3247" />




