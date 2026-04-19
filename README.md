# 📩 Email/SMS Spam Classifier

A machine learning web application that classifies messages as **Spam** or **Not Spam** using NLP techniques.

---

## 🚀 Live Demo

🔗 https://email-sms-spam-classifier-app.streamlit.app/

---

## 📌 Project Overview

This project builds an end-to-end spam detection system:

* Text preprocessing using NLP techniques
* Feature extraction using TF-IDF
* Model training using Multinomial Naive Bayes
* Deployment using Streamlit

---

## 🧠 Tech Stack

* Python
* Scikit-learn
* NLTK
* Streamlit

---

## 📂 Dataset

* SMS Spam Collection Dataset
* Contains labeled messages:

  * `0` → Not Spam
  * `1` → Spam

---

## 🔍 Exploratory Data Analysis (EDA)

* Checked class distribution (imbalanced dataset)
* Analyzed message lengths
* Observed spam messages contain:

  * promotional words (free, win, offer)
  * higher frequency of certain keywords

---

## ⚙️ Text Preprocessing

Steps performed:

1. Convert text to lowercase
2. Remove special characters using regex
3. Tokenize text using `.split()`
4. Remove stopwords (with custom retention for important words)
5. Apply stemming using PorterStemmer

---

## 🔢 Feature Engineering

* Used **TF-IDF Vectorizer**
* Parameters:

  * `max_features=5000`
  * `ngram_range=(1,2)` (unigrams + bigrams)

---

## 🤖 Model Training

Models tested:

* Gaussian Naive Bayes
* Multinomial Naive Bayes
* Bernoulli Naive Bayes
* Other ML models (SVM, Random Forest, etc.)

### ✅ Final Model:

**Multinomial Naive Bayes**

* Best precision for spam detection
* Works well with text frequency features

---

## 💾 Model Export

```python
pickle.dump(tfidf, open('vectorizer.pkl','wb'))
pickle.dump(model, open('model.pkl','wb'))
```

---

## 🌐 Streamlit App

Features:

* User input for message
* Real-time prediction
* Clean UI with feedback messages

---

## 🖥️ How to Run Locally

### 1. Clone Repository

```
git clone <your-repo-link>
cd sms-spam-classifier
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Run App

```
streamlit run app.py
```

---

## 📸 Screenshots

### 🔹 App Interface

(Add screenshot here)

### 🔹 Prediction Example

(Add screenshot here)

---

## 📈 Future Improvements

* Improve accuracy with advanced models (e.g., XGBoost, LSTM)
* Add message history tracking
* Deploy with authentication system

---

## ⭐ Acknowledgements

* Dataset from UCI Machine Learning Repository
* Inspired by NLP spam detection problems
