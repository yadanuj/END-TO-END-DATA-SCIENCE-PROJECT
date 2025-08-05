# END-TO-END-DATA-SCIENCE-PROJECT
*COMPANY*: CODTECH IT SOLUTIONS

 *NAME*: ANUJ YADAV

*INTERN ID*: CT04DH1868

*DOMAIN*: DATA SCIENCE

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTHOSH

# 🎨 IMDb Review Sentiment Analyzer using Flask & Machine Learning

---

## 📈 Project Overview

This repository contains a complete end-to-end **Sentiment Analysis Web Application**. The application is designed to predict whether an IMDb movie review expresses a **positive** or **negative** sentiment. It leverages a **Logistic Regression** machine learning model trained on real-world Twitter sentiment data, utilizing **TF-IDF vectorization** for text preprocessing. The model is deployed as a user-friendly web application using the **Flask** framework.

Users can input any movie review text via a simple HTML interface. The backend processes the input in real-time and outputs whether the sentiment is **Positive** or **Negative**.

---

## 💪 Key Features

- 🧠 **ML Model**: Logistic Regression trained on pre-labeled sentiment data.
- 💬 **User Input Form**: Submit any movie review text to be analyzed.
- 🔍 **Real-time Sentiment Prediction**: Immediate result displayed on the same page.
- 🌟 **Minimalist UI**: Simple and intuitive design built using HTML and CSS.
- ⚙️ **Flask-based Backend**: Routes user input to the trained model and renders the result.

---

## 🧰 Technologies Used

| Technology                      | Purpose                                  |
| ------------------------------- | ---------------------------------------- |
| Python 3.x                      | Core programming language                |
| Pandas                          | Data loading and preprocessing           |
| Scikit-Learn                    | Text vectorization and ML model training |
| Joblib                          | Model and vectorizer serialization       |
| Flask                           | Web framework for backend deployment     |
| HTML/CSS                        | Frontend interface                       |
| Seaborn & Matplotlib (optional) | Visualization tools                      |

---

## 🔧 Project Structure

```
.
├── train_model.py           # ML model training script
├── app.py                   # Flask app for prediction and routing
├── templates/
│   └── index.html           # HTML UI for input and output
├── model.pkl                # Trained Logistic Regression model
├── vectorizer.pkl           # Trained TF-IDF vectorizer
└── README.md                # Project documentation (this file)
```

---

## ⚖️ Workflow Breakdown

### 1. Model Training (`train_model.py`)

- Loads a pre-labeled sentiment dataset from GitHub.
- Cleans and normalizes the text data (lowercasing, removing nulls).
- Applies **TF-IDF vectorization** (top 5000 features).
- Trains a **Logistic Regression** classifier.
- Evaluates model performance with accuracy score.
- Saves model and vectorizer as `model.pkl` and `vectorizer.pkl` using **Joblib**.

### 2. Web Application (`app.py` and `index.html`)

- Loads saved model and vectorizer.
- Defines two Flask routes:
  - `'/'` renders the homepage with review input form.
  - `'/predict'` processes input, predicts sentiment, and renders result.
- UI dynamically displays the predicted sentiment (Positive/Negative).

---

## 🚀 How to Run the Project Locally

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/sentiment-analyzer.git
cd sentiment-analyzer
```

2. **Create & Activate Virtual Environment**

```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Train the Model (if not already done)**

```bash
python train_model.py
```

5. **Run the Flask Application**
```bash
   python app.py
```

6. **Access in Browser**

-Go to: http://127.0.0.1:5000/

---

## 📊 Results & Evaluation

- The trained Logistic Regression model achieves strong accuracy on test data.
- **Confusion Matrix** and metrics can be added for deeper evaluation.
- **Real-time predictions** allow qualitative analysis of model performance.

---

## 📊 Future Enhancements

- Deploy app using **Heroku**, **Render**, or **Docker**.
- Add **database** to store reviews and predictions.
- Integrate **real IMDb review dataset** for training.
- Add **user feedback loop** for retraining model.
- Enhance UI with **Bootstrap** or **React** frontend.

---

## Output

<img width="1915" height="877" alt="Image" src="https://github.com/user-attachments/assets/7b16617a-b4e5-41c5-b190-6d3220682c9f" />
