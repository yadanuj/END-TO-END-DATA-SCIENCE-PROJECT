from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        review_vector = vectorizer.transform([review.lower()])
        prediction = model.predict(review_vector)[0]
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        return render_template('index.html', prediction=sentiment, review=review)

if __name__ == '__main__':
    app.run(debug=True)
