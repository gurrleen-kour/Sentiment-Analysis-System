import os
from dotenv import load_dotenv
from flask import Flask, render_template, request
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions, EmotionOptions 
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
load_dotenv()

app = Flask(__name__)

# --- HARD-CODE YOUR CREDENTIALS HERE ---
# This ensures the "None" error never happens again
IBM_KEY = os.getenv("IBM_KEY")
IBM_URL = os.getenv("IBM_URL")

nlu_auth = IAMAuthenticator(IBM_KEY)
nlu_service = NaturalLanguageUnderstandingV1(
    version='2021-03-25',
    authenticator=nlu_auth
)
nlu_service.set_service_url(IBM_URL)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    user_text = request.form.get('text')
    if not user_text:
        return render_template('index.html', error="Please enter text.")

    try:
        # We are now asking for Sentiment AND Emotion (Confidence Breakdown)
        response = nlu_service.analyze(
            text=user_text,
            features=Features(
                sentiment=SentimentOptions(),
                emotion=EmotionOptions() # Adds Joy, Sadness, etc.
            )
        ).get_result()
        
        sentiment_data = response['sentiment']['document']
        emotion_data = response['emotion']['document']['emotion']
        
        # Convert score to percentage for the progress bar
        # Score is -1 to 1, we turn it into 0% to 100%
        display_score = round((sentiment_data['score'] + 1) * 50, 1)
        
        return render_template('index.html', 
                               sentiment=sentiment_data['label'], 
                               score=display_score,
                               raw_score=sentiment_data['score'],
                               emotions=emotion_data,
                               original_text=user_text)
    except Exception as e:
        return render_template('index.html', error=str(e))
if __name__ == '__main__':
    app.run(debug=True)