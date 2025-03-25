from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt  # Add this for POST requests
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model # type: ignore
import pickle
import random

# Initialize NLTK and load resources
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Load model and data
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
with open('intents1.json', 'r') as file:
    intents = json.load(file)["intents"]

# Preprocessing functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't understand that."
    tag = intents_list[0]['intent']
    for i in intents_json:  # intents_json is already a list, no need for ['intents']
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "Sorry, I didn't understand that."

def chatbot_response(text):
    ints = predict_class(text, model)
    res = get_response(ints, intents)
    return res

# Views
def index(request):
    return render(request, 'chatbot/index.html')

@csrf_exempt  # Exempt CSRF for simplicity (use with caution in production)
def get_response_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        message = data.get('message', '')
        response = chatbot_response(message)
        return JsonResponse({'response': response})
    return JsonResponse({'error': 'Invalid request'}, status=400)