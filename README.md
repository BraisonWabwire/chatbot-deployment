# Chatbot Project

This is a Django-based chatbot application that uses a trained neural network model to classify user inputs and provide responses based on predefined intents. The chatbot is designed to handle various medical-related queries and general interactions, leveraging TensorFlow for machine learning and NLTK for natural language processing.
Features
Neural Network Model: Trained using TensorFlow/Keras to classify user inputs into intents.

Natural Language Processing: Uses NLTK for tokenization and lemmatization.

Web Interface: A simple Django-powered web interface for interacting with the chatbot.

Deployable: Can be deployed locally or on platforms like Heroku.

# **Prerequisites**

Python 3.10 or higher

Git (for version control and deployment)

Required Python libraries (listed in requirements.txt)

# **Training the Model**

If you need to retrain the model:
Modify intents1.json with new patterns or responses.

Use the training script named as "training_script.py"

Replace chatbot_model.h5, words.pkl, and classes.pkl with the newly generated files.



