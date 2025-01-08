# app/routes.py
from flask import render_template, jsonify, request
from threading import Thread
from app import app
import pandas as pd
import random
import os
from .model.emotion_predictor import get_emotion_from_text
from .helper import select_best_matching_song
from .model.bertmodel import trainning
from flask_cors import CORS

# Enable CORS for the entire app
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "https://id-2223-song-recommender.vercel.app"]}})

@app.route('/')
def index():
    return "What are you doing here my friend?"

@app.route('/get_random_track', methods=['GET'])
def get_random_track():
    csv_path = os.path.join(app.root_path, '..', 'data', 'songs.csv')
    df = pd.read_csv(csv_path)
    random_track_ids = random.sample(df['Track_ID'].dropna().tolist(), 5)

    return jsonify(track_ids=random_track_ids)

@app.route('/get_emotion_based_song', methods=['POST'])
def get_emotion_based_song():
    data = request.json
    text = data.get('text', '')

    if text:
        predicted_emotions = get_emotion_from_text(text)
        track_id = select_best_matching_song(predicted_emotions)
        return jsonify(track_id=track_id)  
    else:
        return jsonify({"error": "No text provided"}), 400


@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    text = data.get('text', '')
    emotions = data.get('emotions', [])

    if text and emotions:
        # Load the dataset
        file_path = os.path.join(app.root_path, '..', 'data', 'Finalpreprocessed_data.csv')
        df = pd.read_csv(file_path)

        # Create a new row with the user feedback
        new_row = {
            'Cleaned_Dialogue': text
        }
        for emotion in emotions:
            new_row[emotion] = 1

        # Append the new row to the top of the DataFrame
        df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)

        # Save the updated DataFrame back to the CSV
        df.to_csv(file_path, index=False)

        return jsonify({"message": "Feedback received successfully"})
    else:
        return jsonify({"error": "Invalid data provided"}), 400

def background_task():
    """Run the BERT model in a separate thread."""
    trainning()

@app.route('/run-model', methods=['POST'])
def run_model():
    """Trigger the BERT model as a background task."""
    thread = Thread(target=background_task)
    thread.start()
    return jsonify({"message": "BERT model is running in the background"}), 202