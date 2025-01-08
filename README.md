## Music Prediction on Emotion
<img src="https://github.com/imminh123/ID2223-Scalable-Machine-Learning-Project/blob/main/data/assets/app_screenshot.png?raw=true" alt="Application Architecture"  height="300">

## Overview
This project implements a music prediction system that uses user-provided emotional input to suggest songs. The system train a prediction model, provide real-time music recommendations and model re-train mechanism.

[**Demo URL**](https://id-2223-song-recommender.vercel.app/)


## Architecture
<img src="https://github.com/imminh123/ID2223-Scalable-Machine-Learning-Project/blob/main/data/assets/project_architecture.png?raw=true" alt="Application Architecture"  height="300">
<br>

The system's architecture is structured as follows:

1. **Input Data:**
   - **Conversation Data:** [Lines from the TV series Friends](https://www.kaggle.com/datasets/gopinath15/friends-netflix-script-data) are used as sample conversations.
   - **Songs Data:** A database of songs collected from a Spotify playlist of 500 songs is used for recommendation mapping.

2. **Labeling and Training:**
   - The **Conversation Data** is preprocessed and labeled with emotions to create a training dataset (GPT 3.5).
   - Similarly, **Songs Data** is processed into a dataset labeled with corresponding emotions (GPT 3.5).

3. **Model Training:**
   - A [**BERT-based Sentiment Analysis Model**](https://huggingface.co/minhnguyen5293/bert-base-uncased-emotion-classifier) is trained on the labeled conversation data to classify emotions.

4. **Inference:**
   - The trained sentiment model predicts the emotion from user input.
   - Based on the predicted emotion, the system retrieves relevant songs from the **Emotion With Labels** dataset.

5. **Feedback Loop:**
   - Users provide feedback on the recommendations, which is used to re-train and improve the model in subsequent iterations.

## Output
The inference pipeline is able to give relevant results base on user emotional input. We are awared that the limited 500 songs dataset might further limit the recommendation quality. This can be part of the future work.

Due to time shortage, we have not found an adequate way to evaluate the quality of the fine-tuned BERT model.

## How to Run
1. **Setup Environment:**
   - docker-compose up -d --build

2. **Train the Model:**
   - Train the sentiment model using the `app/app/model/bertmodel.py` script.

3. **Run Inference:**
   - Use the `run.py` script to run Flask application or docker-compose like the 1st step.
   - Retrieve song recommendations based on the predicted emotion.

## Future Enhancements
- Dynamically expand song dataset with more diverse genres and emotional labels update regularly.
- Incoporating feature store (Hopswork) for training & inference data.
- Improve emotion classification accuracy by incorporating multimodal data (e.g., voice tone).

