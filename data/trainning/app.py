import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
from typing import Dict, Text
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna(subset=["Cleaned_Dialogue", "Song Recommendations"])
    data["Cleaned_Dialogue"] = data["Cleaned_Dialogue"].astype(str)
    data["Song Recommendations"] = data["Song Recommendations"].astype(str)

    data["Cleaned_Dialogue"] = data["Cleaned_Dialogue"].fillna("")
    data["Song Recommendations"] = data["Song Recommendations"].fillna("")

    return data


data = load_data("friends_positive_with_songs_lite.csv")
songs_raw = pd.read_csv("processed_songs_original_01.csv")

# Ensure the relevant columns are strings
data["Cleaned_Dialogue"] = data["Cleaned_Dialogue"].astype(str)
data["Song Recommendations"] = data["Song Recommendations"].astype(str)

cleaned_dialogue_list = data["Cleaned_Dialogue"].tolist()
song_recommendations_list = data["Song Recommendations"].tolist()

songs_data = tf.data.Dataset.from_tensor_slices({
    "song": songs_raw['Song Name'].tolist(),
    "track_id": songs_raw['Track_ID'].tolist(),
})

tf_data = tf.data.Dataset.from_tensor_slices({
    "user_input": cleaned_dialogue_list,
    "song": song_recommendations_list
})

tf_data = tf_data.shuffle(10000)

songs_data_map = songs_data.map(lambda x: x["song"]).unique()

# Define the model
class SongRecommendationModel(tfrs.Model):
    def __init__(self, 
    user_model: tf.keras.Model,
    song_model: tf.keras.Model,
    task: tfrs.tasks.Retrieval):
        super().__init__()
        # Set up user and movie representations.
        self.user_model = user_model
        self.song_model = song_model
        self.task = task

        # Loss and metrics
        # unique_songs = tf.convert_to_tensor(data["Song Recommendations"].unique(), dtype=tf.string)
        candidates = songs_data_map.batch(128).map(
            lambda x: self.song_model(x)
        )

    def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        # Forward pass for the model
        user_embeddings = self.user_model(features["user_input"])
        song_embeddings = self.song_model(features["song_input"])
        return user_embeddings, song_embeddings

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # Define how the loss is computed.

        user_embeddings = self.user_model(features["user_input"])
        song_embeddings = self.song_model(features["song"])

        return self.task(user_embeddings, song_embeddings)

user_input_lookup = tf.keras.layers.StringLookup(mask_token=None)
user_input_lookup.adapt(tf_data.map(lambda x: x["user_input"]))

user_model = tf.keras.Sequential([
    user_input_lookup,
    tf.keras.layers.Embedding(user_input_lookup.vocabulary_size(), output_dim=64),
])

song_input_vectorization = tf.keras.layers.TextVectorization(max_tokens=10000, output_sequence_length=50)
song_input_vectorization.adapt(songs_data_map)  # Adapt on user dialogues

song_model = tf.keras.Sequential([
    song_input_vectorization,
    tf.keras.layers.Embedding(song_input_vectorization.vocabulary_size(), output_dim=64),
    tf.keras.layers.GlobalAveragePooling1D()  # Summarize sequence embeddings
])

task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
    songs_data_map.batch(128).map(song_model)
))

model = SongRecommendationModel(user_model, song_model, task)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
model.fit(tf_data.batch(1000), epochs=1)

# FastAPI app setup
app = FastAPI()

# Request and response schemas
class RecommendRequest(BaseModel):
    dialogue: str

class RecommendResponse(BaseModel):
    recommendations: List[str]

@app.get("/train")
async def train():
    try:
        # Instantiate and compile the model
        model = SongRecommendationModel(user_model, song_model, task)
        model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
        model.fit(tf_data.batch(2), epochs=1)
        model.build(input_shape={"user_input": (None,), "song_input": (None,)})
        model.save_weights('models/recommend_model_weights', save_format='tf')
        return {"message": "Model trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    try:
        # model = SongRecommendationModel(user_model, song_model, task)
        # model.load_weights('models/recommend_model_weights')
        loaded_model = model

        if loaded_model is None:
            print("Model not loaded")
        print("Model loaded successfully")

        print("Loading candidate song embeddings...")
        candidate_songs = list(data["Song Recommendations"].unique())  # Ensure unique candidates
        song_embeddings = loaded_model.song_model(tf.constant(candidate_songs))
        song_embeddings = tf.nn.l2_normalize(song_embeddings, axis=1)
        candidate_song_embeddings = song_embeddings

        # Preprocess user input
        user_input = tf.constant([request.dialogue])
        user_embeddings = model.user_model(user_input)
        user_embeddings = tf.nn.l2_normalize(user_embeddings, axis=1)

        # Compute similarity scores
        scores = tf.linalg.matmul(user_embeddings, candidate_song_embeddings, transpose_b=True)
        top_k = 5
        top_k_indices = tf.math.top_k(scores, k=top_k).indices.numpy()[0]

        # Retrieve top-k recommended songs
        recommendations = [candidate_songs[i] for i in top_k_indices]
        return RecommendResponse(recommendations=recommendations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# # Sample user input for inference
# test_user_input = ["oh oh"]

# # Preprocess the user input
# test_user_embeddings = loaded_model.user_model(tf.constant(test_user_input))
# test_user_embeddings = tf.nn.l2_normalize(test_user_embeddings, axis=1)
# # Retrieve candidate song embeddings
# candidate_songs = list(data["Song Recommendations"].unique())  # Ensure unique candidates

# candidate_song_embeddings = loaded_model.song_model(tf.constant(candidate_songs))
# candidate_song_embeddings = tf.nn.l2_normalize(candidate_song_embeddings, axis=1)

# # Compute similarity scores
# # Use dot product or another similarity metric to rank songs
# scores = tf.linalg.matmul(test_user_embeddings, candidate_song_embeddings, transpose_b=True)
# # Flatten scores to make them easier to work with

# # Combine candidate songs with their scores
# song_score_pairs = list(zip(candidate_songs, scores.numpy().flatten()))

# # Sort songs by scores in descending order
# sorted_song_score_pairs = sorted(song_score_pairs, key=lambda x: x[1], reverse=True)

# # print("All Songs with Scores:")
# # for song, score in sorted_song_score_pairs:
# #     print(f"{song}: {score:.4f}")

# # Get top-K recommendations
# top_k = 5  # Number of recommendations to retrieve
# top_k_indices = tf.math.top_k(scores, k=top_k).indices.numpy()
# recommended_songs = [candidate_songs[i] for i in top_k_indices[0]]

# # Display recommendations
# print("Recommended Songs:", recommended_songs)
