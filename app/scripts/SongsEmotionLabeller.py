import os
import openai
import time
import requests
import pandas as pd

# Set your OpenAI API key
openai.api_key = os.environ.get('$OPEN_AI_API_KEY') 

# List of emotions
emotions = [
    "Happiness", "Contentment", "Confidence", "Neutral", "Sadness",
    "Anger", "Fear", "Surprise", "Disgust", "Love",
    "Excitement", "Anticipation", "Nostalgia", "Confusion",
    "Frustration", "Longing", "Optimism"
]

input_file = 'data/songs.csv'
output_dir = 'data'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df_songs = pd.read_csv(input_file)

for emotion in emotions:
    df_songs[emotion] = 0 

def get_emotions_from_response(response):
    extracted_emotions = {emotion: 0 for emotion in emotions}
    for emotion in response.split(', '):
        if emotion in extracted_emotions:
            extracted_emotions[emotion] = 1
    return extracted_emotions

def process_songs(df):
    for index, row in df.iterrows():
        song_name = row['Song Name']
        artists = row['Artists']

        print(f"Processing '{song_name}' by {artists}...")  

        prompt = f"Identify the emotions from this list - {', '.join(emotions)} - that are most likely conveyed in the song '{song_name}' by {artists}. Return 3 emotion names only from this list:"

        while True:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                print(f"Emotions identified for '{song_name}' by {artists}: {response.choices[0].message.content}")  # Debug message
                break
            except openai.error.Timeout as e:
                print(f"Timeout error for '{song_name}' by {artists}: {e}. Retrying...")
                time.sleep(5)
            except openai.error.APIError as e:
                print(f"API error for '{song_name}' by {artists}: {e}. Retrying...")
                time.sleep(10)
            except openai.error.ServiceUnavailableError as e:
                print(f"Service unavailable error for '{song_name}' by {artists}: {e}. Sleeping for 30 minutes...")
                time.sleep(1800)
            except (requests.exceptions.ConnectionError, openai.error.APIConnectionError) as e:
                print(f"Network error for '{song_name}' by {artists}: {e}. Retrying...")
                time.sleep(2)
            except openai.error.RateLimitError as e:
                print(f"Rate limit exceeded for '{song_name}' by {artists}: {e}. Waiting before retrying...")
                time.sleep(120)  

        emotion_values = get_emotions_from_response(response.choices[0].message.content)

        for emotion, value in emotion_values.items():
            df.at[index, emotion] = value

        time.sleep(1.3)  

    return df

df_processed = process_songs(df_songs)

output_file_path = os.path.join(output_dir, 'processed_songs.csv')
df_processed.to_csv(output_file_path, index=False)

print("Processing completed.")
