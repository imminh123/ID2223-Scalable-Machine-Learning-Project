import pandas as pd
import os
from app import app

def select_best_matching_song(emotion_predictions, num_results=3):
    # Path to the song data
    data_path = os.path.join(app.root_path, '..', 'data', 'processed_songs.csv')
    song_data = pd.read_csv(data_path)

    # Calculate matches for each song based on the input emotions
    emotion_match_count = song_data.loc[:, emotion_predictions].sum(axis=1)
    song_data['Total_Matches'] = emotion_match_count

    # Determine the maximum number of matches
    highest_match_count = emotion_match_count.max()

    # Filter songs that have the maximum match count
    optimal_songs = song_data[song_data['Total_Matches'] == highest_match_count]

    # Sort by 'Times_played' to prioritize less played songs if there are ties
    if len(optimal_songs) > 5:
        optimal_songs = optimal_songs.sort_values(by='Times_played')

    # Get the top N results
    selected_songs = optimal_songs.head(num_results)

    # Update the play count for these songs
    for idx in selected_songs.index:
        song_data.at[idx, 'Times_played'] += 1

    # Save the modified data back to the CSV file
    song_data.to_csv(data_path, index=False)

    # Return the list of track IDs for the selected songs
    return selected_songs['Track_ID'].values.tolist()
