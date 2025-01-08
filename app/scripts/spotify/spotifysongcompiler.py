import spotipy
from spotipy.oauth2 import SpotifyOAuth
from credentials import APIKeys
import pandas as pd

# Retrieve Spotify API credentials
client_id = APIKeys.get_client_id()
client_secret = APIKeys.get_client_secret()

# Spotify OAuth settings
oauth_settings = {
    'client_id': client_id,
    'client_secret': client_secret,
    'redirect_uri': 'http://localhost:8000/callback',
    'scope': 'playlist-read-private'
}

# Initialize Spotify client with OAuth
spotify = spotipy.Spotify(auth_manager=SpotifyOAuth(**oauth_settings))

# Spotify playlist ID
playlist_id = '6uRb2P6XRj5ygnanxpMCfS'

# Function to fetch all tracks from a playlist
def fetch_playlist_tracks(playlist_id):
    response = spotify.playlist_tracks(playlist_id)
    tracks_list = response['items']

    # Paginate through the playlist if it has many tracks
    while response['next']:
        response = spotify.next(response)
        tracks_list.extend(response['items'])

    return tracks_list

# Fetch tracks
tracks = fetch_playlist_tracks(playlist_id)

# Prepare data for DataFrame
track_data = []

# Process track information
for i, track_item in enumerate(tracks):
    track_info = track_item.get('track')
    if not track_info:
        print(f"Track {i+1} is unavailable or missing data.")
        continue

    # Collect necessary track details
    song_title = track_info['name']
    artist_details = ', '.join(artist['name'] for artist in track_info['artists'])
    first_artist_id = track_info['album']['artists'][0]['id'] if track_info['album']['artists'] else None
    track_unique_id = track_info['id']

    # Append track data to list
    track_data.append({
        'Song Title': song_title,
        'Artists': artist_details,
        'Artist ID': first_artist_id,
        'Track ID': track_unique_id
    })

# Convert track data into a DataFrame
track_df = pd.DataFrame(track_data)

# Save to CSV
track_df.to_csv('data/songs.csv', index=False)
