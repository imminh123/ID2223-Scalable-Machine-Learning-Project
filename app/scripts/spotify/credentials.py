import os

class APIKeys:
    def get_client_id():
        #Returns the Spotify Client ID from the set environment variables
        return os.environ.get('SPOTIFY_CLIENT_ID')

    def get_client_secret():
        #Returns the Spotify Client Secret from the set environment variables
        return os.environ.get('SPOTIFY_CLIENT_SECRET')

