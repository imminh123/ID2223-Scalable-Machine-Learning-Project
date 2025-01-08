from flask import Flask

# Initialize the Flask application
app = Flask(__name__)

# Import routes
from app import routes

# You can also initialize other components here
# (e.g., database, login manager, etc.)

