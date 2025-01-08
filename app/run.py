from app import app
import os

if __name__ == '__main__':
    if "DOCKER_ENV" in os.environ:
        # Running within Docker, use port 5000
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        # Running locally, use port 5001
        app.run(host='0.0.0.0', port=5001, debug=True)
