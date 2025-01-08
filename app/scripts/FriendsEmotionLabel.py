import os
import csv
import openai
import time
import requests

# Set your OpenAI API key
openai.api_key = os.environ.get('$OPEN_AI_API_KEY')

# List of emotions
emotions = [
    "Happiness", "Contentment", "Confidence", "Neutral", "Sadness",
    "Anger", "Fear", "Surprise", "Disgust", "Love",
    "Excitement", "Anticipation", "Nostalgia", "Confusion",
    "Frustration", "Longing", "Optimism"
]

input_dir = 'friendscsv'
output_dir = 'finaldata'

# Create output directory if not exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def get_emotions_from_response(response):
    extracted_emotions = []
    for emotion in emotions:
        if emotion in response:
            extracted_emotions.append(emotion)
    return extracted_emotions

def process_file(file_path, filename_without_ext):
    processed_data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dialogue = row['dialogue']

            while True:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{
                            "role": "user",
                            "content": "Analyze the dialogue: '{}'. Identify three emotions from this list - {} - that best convey the overall sentiment of the dialogue. Return 3 emotion names only from this list:".format(dialogue, ', '.join(emotions))
                        }]
                    )
                    break
                except openai.error.Timeout as e:
                    print(f"Timeout error: {e}. Retrying...")
                    time.sleep(5)
                except openai.error.APIError as e:
                    print(f"API error: {e}. Retrying...")
                    time.sleep(10)
                except openai.error.ServiceUnavailableError as e:
                    print(f"Service unavailable error: {e}. Sleeping for 30 minutes...")
                    time.sleep(1800)
                except (requests.exceptions.ConnectionError, openai.error.APIConnectionError) as e:
                    print(f"Network error: {e}. Retrying...")
                    time.sleep(2)
                except openai.error.RateLimitError as e:
                    print(f"Rate limit exceeded: {e}. Waiting before retrying...")
                    time.sleep(120)  # Wait for 2 minutes before retrying
                

            emotions_list = get_emotions_from_response(response.choices[0].message.content)
            joined_emotions = ', '.join(emotions_list) if emotions_list else "Unknown"
            print(f"Episode: {filename_without_ext}\nDialogue: {dialogue}\nIdentified Emotions: {joined_emotions}\n")
            processed_data.append({'Dialogue': dialogue, 'Emotions': joined_emotions})

            time.sleep(1.3)

    return processed_data

# Process each file in the input directory and save to new CSV files
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_dir, filename)
        filename_without_ext = os.path.splitext(filename)[0]  # Remove the .csv extension
        processed_data = process_file(file_path, filename_without_ext)

        # Write processed data to CSV
        output_file_path = os.path.join(output_dir, filename)
        with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Dialogue', 'Emotions'])
            writer.writeheader()
            for data in processed_data:
                writer.writerow(data)

        # Delete the file from input directory after processing
        os.remove(file_path)

print("Processing completed.")
