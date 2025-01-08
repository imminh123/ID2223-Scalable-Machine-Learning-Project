import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.nn import BCEWithLogitsLoss
import pandas as pd
from tqdm import tqdm  # for progress bars

def trainning():
    # Load the preprocessed CSV file
    df = pd.read_csv('data/Friends_PreProcessed_data.csv')

    # Initialize the tokenizer for BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the dialogues
    tokenized_data = tokenizer.batch_encode_plus(
        df['Cleaned_Dialogue'].tolist(),
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Extract input IDs and attention masks from the tokenized data
    input_ids = tokenized_data['input_ids']
    attention_masks = tokenized_data['attention_mask']

    # Define your emotion labels
    emotions = [
        "Happiness", "Contentment", "Confidence", "Neutral", "Sadness",
        "Anger", "Fear", "Surprise", "Disgust", "Love",
        "Excitement", "Anticipation", "Nostalgia", "Confusion",
        "Frustration", "Longing", "Optimism"
    ]
    labels = df[emotions].values

    # Custom dataset class
    class EmotionDataset(Dataset):
        def __init__(self, input_ids, attention_masks, labels):
            self.input_ids = input_ids
            self.attention_masks = attention_masks
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_masks[idx],
                'labels': torch.tensor(self.labels[idx], dtype=torch.float)
            }

    #Create the dataset
    dataset = EmotionDataset(input_ids, attention_masks, labels)

    #DataLoader
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(emotions)
    )

    optimizer = AdamW(model.parameters(), lr=5e-5)

    loss_fn = BCEWithLogitsLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        print(f"Starting epoch {epoch+1}/{epochs}")

        
        batch_progress = tqdm(enumerate(train_loader), total=len(train_loader), desc="Batch")
        for step, batch in batch_progress:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            model.zero_grad()

            outputs = model(b_input_ids, attention_mask=b_input_mask)

            loss = loss_fn(outputs.logits, b_labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            batch_progress.set_postfix({'Training Loss': '{:.3f}'.format(loss.item())})

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Average training loss: {avg_train_loss}")

    # Save the model
    model.save_pretrained('./saved_model')
    tokenizer.save_pretrained('./saved_model')

    # Push to Hugging Face Hub
    huggingface_token = ""
    repo_id = "minhnguyen5293/bert-base-uncased-emotion-classifier"
    tokenizer.push_to_hub(repo_id, use_auth_token=huggingface_token)

    model.push_to_hub(repo_id, use_auth_token=huggingface_token)