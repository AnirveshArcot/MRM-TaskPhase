import torch
import torch.nn as nn
import pandas as pd
import os
from utils.model import BertModel 
import torch.optim as optim
from utils.train_utils import createSplit,validate
from tqdm import tqdm
from utils.inference import inference

def train(model, train_dataloader, val_dataloader, optimizer, criterion, epochs, device, save_path="best_model_weights.pth"):
    best_val_accuracy = 0.0

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # Wrap the train_dataloader with tqdm for progress bar
        train_dataloader = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}')

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask)
            print([outputs,labels])
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            print([predicted,labels])
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            train_dataloader.set_postfix({'loss': running_loss / (train_dataloader.n + 1)})  # Update the progress bar with current loss

        epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy = correct_predictions / total_predictions * 100.0
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

        # Validation
        val_accuracy = validate(model, val_dataloader, device)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print("Model weights saved.")
        print(f'Validation Accuracy: {val_accuracy:.2f}%')




def train_model(config):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_path = os.path.join(script_dir, '..', config['ds_dir'])
    df = pd.read_csv(absolute_path)
    batch_size = config['batch_size']
    num_layers = config['num_layers']
    hidden_size = config['hidden_size']
    num_attention_heads = config['num_attention_heads']
    intermediate_size = config['intermediate_size']
    num_embeddings = config['num_embeddings']
    dropout = config['dropout']
    num_classes = config['num_classes']
    learning_rate = config['learning_rate']
    epochs = config['epochs']
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    model = BertModel(num_layers, hidden_size, num_attention_heads, intermediate_size, num_embeddings,num_classes,device,dropout)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    sentences = df['v2'].tolist()
    labels = df['v1'].tolist()
    labels = [1 if label == 'spam' else 0 for label in labels]
    train_dataloader ,test_dataloader,val_dataloader = createSplit(sentences,labels,batch_size)
    model.to(device)
    train(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader,optimizer=optimizer, criterion=criterion, epochs=epochs, device=device)
    inference(model=model,test_dataloader=test_dataloader,device=device)

