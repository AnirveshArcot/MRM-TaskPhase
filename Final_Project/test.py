import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW
import pandas as pd
from utils.train_utils import createSplit

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


df = pd.read_csv("./Files/spam.csv")
sentences = df['v2'].tolist()
labels = df['v1'].tolist()
train_loader ,test_loader,val_loader = createSplit(sentences,labels,4)






# Define the BERT model for sequence classification
model = BertForSequenceClassification(num_labels=2)  # Example: binary classification

# Set up optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)  # Example learning rate
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

for epoch in range(3):  # Example: 3 epochs
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

# Evaluation loop
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch in val_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    accuracy = correct / total
    print('Validation Accuracy: {:.2f}%'.format(100 * accuracy))
