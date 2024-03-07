from utils.dataloader import createDataLoader
from sklearn.model_selection import train_test_split
import torch

def createSplit(sentences,labels,batch_size):
    sentences_train, sentences_test, labels_train, labels_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)
    sentences_train, sentences_val, labels_train, labels_val = train_test_split(sentences_train, labels_train, test_size=0.2, random_state=42)
    train_dataloader=createDataLoader(sentences_train,labels_train,batch_size)
    test_dataloader=createDataLoader(sentences_test,labels_test,batch_size)
    val_dataloader=createDataLoader(sentences_val,labels_val,batch_size)
    return train_dataloader ,test_dataloader,val_dataloader



def validate(model, dataloader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs
            _, predicted = torch.max(logits, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    model.train()
    return correct_predictions / total_predictions * 100.0