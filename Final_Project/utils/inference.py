import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def test(model, test_dataloader, device):
    model.load_state_dict(torch.load('last_model_weights.pth'))
    model.eval()  # Set the model to evaluation mode
    predictions = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs, 1)
            
            predictions.extend(predicted.tolist())
    
    return predictions


def evaluate_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    
    return accuracy, precision, recall, f1

def inference(model, test_dataloader, device):
    true_labels = [label for batch in test_dataloader for label in batch["labels"]]
    predicted_labels = test(model, test_dataloader, device)
    accuracy, precision, recall, f1 = evaluate_metrics(true_labels, predicted_labels)
    print(predicted_labels)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)