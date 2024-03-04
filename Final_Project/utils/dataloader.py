import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

class CustomDataset(Dataset):
    def __init__(self, texts, labels, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        attention_mask_flattened = encoding['attention_mask'].flatten()

        # Convert to torch tensor of type bool
        attention_mask = torch.tensor(attention_mask_flattened, dtype=torch.float)
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }
    

def createDataLoader(texts,labels,batch_size):
    dataset = CustomDataset(texts, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
    


if __name__=="__main__":
    # texts = ["Example text 1", "Another example text", "And one more for good measure"]
    texts = ["Example text 1"]
    # labels = [0, 1, 0]  # Example labels
    labels = [0]
    dataloader=createDataLoader(texts,labels,3)
    for batch_idx, batch in enumerate(dataloader):
        print("Batch contents:", batch)
