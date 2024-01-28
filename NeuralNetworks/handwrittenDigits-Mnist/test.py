import torch
import torchvision  
import torch.nn.functional as F
from model import Net
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

batch_size_test = 1000
random_seed = 1
torch.manual_seed(random_seed)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                               ])),
    batch_size=batch_size_test, shuffle=True)
test_data = torch.cat([data for data, _ in test_loader], dim=0)
mean_value = test_data.mean()
std_value = test_data.std()
print(mean_value)
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((mean_value,), (std_value,))
])
test_loader.dataset.transform = transform

network = Net()
model_path = './results/sgdmodel.pth'
network_state_dict = torch.load(model_path)
network.load_state_dict(network_state_dict)

def test():
    network.eval()
    test_loss = 0
    correct = 0
    examples_shown = 0
    all_preds = []
    all_targets = []

    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    axes = axes.flatten()

    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum()

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())


            for i in range(min(10, batch_size_test)):
                if examples_shown < 10: 
                    axes[examples_shown].imshow(data[i].numpy().squeeze(), cmap='gray')
                    axes[examples_shown].set_title(f'Predicted: {pred[i].item()}, Actual: {target[i].item()}')
                    examples_shown += 1

    plt.show()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    f1 = f1_score(all_targets, all_preds, average='weighted')
    print('F1 Score: {:.4f}'.format(f1))

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

test()
