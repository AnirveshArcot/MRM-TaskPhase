import torch
import torchvision  
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from model import Net
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

n_epochs = 3
batch_size_train = 64
learning_rate_adam = 0.01
learning_rate_sgd = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

full_dataset = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor()
                                          ]))

full_data = torch.cat([data for data, _ in full_dataset], dim=0)
mean_value = full_data.mean()
std_value = full_data.std()

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size_train, shuffle=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size_train, shuffle=False)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((mean_value,), (std_value,))
])

train_loader.dataset.transform = transform
val_loader.dataset.transform = transform

examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
plt.show()


sgdNetwork = Net()
adamNetwork = Net()
adam_optimizer = optim.Adam(adamNetwork.parameters(), lr=learning_rate_adam)
sgd_optimizer = optim.SGD(sgdNetwork.parameters(), lr=learning_rate_sgd,momentum=momentum)

def train(epoch,network,optimizer,name):
    train_losses=[]
    train_counter=[]
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(), './results/{}model.pth'.format(name))
            torch.save(optimizer.state_dict(), './results/{}optimizer.pth'.format(name))
    torch.save(network.state_dict(), './results/{}model.pth'.format(name))
    torch.save(optimizer.state_dict(), './results/{}optimizer.pth'.format(name))
    return train_losses,train_counter




def validate(network,loader):
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(loader.dataset)
  print('Val set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(loader.dataset),
    100. * correct / len(loader.dataset)))


adam_counter=[]
adam_losses=[]
for epoch in range(1, n_epochs + 1):
    losses,counter=train(epoch,adamNetwork,adam_optimizer,name="adam")
    adam_counter+=losses
    adam_losses+=counter


sgd_counter=[]
sgd_losses=[]
for epoch in range(1, n_epochs + 1):
    losses,counter=train(epoch,sgdNetwork,sgd_optimizer,name="sgd")
    sgd_counter+=losses
    sgd_losses+=counter
    

print("Validation Set Details for ADAM")
validate(adamNetwork,val_loader)
print("Validation Set Details for SGD")
validate(sgdNetwork,val_loader)


fig = plt.figure()
plt.plot(adam_losses, adam_counter, color='blue')
plt.plot(sgd_losses, sgd_counter, color='orange')
plt.legend(['Adam Loss', 'SGD Loss'], loc='upper right')
plt.xlabel('Number of Samples')
plt.ylabel('negative log likelihood loss')
plt.show()