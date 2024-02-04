import torch
import torchvision  
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from model import Net
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

n_epochs = 1
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

def train(epoch,network,optimizer,max_acc,name):
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
<<<<<<< HEAD
                100. * batch_idx / len(train_loader), loss.item())) 
        train_losses.append(loss.item())
        train_counter.append(batch_idx)       
=======
                100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
>>>>>>> 61aa876edf9930cb203d35d4796eb030a1798a81
    accuracy=val_acc(network,val_loader)
    if max_acc<accuracy:
        torch.save(network.state_dict(), './results/{}model.pth'.format(name))
        torch.save(optimizer.state_dict(), './results/{}optimizer.pth'.format(name))
        max_acc=accuracy
    return train_losses,train_counter,max_acc

<<<<<<< HEAD
=======

>>>>>>> 61aa876edf9930cb203d35d4796eb030a1798a81
def val_acc(network,loader):
  network.eval()
  correct = 0
  with torch.no_grad():
    for data, target in loader:
      output = network(data)
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  return (100. * correct / len(loader.dataset))


adam_counter=[]
adam_losses=[]
max_acc=0
for epoch in range(1, n_epochs + 1):
<<<<<<< HEAD
    losses,counter,acc=train(epoch,adamNetwork,adam_optimizer,max_acc=max_acc,name="adam",)
    adam_counter+=counter
    adam_losses+=losses
=======
    losses,counter,acc=train(epoch,adamNetwork,adam_optimizer,max_acc,name="adam")
    adam_counter+=losses
    adam_losses+=counter
>>>>>>> 61aa876edf9930cb203d35d4796eb030a1798a81
    max_acc=acc


sgd_counter=[]
sgd_losses=[]
max_acc=0
for epoch in range(1, n_epochs + 1):
<<<<<<< HEAD
    losses,counter,acc=train(epoch,sgdNetwork,sgd_optimizer,max_acc=max_acc,name="sgd",)
    sgd_counter+=counter
    sgd_losses+=losses
=======
    losses,counter,acc=train(epoch,sgdNetwork,sgd_optimizer,max_acc,name="sgd")
    sgd_counter+=losses
    sgd_losses+=counter
>>>>>>> 61aa876edf9930cb203d35d4796eb030a1798a81
    max_acc=acc
    

print("Validation Set Details for ADAM")
print("Accuracy : {}".format(val_acc(adamNetwork,val_loader)))
print("Validation Set Details for SGD")
print("Accuracy : {}".format(val_acc(sgdNetwork,val_loader)))
<<<<<<< HEAD
=======

>>>>>>> 61aa876edf9930cb203d35d4796eb030a1798a81

fig = plt.figure()
plt.plot(adam_counter, adam_losses, color='blue')
plt.plot(sgd_counter, sgd_losses, color='orange')
plt.legend(['Adam Loss', 'SGD Loss'], loc='upper right')
plt.xlabel('Number of Batches')
plt.ylabel('negative log likelihood loss')
plt.show()
