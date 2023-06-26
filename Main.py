import torch.cuda
from torch.utils.data import DataLoader, random_split
from Test import ClassificationModel
from DataSet import ImageDataset
import torch.optim as optim
from torch import nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'


total_dataset = torch.load('total_dataset.pt')
train_dataset, test_dataset = random_split(total_dataset, [0.8, 0.2])
train_dataloader = DataLoader(train_dataset, batch_size = 16, shuffle = True, drop_last = False)
test_dataloader = DataLoader(test_dataset, batch_size = 16, shuffle = False, drop_last = False)



criterion = nn.CrossEntropyLoss()
net = ClassificationModel()
net = net.to(device)
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum = 0.9, weight_decay=1e-4)
net.load_state_dict()
losses_train = []
losses_test = []
accuracies_test = []

for epoch in tqdm(range(5)):  # loop over the dataset multiple times
    loss_train = 0
    num_train = 0
    net.train()
    start_time = time.time()
    for data in train_dataloader:
        # get the inputs; data is a list of [inputs, labels]
        tensor_img_augmented, tensor_img, labels = data
        inputs = tensor_img_augmented.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_train += loss.item() * len(inputs)
        num_train += len(inputs)

    losses_train.append(loss_train / num_train)
    elapsed_time = time.time() - start_time

    loss_test = 0
    num_test = 0

    acc_test = 0
    net.eval()
    for data in test_dataloader:
        tensor_img_augmented, tensor_img, labels = data
        inputs = tensor_img.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = net(inputs)
            label_pred = torch.argmax(outputs, dim=1)
            acc_test += (label_pred == labels).sum().item()
            loss = criterion(outputs, labels)
            loss_test += loss.item() * len(inputs)
            num_test += len(inputs)

    losses_test.append(loss_test / num_test)
    accuracies_test.append(100 * acc_test / num_test)

    print(f"epoch: {epoch}, loss train : {losses_train[-1]}, loss test: {losses_test[-1]}, acc test: {accuracies_test[-1]}%, time = {elapsed_time}")

plt.plot(list(range(0, len(losses_train))), losses_train)
plt.plot(list(range(0, len(losses_test))), losses_test)
plt.show()

plt.plot(list(range(0, len(accuracies_test))), accuracies_test)
plt.show()

state_dict = net.state_dict()
torch.save(state_dict, "Final files/checkpoint.pth")