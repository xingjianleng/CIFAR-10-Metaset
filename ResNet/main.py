import torch

from dataloader import get_trainloader, get_testloader
from modelloader import ResNet152
from train import train_model
from test import test_model

# parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-1
batch_size = 128
epochs = 200


trainloader = get_trainloader(batch_size=batch_size)
testloader = get_testloader(batch_size=batch_size)
model = ResNet152(device=device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr,
    momentum=0.9, weight_decay=1e-4
)

for epoch in range(epochs):
    # training part
    print(f"\nEpoch {epoch + 1}:")
    train_model(
        model=model,
        trainloader=trainloader,
        device=device,
        optimizer=optimizer,
        loss_fn=loss_fn
    )

# eval part
test_loss, test_acc = test_model(model=model, testloader=testloader, device=device, loss_fn=loss_fn)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_acc}")

# save model parameters
torch.save(model.state_dict(), f"../model/resnet152-{epochs}.pt")
