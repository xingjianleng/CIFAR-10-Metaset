def train_model(model, trainloader, device, optimizer, loss_fn):
    # train one epoch
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(f"Loss for current epoch: {train_loss / len(trainloader)}")
    print(f"Accuracy for current epoch: {correct / total}")
