import torch

def train_for_epochs(model, train_loader, train_dataset,  val_loader, val_dataset, device, learning_rate = 0.001, momentum = 0.9, number_epochs = 10):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum)

    train_loss = []
    val_loss = []

    for epoch in range(number_epochs):
        print(f"Starting epoch {epoch}")

        model.train()
        train_loss.append(0)
        for images, labels in train_loader:
            
            images = list(image.to(device) for image in images)
            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

            loss_dict = model(images, labels)

            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step() 

            train_loss[epoch] += losses.cpu().data
        train_loss[epoch] /= len(train_dataset)

        if epoch > 0:
            print(f'Training Loss : {train_loss[epoch]}')

        val_loss.append(0)
        with torch.no_grad():
            for images, labels in val_loader:
                images = list(image.to(device) for image in images)
                labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

                loss_dict = model(images, labels)
                losses = sum(loss for loss in loss_dict.values())
                
                val_loss[epoch] += losses.cpu().data
            val_loss[epoch] /= len(val_dataset)

        if epoch > 0:
            print(f'Validation Loss : {val_loss[epoch]}')