# perform training on neural networks

import os
from cnn import CNN
from yoga_dataset import YogaDataset
import config
# import tqdm
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader


def train(train_loader, val_loader, model, optimizer, loss_func):
    # to calculate running training loss
    run_loss = 0

    model.train()
    for input, target in train_loader:
        # put tensors on correct device
        input, target = input.to(config.DEVICE), target.to(config.DEVICE)
        
        # forward pass
        output = model(input)
        loss = loss_func(output, target)
        run_loss += loss.detach().item()

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # test on validation set
    with torch.no_grad():
        model.eval()
        for val_inp, val_targ in val_loader:
            val_inp, val_targ = val_inp.to(config.DEVICE), val_targ.to(config.DEVICE)
            
            val_out = model(val_inp)
            val_loss = loss_func(val_out, val_targ).item()
            pred_class = val_out.argmax(dim=1)
            val_acc = (pred_class == val_targ).sum() / val_inp.shape[0]

    return run_loss / len(train_loader), val_loss, val_acc


def main():
    # initialize things needed during train loop
    train_dataset = YogaDataset('train_data', config.MODEL)
    train_loader = DataLoader(
        train_dataset,
        config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=config.NUM_WORKERS
    )

    val_dataset = YogaDataset('val_data', config.MODEL)
    val_loader = DataLoader(
        val_dataset,
        batch_size=len(os.listdir('val_data')),
        shuffle=False,
        pin_memory=True,
        num_workers=config.NUM_WORKERS
    )

    if config.MODEL == 'fc':
        model = None
    elif config.MODEL == 'cnn':
        model = CNN([3, 16, 32, 64])

    model = model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), config.LEARNING_RATE)
    loss_func = nn.CrossEntropyLoss().to(config.DEVICE)

    # begin training loop
    print('Training started!!')
    best_model = None
    best_val_acc = 0

    for epoch in range(config.NUM_EPOCHS):
        loss, val_loss, val_acc = train(
            train_loader,
            val_loader,
            model,
            optimizer,
            loss_func
        )

        # check for improvement in validation loss
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model

        # print statistics about epoch
        print(f'Epoch [{epoch + 1}/{config.NUM_EPOCHS}]')
        print(f'\t Train loss = {loss}')
        print(f'\t Validation loss = {val_loss}')
        print(f'\t Validation accuracy = {val_acc}')
        print('=' * 100)

    # save best performing model
    torch.save(
        best_model.state_dict(),
        f'pytorch_models/{config.MODEL}/model.pth.tar'
    )


if __name__ == '__main__':
    main()