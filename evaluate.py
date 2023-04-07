# calculates accuracy on validation and test sets

import os
import config
import torch
from cnn import CNN
from fc import FC
from yoga_dataset import YogaDataset
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score


def evaluate(model_type):
    # instantiate val and test data loaders
    val_dataset = YogaDataset('val', model_type)
    val_loader = DataLoader(
        val_dataset,
        batch_size=len(val_dataset),
        shuffle=False,
        pin_memory=True,
        num_workers=config.NUM_WORKERS
    )

    test_dataset = YogaDataset('test', model_type)
    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=False,
        pin_memory=True,
        num_workers=config.NUM_WORKERS
    )

    # load in model weights
    if model_type == 'fc':
        model = FC([69, 128, 256, 512])
    elif model_type == 'cnn':
        model = CNN([3, 16, 32, 64])

    model.load_state_dict(
        torch.load(
        f'pytorch_models/{model_type}/model.pth.tar',
        config.DEVICE
        )
    )
    
    # test on validation set
    with torch.no_grad():
        model.eval()
        for val_inp, val_targ in val_loader:
            val_inp, val_targ = val_inp.to(config.DEVICE), val_targ.to(config.DEVICE)
            
            val_out = model(val_inp)
            pred_class = val_out.argmax(dim=1)
            val_acc = balanced_accuracy_score(val_targ, pred_class)

    print(f'Validation accuracy = {val_acc:.4f}')

    # test on test set
    with torch.no_grad():
        model.eval()
        for test_inp, test_targ in test_loader:
            test_inp, test_targ = test_inp.to(config.DEVICE), test_targ.to(config.DEVICE)
            
            test_out = model(test_inp)
            pred_class = test_out.argmax(dim=1)
            test_acc = balanced_accuracy_score(test_targ, pred_class)

    print(f'Test accuracy = {test_acc:.4f}')    