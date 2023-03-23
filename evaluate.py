# calculates accuracy on validation and test sets

import os
import config
import torch
from cnn import CNN
from yoga_dataset import YogaDataset
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score


def evaluate():
    # instantiate val and test data loaders
    val_dataset = YogaDataset('val_data', config.MODEL)
    val_loader = DataLoader(
        val_dataset,
        batch_size=len(os.listdir('val_data')),
        shuffle=False,
        pin_memory=True,
        num_workers=config.NUM_WORKERS
    )

    test_dataset = YogaDataset('test_data', config.MODEL)
    test_loader = DataLoader(
        test_dataset,
        batch_size=len(os.listdir('test_data')),
        shuffle=False,
        pin_memory=True,
        num_workers=config.NUM_WORKERS
    )

    # load in model weights
    if config.MODEL == 'fc':
        pass
    elif config.MODEL == 'cnn':
        model = CNN([3, 16, 32, 64])

    model.load_state_dict(
        torch.load(
        f'pytorch_models/{config.MODEL}/model.pth.tar',
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
            # val_acc = (pred_class == val_targ).sum() / val_inp.shape[0]
            val_acc = balanced_accuracy_score(val_targ, pred_class)

    print(f'Validation accuracy = {val_acc:.4f}')

    # test on test set
    with torch.no_grad():
        model.eval()
        for test_inp, test_targ in test_loader:
            test_inp, test_targ = test_inp.to(config.DEVICE), test_targ.to(config.DEVICE)
            
            test_out = model(test_inp)
            pred_class = test_out.argmax(dim=1)
            # test_acc = (pred_class == test_targ).sum() / test_inp.shape[0]
            test_acc = balanced_accuracy_score(test_targ, pred_class)

    print(f'Test accuracy = {test_acc:.4f}')    