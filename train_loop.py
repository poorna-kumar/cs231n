import copy
import time

import torch
import numpy as np

from eval_utils.metrics import avg_auc_macro
from general_utils import convert_to_np

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Train loop: will use device {device}")


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    auc_history = {'train': [], 'val': []}
    loss_history = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_auc = 0.0

    for epoch in range(num_epochs):
        start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            y_preds = np.array([])
            y_trues = np.array([])

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    y_preds = np.append(y_preds, convert_to_np(torch.sigmoid(outputs), device))
                    y_trues = np.append(y_trues, convert_to_np(labels, device))

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            loss_history[phase].append(epoch_loss)
            epoch_auc = avg_auc_macro(y_trues, y_preds)
            auc_history[phase].append(epoch_auc)
            print('{} Loss: {:.4f}; Average AUC: {:.4f}'.format(phase, epoch_loss, epoch_auc))

            # deep copy the model
            if phase == 'val' and epoch_auc > best_val_auc:
                best_val_auc = epoch_auc
                best_model_wts = copy.deepcopy(model.state_dict())

        print(f"Epoch done. Took {time.time()-start} seconds")
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(f'Best val AUc: {best_val_auc:.4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, auc_history, loss_history


def _vstack_wrapper(first, second):
    if first is None:
        return second
    return np.vstack([first, second])


def get_predictions(model, dataloader):
    outputs_list = None
    labels_list = None
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = torch.sigmoid(model(inputs))
        outputs_list = _vstack_wrapper(outputs_list, convert_to_np(outputs, device))
        labels_list = _vstack_wrapper(labels_list, convert_to_np(labels, device))

    return outputs_list, labels_list
