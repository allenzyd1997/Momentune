import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImageClassificationBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_dict = {}
        self.weight_difference = {}
        self.weight_normalized_dict = {}
        self.layernamelist = []

    def training_step(self, batch, epoch_num):
        images, labels = batch

        out = self(images)  # Generate predictions

        loss = F.cross_entropy(out, labels)  # Calculate loss

        return loss

    def validation_step(self, batch):
        images, labels = batch

        out = self(images)  # Generate predictions

        loss = F.cross_entropy(out, labels)  # Calculate loss

        acc = self.accuracy(out, labels)  # Calculate accuracy

        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs, validate_or_not):
        batch_losses = [x['val_loss'] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses

        batch_accs = [x['val_acc'] for x in outputs]

        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()} if validate_or_not else {
            'test_loss': epoch_loss.item(), 'test_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        # save the weight_dict
        # name_tmp use to all be name
        if epoch == 0:
            for name, parameters in self.named_parameters():
                name_tmp = name[8:]
                if name_tmp in self.layernamelist:
                    self.weight_dict[name_tmp] 				= parameters.cpu().detach().numpy()
                    self.weight_difference[name_tmp] 		= [0]
                    self.weight_normalized_dict[name_tmp] 	= []
        else:
            for name, parameters in self.named_parameters():
                name_tmp = name[8:]
                if name_tmp in self.layernamelist:
                    self.weight_difference[name_tmp] 		+= [self.mean_squared_error(parameters.cpu().detach().numpy(), self.weight_dict[name_tmp])]
                    self.weight_normalized_dict[name_tmp]	+= [self.weight_difference[name_tmp][-1] / self.weight_difference[name_tmp][-2]] if self.weight_difference[name_tmp][-2] != 0 else [0.0]
                    self.weight_dict[name_tmp] 			    = parameters.cpu().detach().numpy()

        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(

            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

    def accuracy(self, outputs, labels):
        '''Giving a way to calculate the accuracy '''
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def mean_squared_error(self, y, t):
        '''Calculate the weight difference between two epoch'''
        return 0.5 * np.sum((y - t) ** 2)


def main():
    return 0
if __name__ == '__main__':
    main()
