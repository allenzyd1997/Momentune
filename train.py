import os

import numpy as np
import torch
import torch.nn as nn
from getConfig import getConfig
from getDevice import DeviceDataLoader, to_device
from getData import getYourData, getIndoorData,getCifar10Data, getCifar100Data
from getModel import ResNet50_C
from plot import plot_figures

from time import strftime, gmtime
exp_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

# model_data is the folder for saving the model_dict
os.system("rm -r model_data")
os.system("mkdir model_data")


layernamelist = ["conv1.weight",
			"bn1.weight",
            "bn1.bias",
			"layer1.0.conv1.weight",
			"layer1.0.bn1.weight",
			"layer1.0.bn1.bias",
			"layer2.0.conv1.weight",
			"layer2.0.bn1.weight",
			"layer2.0.bn1.bias",
			"layer3.0.conv1.weight",
			"layer3.0.bn1.weight",
			"layer3.0.bn1.bias",
			"layer4.0.conv1.weight",
			"layer4.0.bn1.weight",
			"layer4.0.bn1.bias",
			"fc.5.weight",
			]


def getData(configs):
    return getYourData(configs)

def getModel(configs):
    return ResNet50_C(configs, layernamelist)

def setSeeds(seed):
    torch.backends.cudnn.benchmark = False
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # set the backedns in the cuda network to be a fixed value
    torch.backends.cudnn.deterministic = True
    return


def train(model, train_dl, val_dl, configs):
    # setting the optimizer
    optimizer = torch.optim.SGD(model.parameters(), configs.learning_rate)
    # record the value in every epoch
    record = []

    for epoch in range(configs.epochs):

        # if epoch == configs.epochs // 2:
            # model.unfreezeBN()
            # print("unfreeze BN layers")
        # open the training mode
        model.train()

        # save the training loss
        train_losses = []
        if epoch == configs.epochs // 2:
            optimizer = torch.optim.SGD(model.parameters(), configs.learning_rate * 0.9)
        if epoch == (configs.epochs // 4) * 3:
            optimizer = torch.optim.SGD(model.parameters(), configs.learning_rate * 0.9 * 0.9)

        # Train the model
        for batch in train_dl:
            loss = model.training_step(batch, epoch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # model.momenTuneBN()


        # Validate the model

        output = evaluate(model, val_dl, True)
        output["train_loss"] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, output)
        record.append(output)
    return record


def evaluate(model, val_dl, validate_or_not):
    # close training mode
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_dl]
    # calculate the loss and return the value
    return model.validation_epoch_end(outputs, validate_or_not)


def control_center():
    # get the model data and all the configurations

    configs = getConfig()
    model = getModel(configs)
    # set the GPU Num here
    # torch.cuda.set_device(configs.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_data, val_data, test_data = getData(configs)

    # set seeds based on config
    setSeeds(configs.seed)

    # put the network upon the CUDA
    model = to_device(model,device)
    train_data, val_data, test_data            = DeviceDataLoader(train_data, device), DeviceDataLoader(val_data, device),  DeviceDataLoader(test_data, device),
    record = train(model, train_data, val_data, configs)

    # check the parameters is freeze or not
    for name, parameters in model.named_parameters():
        print(name)
        print(parameters.requires_grad)

    test_acc = evaluate(model, test_data, validate_or_not=False)['test_acc']


    exp_name = configs.exp_name
    img_path = configs.img_pth + exp_name
    data_path = configs.root+"res/"+exp_name

    torch.save(model.state_dict(), data_path + ".pth")

    np.save(data_path+'_weight_dif.npy',model.weight_difference)
    np.save(data_path+'_weight_ndf.npy',model.weight_normalized_dict)

    tp = {}
    for th, k in enumerate(record):
        tp[str(th) + ' val_loss'] = k['val_loss']
        tp[str(th) + ' val_acc'] = k['val_acc']
    np.save(data_path+'_record.npy', tp)

    # data = np.load("data.npy", allow_pickle=True).item()

    file = open(data_path+"_test_acc.txt", "w")
    file.write("test_acc: ")
    file.write(str(test_acc))
    file.close()

    figures_name = plot_figures(img_path, record, model)

    file = open("fig_name.csv", "w")
    file.write(",".join(figures_name))
    file.write(","+data_path+"_test_acc.txt"+","+ data_path + ".pth"+ ","+ data_path+'_record.npy')
    file.write("," + data_path + "_weight_dif.npy" + "," + data_path + "_weight_ndf.npy")
    file.close()

    return

def main():
    control_center()
# test_data()
if __name__ == '__main__':
    main()

