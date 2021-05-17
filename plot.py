import numpy as np

import matplotlib

from matplotlib import cm
from matplotlib.ticker import LinearLocator

import matplotlib.pyplot as plt


def plot_figures(filename, record, model):
    figures_name = []

    figures_name += plot_accuracies(filename, record)

    figures_name += plot_losses(filename, record)
    # WE PLOT THE CONV LAYER AND BN LAYER WEIGHT DIFFERENCE BASED ON DIFFERENT LAYER LEVEL
    # figures_name += plot_weight_difference(filename, model.weight_difference, model.layernamelist)
    # figures_name += plot_weight_Normalized_difference(filename, model.weight_normalized_dict, model.layernamelist)
    figures_name += plot_split(filename, model)

    figures_name += plot_vdf(filename, model)

    # plot_3d(filename, model.weight_normalized_dict)
    return figures_name


def plot_accuracies(filename, history):
    '''plot the Accuracy'''
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')


    figure_name = filename+"_acc.png"

    plt.savefig(figure_name, dpi = 300)
    plt.close()
    return [figure_name]


def plot_losses(filename, history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')

    figure_name = filename+"_los.png"
    plt.savefig(figure_name, dpi = 300)
    plt.close()
    return [figure_name]


def plot_split(filename, model):
    conv1_name = filename + "_conv"
    lyr1_name  = filename + "_lyr1"
    lyr2_name  = filename + "_lyr2"
    lyr3_name  = filename + "_lyr3"
    lyr4_name  = filename + "_lyr4"

    layernamelist0 = ["conv1.weight", "bn1.weight", "bn1.bias"]
    layernamelist1 = ["layer1.0.conv1.weight", "layer1.0.bn1.weight", "layer1.0.bn1.bias"]
    layernamelist2 = ["layer2.0.conv1.weight", "layer2.0.bn1.weight", "layer2.0.bn1.bias"]
    layernamelist3 = ["layer3.0.conv1.weight", "layer3.0.bn1.weight", "layer3.0.bn1.bias"]
    layernamelist4 = ["layer4.0.conv1.weight", "layer4.0.bn1.weight", "layer4.0.bn1.bias"]

    figures_name = []

    figures_name += plot_weight_difference(conv1_name, model.weight_difference, layernamelist0)
    figures_name += plot_weight_difference(lyr1_name, model.weight_difference, layernamelist1)
    figures_name += plot_weight_difference(lyr2_name, model.weight_difference, layernamelist2)
    figures_name += plot_weight_difference(lyr3_name, model.weight_difference, layernamelist3)
    figures_name += plot_weight_difference(lyr4_name, model.weight_difference, layernamelist4)

    figures_name += plot_weight_Normalized_difference(conv1_name, model.weight_normalized_dict, layernamelist0)
    figures_name += plot_weight_Normalized_difference(lyr1_name, model.weight_normalized_dict, layernamelist1)
    figures_name += plot_weight_Normalized_difference(lyr2_name, model.weight_normalized_dict, layernamelist2)
    figures_name += plot_weight_Normalized_difference(lyr3_name, model.weight_normalized_dict, layernamelist3)
    figures_name += plot_weight_Normalized_difference(lyr4_name, model.weight_normalized_dict, layernamelist4)

    return figures_name

def plot_vdf(filename, model):
    figure_name = filename+"_var.png"
    var_dict = {"conv":[], "bn_w":[], "bn_b":[]}
    for layer in model.layernamelist:
        if "conv1" in layer:
            var_dict["conv"].append(sum(model.weight_normalized_dict[layer]) / len(model.weight_normalized_dict[layer]))
        elif "bn1.weight" in layer:
            var_dict["bn_w"].append(sum(model.weight_normalized_dict[layer]) / len(model.weight_normalized_dict[layer]))
        elif "bn1.bias" in layer:
            var_dict["bn_b"].append(sum(model.weight_normalized_dict[layer]) / len(model.weight_normalized_dict[layer]))

    plt.plot(var_dict["conv"])
    plt.plot(var_dict["bn_w"])
    plt.plot(var_dict["bn_b"])

    plt.ylabel('difference')
    plt.legend(["conv", "bn.weight", "bn.bias"])
    plt.title("Variance of Weights in Different Layer")
    scale_ls = range(5)
    index_ls = ['Conv1', 'Layer1', 'Layer2', 'Layer3', 'Layer4']
    _ = plt.xticks(scale_ls,index_ls)

    plt.savefig(figure_name, dpi=300)
    plt.close()

    return [figure_name]

def plot_weight_difference(filename, weight_dict, namelist):
    for name in namelist:
        plt.plot(weight_dict[name])


    plt.xlabel('epoch')
    plt.ylabel('difference')
    plt.legend(namelist, prop={'size': 8})

    figure_name = filename+"_dif.png"
    plt.title('Weight '+filename[-4:]+' Difference in Various Layers')

    plt.savefig(figure_name, dpi = 300)
    plt.close()
    return [figure_name]

def plot_weight_Normalized_difference(filename, weight_dict, namelist):
    for name in namelist:
        plt.plot(weight_dict[name])
    plt.xlabel('epoch')
    plt.ylabel('difference')
    plt.ylim((0, 2))
    plt.legend(namelist, prop={'size': 8})
    plt.title('Normalized Weight '+filename[-4:]+' Difference in Various Layers')
    figure_name = filename+"_ndf.png"
    plt.savefig(figure_name, dpi = 300)
    plt.close()
    return [figure_name]

def plot_3d(filename, val_dict):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    keys_v = list(val_dict.keys())
    X = np.arange(0, len(keys_v), 1)
    Y = np.arange(0, len(val_dict[keys_v[0]]), 1)
    X, Y = np.meshgrid(X, Y)
    Z = []
    for key in keys_v:
        Z.append(val_dict[key])
    Z = np.array(Z)
    Z =Z.T
    print(np.shape(Z))
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
	                       linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))

    # A StrMethodFormatter is used automatically
    ax.set_zlim(0,8)

    ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf, shrink=0.5, aspect=5)


    ax.view_init(elev=38, azim=200)

    figure_name = filename+"_3d.png"
    plt.savefig(figure_name, dpi=300)
    plt.close()
    return [figure_name]
