import argparse

def getConfig():
    parser = argparse.ArgumentParser(description="Train Model For Checking the Batch Normalization Parameters")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--root", default="/home/zyd/exp1/re_para/", help='root of the experiment')
    parser.add_argument("--data_path", default="./",
                        help='the path of the experiment data')
    parser.add_argument("--class_num", type=int, default=67)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--print_frequeny", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--img_pth", default="/home/zyd/exp1/re_para/img/")
    parser.add_argument("--exp_name", default="Exp")
    parser.add_argument("--pretrain", type=int, default=0)
    parser.add_argument("--moco", type=int, default=1)
    parser.add_argument("--moco_pth", default="/home/zyd/exp1/re_para/moco_v2_200ep_pretrain.pth.tar")
    parser.add_argument("--dataset",type=int, default=0)


    # parser.add_argument("--gpu", type=int, default=4)
    configs = parser.parse_args()
    return configs

