import os
from time import strftime, gmtime

exp_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

os.system("pwd")
# ifconfig en0
# ip_address 	= "10.30.107.55"
ip_address 	= "10.30.44.112"
zyd_address = "zhangyidan@" + ip_address
rs_address	= "/Users/zhangyidan/Researching/cuhksz/CV/BN_para/result/May16_Mor/"
sv_address 	=  zyd_address+ ":"+rs_address
psw 		= "WZXallen1997"

# build the experiment result file 
def mkdir_result_script(dir_name):
	# dir_name is the name of the target
	file = open("mkdir_r.sh", "w")
	file.write("#!/usr/bin/expect\n")
	file.write("set timeout 10\n")
	file.write("spawn ssh "+zyd_address+"\n")
	file.write("expect \"word:\"" + "\n")
	file.write("send \""+psw+"\\r\"" + "\n")
	file.write("expect \"~\"" +"\n")
	file.write("send \"mkdir "+rs_address +dir_name+"\\r\"\n") 
	file.write("expect \"~\"" +"\n")
	file.write("send \"exit\\r\"" + "\n")
	file.write("interact" + "\n")
	file.close()
	return 0

def send_figure_script(fg_inf, fg_name_list, dir_adr):
	# fg_inf is the address of the figure not include name 
	# fg_name_list include the name of the figure
	# dir_adr is the address of the director address the lase should be /
	file = open("sd_img.sh", "w")
	file.write("#!/usr/bin/expect\n")
	file.write("set timeout 20\n")
	for name in fg_name_list:
		fg_adr = name 
		file.write("spawn scp -r "+ fg_adr +" "+dir_adr+"\n")
		file.write("expect \"word:\"" + "\n")
		file.write("send \""+psw+"\\r\"" + "\n")
		file.write("interact" + "\n")
		file.write("\n")
	file.close()
	return 0


def get_namelist(exp_name):
	filename 	= "fig_name.csv"
	file 		= open(filename, "r")	
	namelist 	= file.readline().split(",")
	file.close()
	return namelist

def exp(dataset = 0 , pretrain=0, exp_name = "Exp",  epoch = 50, moco=0, seed = 42):
	# Data Set
	# 0: indoor;
	# 1 : cifar100;
	# 2 : cifar10

	print(exp_name + " という名の実験が行われる" + " SEED: "+str(seed)+" EPOCH "+ str(epoch) + " MoCo "+str(moco))
	exp_name = exp_name+"_pt-"+ str(pretrain)+"_ep-"+str(epoch)+"_mc-"+str(moco)+"_ds-"+str(dataset)

	# exec file
	if dataset == 0 :
		if moco:
			os.system("python3.7 train.py --seed " + str(seed) + " --dataset " + str(dataset) + " --exp_name " + exp_name+ " --pretrain " + str(pretrain) + " --epochs " + str(epoch) + " --moco " + str(moco)+ " --class_num " + str(67))
		else:
			os.system("python3.7 train.py --seed "+str(seed)+ " --dataset " + str(dataset) + " --exp_name "+ exp_name +" --pretrain "+ str(pretrain)+" --epochs " +str(epoch)+ " --moco " + str(moco)+ " --class_num " + str(67))
	elif dataset == 1 :
		if moco:
			os.system("python3.7 train.py --seed " + str(seed) + " --dataset " + str(dataset) + " --exp_name " + exp_name+ " --pretrain " + str(pretrain) + " --epochs " + str(epoch) + " --moco " + str(moco)+ " --class_num " + str(100))
		else:
			os.system("python3.7 train.py --seed "+str(seed)+ " --dataset " + str(dataset) + " --exp_name "+exp_name +" --pretrain "+ str(pretrain)+" --epochs " +str(epoch)+ " --moco " + str(moco)+ " --class_num " + str(100))
	else:
		if moco:
			os.system("python3.7 train.py --seed " + str(seed) + " --dataset " + str(dataset) + " --exp_name " + exp_name+ " --pretrain " + str(pretrain) + " --epochs " + str(epoch) + " --moco " + str(moco)+ " --class_num " + str(10))
		else:
			os.system("python3.7 train.py --seed "+str(seed)+ " --dataset " + str(dataset) + " --exp_name "+exp_name+" --pretrain "+ str(pretrain)+" --epochs " +str(epoch)+ " --moco " + str(moco)+ " --class_num " + str(10))
	# make director
	print("リモートで宛先アドレスを作成する")
	mkdir_result_script(exp_name)
	os.system("expect mkdir_r.sh")
	# send figure 
	print("画像のアップロード")
	namelist = get_namelist(exp_name)

	send_figure_script("/home/zyd/exp1/re_para/img/", namelist, sv_address+exp_name+"/")
	os.system("expect sd_img.sh")
	return 0
	# pooL

def main():
	# exp(dataset= 0, exp_name="MMTMoco0.7", pretrain=0, epoch=32, moco=1)
	# exp(dataset=0, exp_name="MMT", pretrain=0, epoch=16, moco=1)
	# exp(dataset= 0, exp_name="MMT", pretrain=0, epoch=16, moco=0)

	exp(dataset= 1, exp_name="frcifar", pretrain=1, epoch=16, moco=0)
	exp(dataset= 1, exp_name="frcifar", pretrain=0, epoch=16, moco=1)
	# exp(dataset= 1, exp_name="MMT_Cifar", pretrain=0, epoch=16, moco=0)

	# exp(dataset= 1, exp_name="Cifar100baselineP", pretrain=1, epoch=16, moco=0)
	# exp(dataset= 1, exp_name="Cifar100baselineM", pretrain=0, epoch=16, moco=1)
	# exp(dataset= 1, exp_name="MocoFreeBS", pretrain=0, epoch=16, moco=1)
	# exp(dataset= 0, exp_name="MocoFreeBS", pretrain=0, epoch=16, moco=1)
	# exp(dataset= 1, exp_name="Cifar100baselineS", pretrain=0, epoch=16, moco=0)

	# exp(dataset= 1, exp_name="", pretrain=1, epoch=16, moco=0)
	# exp(dataset= 1, exp_name="", pretrain=1, epoch=32, moco=0)
	# exp(dataset= 1, exp_name="", pretrain=0, epoch=16, moco=1)
	# exp(dataset= 1, exp_name="", pretrain=0, epoch=32, moco=1)
	# exp(dataset= 1, exp_name="", pretrain=0, epoch=16, moco=0)
	# exp(dataset= 1, exp_name="", pretrain=0, epoch=32, moco=0)

if __name__ == '__main__':
	main()