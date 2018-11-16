import network
import minibatcher
import numpy as np
import configparser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

config=configparser.ConfigParser()
config.read('./config.ini')

cat_mat=np.load(config.get('DataPath','cat_name_mat_path'))
name_mat=np.load(config.get('DataPath','name_mat_path'))
description_mat=np.load(config.get('DataPath','description_mat_path'))
brand_mat=np.load(config.get('DataPath','brand_mat_path'))
item_condition_1hot_mat=np.load(config.get('DataPath','cond_mat_path'))
shipping_mat=np.load(config.get('DataPath','ship_mat_path'))
y_tr=np.load(config.get('DataPath','y_train_path'))
y_val=np.load(config.get('DataPath','y_val_path'))
cats_stats_scaled_mat=np.load(config.get('DataPath','stats_features_path'))
word_embedding_initial=np.load(config.get('DataPath','word_embd_init_path'))

cat_num=int(config.get('Parameters','cat_num'))
brand_num=int(config.get('Parameters','brand_num'))
tr_len=int(config.get('Parameters','tr_len'))
price_std=float(config.get('Parameters','price_std'))
price_mean=float(config.get('Parameters','price_mean'))

use_drop=config.getboolean('Parameters','use_drop')
use_skip=config.getboolean('Parameters','use_skip')
EPOCH=int(config.get('Parameters','epoch_num'))
learning_rate=float(config.get('Parameters','lr'))
BATCHSIZE=int(int(config.get('Parameters','batch_size')))
minibatcher=minibatcher.MiniBatcher(BATCHSIZE,tr_len)


use_cuda = torch.cuda.is_available()
if use_cuda:
	print ('using gpu...')
else:
	print ('using cpu...')
device = torch.device("cuda" if use_cuda else "cpu")



def dataGenerator(idxs,cat_mat,name_mat,description_mat,\
                  brand_mat,item_condition_1hot_mat,\
                 shipping_mat, cats_stats_scaled_mat,\
                 y_train):
    """
    idxs: np.array [n,]
    """
    cat=torch.from_numpy(cat_mat[idxs]).long()
    name=torch.from_numpy(name_mat[idxs]).long()
    desc=torch.from_numpy(description_mat[idxs]).long()
    brand=torch.from_numpy(brand_mat[idxs]).long()
    cond=torch.from_numpy(item_condition_1hot_mat[idxs].astype('float32'))
    ship=torch.from_numpy(shipping_mat[idxs].astype('float32'))
    stats=torch.from_numpy(cats_stats_scaled_mat[idxs].astype('float32'))
    y=torch.from_numpy(y_train[idxs].astype('float32'))
           
    return  cat,name,desc,brand,cond,ship,stats,y
    

# y_tr is centering scaled price_log
cat_tr,name_tr,desc_tr,brand_tr,cond_tr,ship_tr,stats_tr=cat_mat[:tr_len],name_mat[:tr_len],description_mat[:tr_len],brand_mat[:tr_len],item_condition_1hot_mat[:tr_len],shipping_mat[:tr_len], cats_stats_scaled_mat[:tr_len]  

cat_val,name_val,desc_val,brand_val,cond_val,ship_val,stats_val=cat_mat[tr_len:],name_mat[tr_len:],description_mat[tr_len:],brand_mat[tr_len:],item_condition_1hot_mat[tr_len:],shipping_mat[tr_len:], cats_stats_scaled_mat[tr_len:]
# y_val is original price
def eval_val():
	def rmsle(y, y0):
		assert len(y) == len(y0)
		return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))
	print('evaluating validation set...')
	val_idxs=np.array(range(cat_val.shape[0]))
	val_minibatchsize=1000
	val_start_idx=0
	terminate=False
	val_outs=[]
	val_ys=[]
	step=0
	while True:
		#print ('step:{}/{}'.format(step,cat_val.shape[0]//val_minibatchsize))
		if val_start_idx+val_minibatchsize<cat_val.shape[0]:
			val_end_idx=val_start_idx+val_minibatchsize
		else:
			val_end_idx=cat_val.shape[0]
			terminate=True


		subvalidxs=np.array(range(val_start_idx,val_end_idx))

		val_cat_mat,val_name_mat,val_description_mat,val_brand_mat,val_item_condition_1hot_mat,val_shipping_mat,val_cats_stats_scaled_mat,val_y=dataGenerator(subvalidxs,cat_val,name_val,desc_val,brand_val,cond_val,ship_val,stats_val,y_val)      
		val_cat_mat,val_name_mat,val_description_mat,val_brand_mat,val_item_condition_1hot_mat,val_shipping_mat,val_cats_stats_scaled_mat,val_y=val_cat_mat.to(device),val_name_mat.to(device),val_description_mat.to(device),val_brand_mat.to(device),val_item_condition_1hot_mat.to(device),val_shipping_mat.to(device),val_cats_stats_scaled_mat.to(device),val_y.to(device)
		val_out=net.forward(val_name_mat,\
				val_description_mat,\
				val_cat_mat,\
				val_brand_mat,\
				val_item_condition_1hot_mat,\
				val_shipping_mat,\
				val_cats_stats_scaled_mat)
		val_outs.append(np.array(val_out.data))
		val_ys.append(np.array(val_y.data))
		if terminate:
			break
		val_start_idx=val_end_idx
		step+=1
	val_stack_outs=np.vstack(val_outs)
	val_stack_ys=np.vstack(val_ys)
	val_out_adjusted=np.expm1(val_stack_outs*price_std+price_mean)
	return rmsle(val_out_adjusted, val_stack_ys)

net=network.Network(word_embedding_initial,brand_num,32,\
			cat_num,32,\
			5,20,use_drop,use_skip).to(device)
optimizer=torch.optim.Adam(net.parameters(),learning_rate)
loss_func = nn.MSELoss()


for epoch in range(EPOCH):
	print ('*'*10,'epoch',str(epoch),'*'*10)
	step=0
	for idxs in minibatcher.get_one_batch():
		sample_cat_mat,sample_name_mat,sample_description_mat,sample_brand_mat,sample_item_condition_1hot_mat,sample_shipping_mat,sample_cats_stats_scaled_mat,sample_y_train=dataGenerator(idxs,cat_tr,name_tr,desc_tr,brand_tr,cond_tr,ship_tr,stats_tr,y_tr)      
		sample_cat_mat,sample_name_mat,sample_description_mat,sample_brand_mat,sample_item_condition_1hot_mat,sample_shipping_mat,sample_cats_stats_scaled_mat,sample_y_train=autograd.Variable(sample_cat_mat).to(device),autograd.Variable(sample_name_mat).to(device),autograd.Variable(sample_description_mat).to(device),autograd.Variable(sample_brand_mat).to(device),autograd.Variable(sample_item_condition_1hot_mat).to(device),autograd.Variable(sample_shipping_mat).to(device),autograd.Variable(sample_cats_stats_scaled_mat).to(device),autograd.Variable(sample_y_train).to(device)
		out=net.forward(sample_name_mat,\
					sample_description_mat,\
				sample_cat_mat,\
				sample_brand_mat,\
				sample_item_condition_1hot_mat,\
				sample_shipping_mat,\
				sample_cats_stats_scaled_mat)
		loss=loss_func(out,sample_y_train)
		if step%100==0:
			print ('trianing loss:{} '.format(loss))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		step+=1
	net.eval()
	print ('valset rmsle:{}'.format(eval_val()))
	net.train()


      

