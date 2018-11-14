import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
# from joblib import Parallel, delayed
from collections import Counter
import re
import csv
#from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import configparser


config=configparser.ConfigParser()
config.read('./config.ini')
GLOVE_PATH =config.get('DataPath','GLOVE_PATH')
PAD_TOKEN='<PAD>'
UNK_TOKEN='<UNK>'	
TR_RATIO=float(config.get('Parameters','TR_RATIO'))

class Tokenizer (object):
	"""
	class used to tokenize textual data
	"""
	def __init__(self,min_ct=10,tokenizer=str.split,vocab2idx=None,limit_length_trans=None,oov=False,unk='<UNK>'):
		self.min_ct=min_ct
		self.tokenizer=tokenizer
		self.counter=Counter()
		self.max_len=0
		self.vocab2idx=vocab2idx
		self.limit_length_trans=limit_length_trans
		self.oov=oov
		self.unk=unk
	def tokenize(self,sentences):
		"""
		sentences: list(str)
		"""
		return [self.tokenizer(sentence) for sentence in sentences]

	def fit(self,sentences):
		"""
		sentences: list(str)
		"""
		tokenized_senteces=self.tokenize(sentences) #list(list)
		for tokenized_text in tokenized_senteces:
			self.counter.update(set(tokenized_text))
			self.max_len=max(self.max_len,len(tokenized_text))
		if not self.vocab2idx:
			vocab=sorted([w for (w,c) in self.counter.items() if c>self.min_ct])
			self.vocab2idx={t:(i+1) for i,t in enumerate(vocab)}

	def sentence2idx(self,sentence):
		"""
		sentence: list[word]
		"""
		if self.oov:
			return [self.vocab2idx[w] if w in self.vocab2idx else self.vocab2idx[self.unk] for w in sentence ]

		else:
			return [self.vocab2idx[w] for w in sentence if w in self.vocab2idx ]

        
	def transform(self,sentences):
		"""
		senteces: list(str)
		"""
		n=len(sentences)
		if self.limit_length_trans:
			max_length=self.limit_length_trans
		else:
			max_length=self.max_len
	
		results=np.zeros([n,max_length])

		tokenized_sentences=self.tokenize(sentences)
		for i, sentence in enumerate(tokenized_sentences):
			sent2idx=self.sentence2idx(sentence[:max_length])
			results[i,:len(sent2idx)]=sent2idx
		return results
	

class TargetEncoder(object):
	def __init__(self,tr_df,te_df,smoothing,min_samples_leaf,noise_level,target='price_log'):
		self.tr_df=tr_df
		self.te_df=te_df
		self.smoothing=smoothing
		self.min_samples_leaf=min_samples_leaf
		self.noise_level=noise_level
		self.target=target
	def add_noise(self,series):
		return series * (1 + self.noise_level * np.random.randn(len(series)))
	def encode1col(self,col):
		tr_series=self.tr_df[col]
		te_series=self.te_df[col]
		target_series=self.tr_df[self.target]
		temp = pd.concat([tr_series, target_series], axis=1)
		averages = temp.groupby([col]).agg({self.target:["mean", "count"]})
		averages.columns=['mean','count']

		# Compute smoothing
		smoothing = 1 / (1 + np.exp(-(averages["count"] - self.min_samples_leaf) / self.smoothing))
		# Apply average function to all target data
		prior = target_series.mean()
		# The bigger the count the less full_avg is taken into account
		averages['encode'] = prior * (1 - smoothing) + averages["mean"] * smoothing
		averages.drop(["mean", "count"], axis=1, inplace=True)
		# Apply averages to trn and tst series
		tr_col_encode=pd.merge(self.tr_df,averages.reset_index(),how='left',on=[col])['encode'].rename(col+"_encode").fillna(prior)
		te_col_encode=pd.merge(self.te_df,averages.reset_index(),how='left',on=[col])['encode'].rename(col+"_encode").fillna(prior)

		return np.concatenate( (np.array(self.add_noise(tr_col_encode)),np.array(self.add_noise(te_col_encode))),axis=0).reshape([-1,1])


def paths(tokens):
	all_paths=['/'.join(tokens[:i+1]) for i in range(len(tokens)) ]
	return ' '.join(all_paths)

def to1hot(cat_array,cat_num):
	"""
	cat_array:[N,]
	cat_num: nat
	"""
	onehot_mat=np.eye(cat_num)
	res=np.zeros([cat_array.shape[0],cat_num])
	for i,c in enumerate(cat_array):
		res[i,:]=onehot_mat[c,:]
	return res

def main():
	print ('loading data...')
	train_df=pd.read_csv('../data/train.tsv', sep='\t')

	# shuffle 
	print('shuffle train df...')
	train_df = train_df.reindex(np.random.permutation(train_df.index)).reset_index(drop=True)
	# fill nan
	print ('fillna..')
	train_df.category_name.fillna('unk_cat',inplace=True)
	train_df.brand_name.fillna('unk_brand',inplace=True)
	train_df.item_description.fillna('unk_desc',inplace=True)

	print ('guessing missing brand...')
	valid_brands=set(train_df[train_df['brand_name']!='unk_brand'].brand_name)

	train_df_unkbrand=train_df[train_df['brand_name']=='unk_brand'][['name','item_description']].astype('str').values

	def guessing_brand(name,item_description):
		for brand in valid_brands:
			if len(brand)>=3 and brand in name or brand in item_description:
				return brand
		return 'unk_brand'

	train_df_unkbrand_guessing=[]
	for name,desc in train_df_unkbrand:
		train_df_unkbrand_guessing.append(guessing_brand(name,desc))
	# train_df_unkbrand_guessing=[guessing_brand(name,desc) for name , desc in train_df_unkbrand]
	train_df.loc[train_df['brand_name']=='unk_brand','brand_name']=train_df_unkbrand_guessing

	foundlen=len([b for b in train_df_unkbrand_guessing if b!='unk_brand'])
	unfoundlen=len(train_df_unkbrand_guessing)-foundlen
	print ('foundlen:{},unfoundlen:{}'.format(foundlen,unfoundlen))
	print ('processing category_name...')

	whitespace_regex = re.compile(r'\s+')
	def cat_process(cat):
	    cat = str(cat).lower()
	    cat = whitespace_regex.sub('', cat)
	    split = cat.split('/')
	    return paths(split)

	# A/B/C -> A A/B A/B/C
	train_df['category_name_cum'] = train_df.category_name.apply(cat_process)
	cat_tokenizer=Tokenizer(min_ct=50)
	cat_tokenizer.fit(train_df['category_name_cum'])

	# cat_mat and cat_num will be used when training model
	cat_mat=cat_tokenizer.transform(train_df['category_name_cum'])
	cat_num=len(cat_tokenizer.vocab2idx)+1

	print ('processing name & description ...')

	print('Loading GloVE...')
	# name & description are textual data which can use pre-trained w2v to initialize
	embeddings_df = pd.read_table(GLOVE_PATH, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

	word_embeddings_matrix = embeddings_df.values.astype(np.float32)
	print('GloVE Word embeddings shape:', word_embeddings_matrix.shape)
	word_embedding_vocab = {t: i for (i, t) in enumerate(embeddings_df.index.tolist())}

	del(embeddings_df)

	# tokenizer will be used to process description and name textual data
	regex_tokenizer = RegexpTokenizer(r'[a-z][\w&]*|[\d]+[\.]*[\w]*|[/!?*:%$"\'\-\+=\.,](?![/!?*:%$"\'-\+=\.,])')
	def regex_tokenizer_nltk(text):
		return regex_tokenizer.tokenize(text.lower())
	name_desc_tokenizer=Tokenizer(min_ct=30,tokenizer=regex_tokenizer_nltk)
	name_desc_tokenizer.fit(train_df['name']+' '+train_df['item_description'])

	# get words in name & description  and also available in pre-trained w2v
	words_with_embd=set([w for w in name_desc_tokenizer.vocab2idx.keys() if w in word_embedding_vocab.keys()])
	# get words in name & description  but not  available in pre-trained w2v
	words_without_embd=set([w for w in name_desc_tokenizer.vocab2idx.keys() if w not in word_embedding_vocab.keys()])
	print ('words with embedding num:{}, words without embedding num:{}'.format(len(words_with_embd),len(words_without_embd)))

	# construct a word 2 idx for all words (pad + unk + word in w2v + word not in w2v)
	all_vocab={w:i for i,w in enumerate([PAD_TOKEN,UNK_TOKEN]+sorted(words_without_embd)+sorted(words_with_embd))}
	# inverse word 2 idx dict
	all_vocab_idx2w={all_vocab[w]:w for w in all_vocab.keys() }

	# generate w2v initialization matrix
	np.random.seed(10)
	max_abs_embedding_random_value = np.sqrt(2 / word_embeddings_matrix.shape[1])
	glove_scaling_factor = word_embeddings_matrix.max() / max_abs_embedding_random_value
	word_without_embd_size=2+len(words_without_embd)
	word_with_embd_size=len(words_with_embd)
	total_vocab_size=len(all_vocab)
	# word in pretrained w2v can use it to initialize
	word_with_embd_mat=word_embeddings_matrix[
		[word_embedding_vocab[all_vocab_idx2w[idx]] for idx in range(word_without_embd_size,total_vocab_size) ],
		:]/ glove_scaling_factor
	# word not in pretrained w2v use Normal Distribution to initialize
	word_without_embd_mat=np.random.normal(word_with_embd_mat.mean(),word_with_embd_mat.std(),[word_without_embd_size,word_with_embd_mat.shape[1]])

	# get the  word embedding initialize matrix  for all words
	word_embedding_initial=np.vstack((word_without_embd_mat,word_with_embd_mat))
	name_tokenizer=Tokenizer(min_ct=0,tokenizer=regex_tokenizer_nltk,vocab2idx=all_vocab,oov=True,limit_length_trans=20)
	name_tokenizer.fit(train_df['name'])

	# name_mat will be used in training model
	name_mat=name_tokenizer.transform(train_df['name'])

	description_tokenizer=Tokenizer(min_ct=0,tokenizer=regex_tokenizer_nltk,vocab2idx=all_vocab,oov=True,limit_length_trans=70)
	description_tokenizer.fit(train_df['item_description'])

	# description_mat will be used in training model
	description_mat=description_tokenizer.transform(train_df['item_description'])

	print ('processing brand ...')
	train_df['brand_name']=train_df['brand_name'].apply(str.lower)
	brand_df=train_df['brand_name']

	brand_ct=Counter(brand_df[brand_df!='unk_brand'])
	valid_brands=[b for b,c in brand_ct.items() if c>=20]
	brand2idx={b:i+1 for i,b in enumerate(valid_brands)}
	# unknow brand will be represented as 0
	# brand_mat and brand_num will be used in training model
	brand_mat=brand_df.apply(lambda x: brand2idx.get(x,0))
	brand_num=len(brand2idx)+1

	print ('processing item_condition & shipping ...')
	# condition is form 1-5, so minus 1 in order to start from 0
	item_condition_mat=np.array(train_df['item_condition_id'])-1
	shipping_mat=np.array(train_df['shipping'])

	# shipping_mat and item_condition_1hot_mat will be used in training model
	# use one hot as input to model
	item_condition_1hot_mat=to1hot(item_condition_mat,len(set(item_condition_mat)))
	shipping_mat=np.reshape(shipping_mat,[shipping_mat.shape[0],1])

	print ('convert price to log(price)..')
	
	# we need to split train and val set in order to avoid data leak
	tr_ratio=TR_RATIO
	tr_len=int(len(train_df)*tr_ratio)
	price_log=np.log1p(train_df[:tr_len]['price'])
	price_mean,price_std=price_log.mean(),price_log.std()
	#normalize
	price_log=(price_log-price_mean)/price_std

	y_train=np.array(price_log) #shape (n,)
	# y_train y_val will be used in training model
	y_train=np.reshape(y_train,[y_train.shape[0],1])
	y_val=np.array(train_df[tr_len:]['price']).reshape([-1,1]) # y_val is original price (no log)

	train_df['price_log']=0
	train_df.loc[:tr_len-1,'price_log']=price_log # only train data is set to price_log value, 
												# val data is set to 0.

	print ('categorical features encoding...')											
	target_encoder=TargetEncoder(train_df[:tr_len],train_df[tr_len:],10,15,0.01)
	cat_encode_mat=target_encoder.encode1col('category_name_cum')
	brand_encode_mat=target_encoder.encode1col('brand_name')
	cond_encode_mat=target_encoder.encode1col('item_condition_id')
	encode_mat=np.concatenate((brand_encode_mat,cat_encode_mat,cond_encode_mat),axis=1)



	print ('processing some statistical features...')

	
	cats_stats_df = train_df[:tr_len].groupby(['category_name', 'brand_name', 'shipping']).agg({'category_name': len,
													'price_log': [np.median, np.mean, lambda x: np.std(x, ddof=0)]})

	category_stats_df= train_df[:tr_len].groupby(['category_name']).agg({'category_name':len,'price_log':[np.median,np.mean,lambda x: np.std(x, ddof=0)]})
	brand_stats_df=train_df[:tr_len].groupby(['brand_name']).agg({'brand_name':len,'price_log':[np.median,np.mean,lambda x: np.std(x, ddof=0)]})

	cats_stats_df.columns = ["count",'price_log_median', 'price_log_mean', 'price_log_std' ]
	category_stats_df.columns=['category_count','category_plog_median','category_plog_mean','category_plog_std']
	brand_stats_df.columns=['brand_count','brand_plog_median','brand_plog_mean','brand_plog_std']

	CAT_STATS_MIN_COUNT=5
	CATEGORY_STATS_MIN_COUNT=5
	BRAND_STATS_MIN_COUNT=20
	cats_stats_df=cats_stats_df[cats_stats_df['count']>=CAT_STATS_MIN_COUNT]
	category_stats_df=category_stats_df[category_stats_df['category_count']>=CATEGORY_STATS_MIN_COUNT]
	brand_stats_df=brand_stats_df[brand_stats_df['brand_count']>=BRAND_STATS_MIN_COUNT]

	cats_stats_df['count']=np.log1p(cats_stats_df['count'])
	category_stats_df['category_count']=np.log1p(category_stats_df['category_count'])
	brand_stats_df['brand_count']=np.log1p(brand_stats_df['brand_count'])
	cats_stats_df['price_std_div_mean']=cats_stats_df['price_log_std']/cats_stats_df['price_log_mean']
	category_stats_df['category_std_div_mean']=category_stats_df['category_plog_std']/category_stats_df['category_plog_mean']
	brand_stats_df['brand_std_div_mean']=brand_stats_df['brand_plog_std']/brand_stats_df['brand_plog_mean']

	STD_SIGMAS=2
	cats_stats_df['min_expected_log_price'] = (cats_stats_df['price_log_mean'] - cats_stats_df['price_log_std']*STD_SIGMAS)
	cats_stats_df['max_expected_log_price'] = (cats_stats_df['price_log_mean'] + cats_stats_df['price_log_std']*STD_SIGMAS)

	category_stats_df['category_min_expected_log_price'] = (category_stats_df['category_plog_mean'] - category_stats_df['category_plog_std']*STD_SIGMAS)
	category_stats_df['category_max_expected_log_price'] = (category_stats_df['category_plog_mean'] + category_stats_df['category_plog_std']*STD_SIGMAS)


	brand_stats_df['brand_min_expected_log_price'] = (brand_stats_df['brand_plog_mean'] - brand_stats_df['brand_plog_std']*STD_SIGMAS)
	brand_stats_df['brand_max_expected_log_price'] = (brand_stats_df['brand_plog_mean'] + brand_stats_df['brand_plog_std']*STD_SIGMAS)

	cats_stats_df=cats_stats_df.reset_index()
	category_stats_df=category_stats_df.reset_index()
	brand_stats_df=brand_stats_df.reset_index()

	train_df=pd.merge(train_df,cats_stats_df,how='left',on=['category_name','brand_name','shipping'])
	train_df=pd.merge(train_df,category_stats_df,how='left',on=['category_name'])
	train_df=pd.merge(train_df,brand_stats_df,how='left',on=['brand_name'])
	cats_stats_mat=train_df[['count','price_log_median','price_log_mean',\
							'price_log_std','price_std_div_mean',\
							'price_std_div_mean','min_expected_log_price',\
							'max_expected_log_price',\
							'category_count','category_plog_median','category_plog_mean','category_plog_std',\
							'category_std_div_mean','category_min_expected_log_price','category_max_expected_log_price',\
							 'brand_count','brand_plog_median','brand_plog_mean','brand_plog_std',\
							'brand_std_div_mean','brand_min_expected_log_price','brand_max_expected_log_price']].fillna(0)

	cats_stats_mat=np.array(cats_stats_mat)
	cats_stats_scaler = StandardScaler()
	cats_stats_scaler.fit(cats_stats_mat)
	# cats_stats_scaled_mat will be used in training model
	cats_stats_scaled_mat=cats_stats_scaler.transform(cats_stats_mat)

	cats_stats_scaled_mat=np.concatenate((cats_stats_scaled_mat,encode_mat),axis=1)


	print ('saving data...')

	np.save(config.get('DataPath','cat_name_mat_path'),cat_mat)
	np.save(config.get('DataPath','name_mat_path'),name_mat)
	np.save(config.get('DataPath','description_mat_path'),description_mat)
	np.save(config.get('DataPath','brand_mat_path'),brand_mat)
	np.save(config.get('DataPath','cond_mat_path'),item_condition_1hot_mat)
	np.save(config.get('DataPath','ship_mat_path'),shipping_mat)
	np.save(config.get('DataPath','y_train_path'),y_train)
	np.save(config.get('DataPath','y_val_path'),y_val)
	np.save(config.get('DataPath','stats_features_path'),cats_stats_scaled_mat)
	np.save(config.get('DataPath','word_embd_init_path'),word_embedding_initial)
	# save cat_num  and brand_num into config
	config.set('Parameters','CAT_NUM',str(cat_num))
	config.set('Parameters','BRAND_NUM',str(brand_num))
	config.set('Parameters','TR_LEN',str(tr_len))
	config.set('Parameters','price_std',str(price_std))
	config.set('Parameters','price_mean',str(price_mean))


	with open('config.ini','w') as handel:
		config.write(handel)

if __name__ == '__main__':
	main()









