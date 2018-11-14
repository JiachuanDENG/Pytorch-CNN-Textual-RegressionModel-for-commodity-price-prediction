import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
class Network(nn.Module):
    def __init__(self,word_embedding_initial,\
                 brand_num,brand_embdDim,\
                 cat_num,cat_embdDim,\
                name_avg_num, desc_avg_num):
        super(Network,self).__init__()
        self.name_avg_num,self.desc_avg_num=name_avg_num,desc_avg_num
        # self.we : word embedding for name & description
        self.we=nn.Embedding(word_embedding_initial.shape[0],word_embedding_initial.shape[1])
        # initialize word embedding
        self.we.weight = nn.Parameter(torch.from_numpy(word_embedding_initial.astype('float32')))
        
        self.brand_emb=nn.Embedding(brand_num,brand_embdDim)
        self.cat_emb=nn.Embedding(cat_num,cat_embdDim)
        
        self.convName1=nn.Conv2d(1,128,(2,word_embedding_initial.shape[1]))
        self.convName2=nn.Conv2d(1,128,(3,word_embedding_initial.shape[1]))
        self.convName3=nn.Conv2d(1,128,(4,word_embedding_initial.shape[1]))
        self.maxpool1dName1=nn.MaxPool1d(19)
        self.maxpool1dName2=nn.MaxPool1d(18)
        self.maxpool1dName3=nn.MaxPool1d(17)
        
        self.convDesc1=nn.Conv2d(1,96,(2,word_embedding_initial.shape[1]))
        self.convDesc2=nn.Conv2d(1,96,(3,word_embedding_initial.shape[1]))
        self.convDesc3=nn.Conv2d(1,96,(4,word_embedding_initial.shape[1]))
        self.maxpool1dDesc1=nn.MaxPool1d(69)
        self.maxpool1dDesc2=nn.MaxPool1d(68)
        self.maxpool1dDesc3=nn.MaxPool1d(67)
        
        self.bn1=nn.BatchNorm1d(995)
        self.FC1=nn.Linear(995,256)
        self.bn2=nn.BatchNorm1d(256)
        self.drop1=nn.Dropout(0.5)
        self.FC2=nn.Linear(256,128)
        
        self.bn3=nn.BatchNorm1d(156)
        self.drop2=nn.Dropout(0.5)
        self.FC3=nn.Linear(156,1)
    
        
    def forward(self,name,desc,cat,brand,cond,ship,stats_features):
        """
        name: [N,max_name_size] long
        desc: [N,max_desc_size] long
        cat: [N,max_cat_size] long
        brand: [N,1] long
        cond: [N,total_cond_num] (onehot) float
        ship: [N,1] float
        stats_features :[N,m] float
        """

        name=self.we(name) #[N,max_name_size,we.shape[1]] 
        desc=self.we(desc) #[N,max_desc_size,we.shape[1]]
        name_avg=torch.mean(name[:,:self.name_avg_num,:],dim=1) #[N,we.shape[1]]
        desc_avg=torch.mean(desc[:,:self.desc_avg_num,:],dim=1) #[N,we.shape[1]]
        
        unsqz_name=torch.unsqueeze(name,dim=1)
        name_conv1=self.convName1(unsqz_name) #[N,128,H1,1]
        name_conv1=F.relu(name_conv1.squeeze(3)) #[N,128,H1,1]
        name_conv2=self.convName2(unsqz_name) #[N,128,H1,1]
        name_conv2=F.relu(name_conv2.squeeze(3)) #[N,128,H1,1]
        name_conv3=self.convName3(unsqz_name) #[N,128,H1,1]
        name_conv3=F.relu(name_conv3.squeeze(3)) #[N,128,H1,1]
        
        unsqz_desc=torch.unsqueeze(desc,dim=1)
        desc_conv1=self.convDesc1(unsqz_desc) # [N,96,H2,1]
        desc_conv1=F.relu(desc_conv1.squeeze(3)) #[N,96,H1,1]
        desc_conv2=self.convDesc2(unsqz_desc) # [N,96,H2,1]
        desc_conv2=F.relu(desc_conv2.squeeze(3)) #[N,96,H1,1]
        desc_conv3=self.convDesc3(unsqz_desc) # [N,96,H2,1]
        desc_conv3=F.relu(desc_conv3.squeeze(3)) #[N,96,H1,1]
        
       
        name_conv1=self.maxpool1dName1(name_conv1).squeeze(2)#[N,128,1] ->[N,128]
        name_conv2=self.maxpool1dName2(name_conv2).squeeze(2)#[N,128,1] ->[N,128]
        name_conv3=self.maxpool1dName3(name_conv3).squeeze(2)#[N,128,1] ->[N,128]
        
        name_conv=torch.cat((name_conv1,name_conv2,name_conv3),1) #[N,128*3]
        
        desc_conv1=self.maxpool1dDesc1(desc_conv1).squeeze(2)# [N,96,1] -> [N,96]
        desc_conv2=self.maxpool1dDesc2(desc_conv2).squeeze(2)# [N,96,1] -> [N,96]
        desc_conv3=self.maxpool1dDesc3(desc_conv3).squeeze(2)# [N,96,1] -> [N,96]
        
        desc_conv=torch.cat((desc_conv1,desc_conv2,desc_conv3),1) #[N,96*3]
        
        cat=self.cat_emb(cat).view(cat.size(0),-1) #[N,max_cat_size,cat_embdDim]->[N,max_cat_size*cat_embdDim]
        brand=self.brand_emb(brand).view(brand.size(0),-1) #[N,1,brand_embdDim]->[N,1*brand_embdDim]
       
        #-> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        x_cat=torch.cat((name_conv,desc_conv,name_avg,desc_avg,cat,brand,cond,ship,stats_features),1)
#         print (x_cat.size())
        x_cat=self.bn1(x_cat)
        x_cat=self.FC1(x_cat)
        x_cat=F.relu(self.bn2(x_cat))
  
        x_cat=self.drop1(x_cat)
            
        x_cat=self.FC2(x_cat)
        x_cat=F.relu(x_cat)
        

        x_cat=self.drop2(x_cat)
        #skip
        x_cat=torch.cat((x_cat,cond,ship,stats_features),dim=1)
        x_cat=self.bn3(x_cat)
        out=self.FC3(x_cat)
        return out
        
        
        