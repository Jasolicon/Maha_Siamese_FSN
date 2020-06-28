import os
import torch
import time
PATH={'omniglot':'E:/DataSets/omniglot/',
      'miniimagenet':'E:/DataSets/miniimagenet/',
      'tieredimagenet':'E:/DataSets/tieredimagenet/'}
ROOT=os.getcwd()
def train_by_GPU(gpu = '0'):
      # gpu = "0"  # which GPU to use
      os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
      os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
      # device = 'cuda' if torch.cuda.is_available() else 'cpu'
      device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

      return device
start_epoch = 0
epoch_num = 100
classes_num=100
meta_train_num=20
meta_test_num=10
feature_dim=64
knn_num=9

resume=False

def save_model(test_accuracy,last_accuracy,episode,*args):
      #model,path,model,path...
      if test_accuracy > last_accuracy:
            # save networks
            for i in range(0,len(args),2):
                  torch.save(args[i].state_dict(), args[i+1])

            print("save networks for episode:", episode)

            last_accuracy = test_accuracy
      return last_accuracy

def set_model_name(root,*args):
      dir=root+'/'+'model/'
      if not os.path.isdir(dir):
            os.makedirs(dir)
      # t=time.asctime(time.localtime(time.time()))
      t=''
      path=dir+'_'.join(args)+'_'+t+'.pkl'
      return path

# model=torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=True)
if __name__ == '__main__':
      pass

