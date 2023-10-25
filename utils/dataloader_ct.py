import os
import copy
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split
import monai
from tqdm.auto import tqdm
import pickle


# raw data path
dir_abdomen = '/mnt/mydata/whlee/Abdomen/'

my_args = {}
my_args['coarse_alpha']=1
my_args['fine_alpha']=1
my_args['concat_weight']=1
my_args['coarse_psize']=1
my_args['fine_psize']=1
my_args['enhance_alpha']=1
my_args['cuda'] = False

DEVICE = 'cpu'

organ_list = ("background", "spleen", "right kidney", "left kidney", "gallbladder", 
              "esophagus", "liver", "stomach", "aorta", "inferior vena cava", 
              "portal vein and splenic vein", "pancreas", "right adrenal gland",
              "left adrenal gland")

simple_transform = torchvision.transforms.Compose([
    monai.transforms.LoadImaged(keys=['image','label'], image_only=False), 
    monai.transforms.AddChanneld(keys=['image','label']),
    monai.transforms.CropForegroundd(keys=['image','label'], source_key='image'),
    monai.transforms.ThresholdIntensityd(
        keys=['image','label'],
        threshold=-135, 
        above=True, 
        cval=-135
    ),
    monai.transforms.ThresholdIntensityd(
        keys=['image','label'],
        threshold=215, 
        above=False, 
        cval=215
    ),
    monai.transforms.NormalizeIntensityd(keys=['image']),
    monai.transforms.ScaleIntensityd(keys=['image']),
    monai.transforms.Resized(keys=['image','label'], spatial_size=(256,256,None),mode='nearest'),
    monai.transforms.ToTensord(keys=['image','label'])
])

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y)
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        if type(X) == np.ndarray:
            X = torch.from_numpy(X)
        return X.to(torch.float), y.to(torch.long)


class Dataloader(object):
    def __init__(
        self,
        imagepath=os.path.join(dir_abdomen, 'image'),
        labelpath=os.path.join(dir_abdomen, 'label'),
        random_seed=0,
        test_size=.25,
        batch_size=1,
        organs=(0, 1, 2, 3, 6), 
    ):
        '''
        organ numbers
            (1) spleen
            (2) right kidney
            (3) left kidney
            (4) gallbladder
            (5) esophagus
            (6) liver
            (7) stomach
            (8) aorta
            (9) inferior vena cava
            (10) portal vein and splenic vein
            (11) pancreas
            (12) right adrenal gland
            (13) left adrenal gland
        '''
        print('Constructing my dataloader...')
        
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.imagefiles = np.array(sorted(glob(os.path.join(imagepath, '*.npy'))))
        self.labelfiles = np.array(sorted(glob(os.path.join(labelpath, '*.npy'))))

        indices = np.arange(len(self.imagefiles))
        self.train_idx, self.test_idx = train_test_split(
            indices, 
            test_size=test_size, 
            random_state=random_seed
        )

        self.organs = set(organs)

        self.X_train, self.X_test, self.y_train, self.y_test = self._pick_organs()
    
    def _pick_organs(self):
        train_idx = self.train_idx
        test_idx = self.test_idx
        imagefiles = self.imagefiles
        labelfiles = self.labelfiles
        organs = self.organs
        
        seed_dir = os.path.join(dir_abdomen, str(self.random_seed))
        if os.path.isdir(seed_dir):
            pass
        else:
            os.mkdir(seed_dir)
        file_dir = str()
        for l in sorted(self.organs):
            file_dir += str(l)+','
        file_dir = file_dir[:-1]
        my_dir = os.path.join(seed_dir, file_dir)
        
        if os.path.isdir(my_dir)\
            and os.path.isfile(os.path.join(my_dir, 'X_train'))\
            and os.path.isfile(os.path.join(my_dir, 'y_train'))\
            and os.path.isfile(os.path.join(my_dir, 'X_test'))\
            and os.path.isfile(os.path.join(my_dir, 'y_test'))\
        :
            print('There are data files.')
            files = glob(os.path.join(my_dir, '*'))
            unit = ['bytes', 'MB', 'GB']
            for f in sorted(files, reverse=True):
                s = os.path.getsize(f)
                for i in range(3,0,-1):
                    size = s//1024**i
                    if size>0:
                        print(f"{f.split('/')[-1]:7s}: {size:3d} {unit[i-1]}.")
                        break
            with open(os.path.join(my_dir, 'X_train'), 'rb') as f:
                X_train = pickle.load(f)
                print('Reading X_train done.')
            with open(os.path.join(my_dir, 'y_train'), 'rb') as f:
                y_train = pickle.load(f)
                print('Reading y_train done.')
            with open(os.path.join(my_dir, 'X_test'), 'rb') as f:
                X_test = pickle.load(f)
                print('Reading X_test done.')
            with open(os.path.join(my_dir, 'y_test'), 'rb') as f:
                y_test = pickle.load(f)
                print('Reading y_test done.')
        else:
            X_train = imagefiles[train_idx]
            y_train = labelfiles[train_idx]
            X_test = imagefiles[test_idx]
            y_test = labelfiles[test_idx]
            
            def __transform(x, y, organs=organs):
                label_dict = dict()
                for i, j in zip(sorted(organs), np.arange(len(organs)).astype(float)):
                    label_dict[i] = j
                
                my_x, my_y = list(), list()
                print(f'transforming length = {len(x)}')
                for _x, _y in tqdm(zip(x, y)):
                    my_dict = {"image":_x, 'label':_y}
                    transformed = simple_transform(my_dict)
                    img, lb = transformed.get('image'), transformed.get('label')
                    
                    for i in range(lb.size(-1)):
                        lb_set = set(lb[0, :, :, i].detach().cpu().numpy().astype(int).flatten())
                        
                        if len(lb_set.intersection(set(organs))) >= len(organs)//2 :
                            _img = img[0, :, :, i].clone()
                            _lb = lb[0, :, :, i].clone()
                            _shape = _lb.shape
                            _lb_holder = torch.zeros(size=_lb.view(-1).size())
                            for j in range(_lb_holder.size(0)):
                                k = int(_lb.view(-1)[j].item()) 
                                v = label_dict.get(k) 
                                if v is None:
                                    v = 0
                                _lb_holder[j] = v
                            _lb = _lb_holder.reshape(_shape)

                            my_x.append(_img)
                            my_y.append(_lb)
                    print(f'# of samples so far: {len(my_x)}')
                return my_x, my_y
            
            print('train set transforming...')
            X_train, y_train = __transform(X_train, y_train)
            print('test set transforming...')
            X_test, y_test = __transform(X_test, y_test)
            
            # make dir for pickle
            if os.path.isdir(my_dir):
                pass
            else:
                os.mkdir(my_dir)
            
            with open(os.path.join(my_dir, 'X_train'), 'wb') as f:
                pickle.dump(X_train, f)
            with open(os.path.join(my_dir, 'y_train'), 'wb') as f:
                pickle.dump(y_train, f)
                
            with open(os.path.join(my_dir, 'X_test'), 'wb') as f:
                pickle.dump(X_test, f)
            with open(os.path.join(my_dir, 'y_test'), 'wb') as f:
                pickle.dump(y_test, f)
                
        print(f"# of data: train = {len(X_train)}, test = {len(X_test)}")
                
        return X_train, X_test, y_train, y_test
        
    def get_dataloader(self):
        trainset = CustomDataset(self.X_train, self.y_train)
        testset = CustomDataset(self.X_test, self.y_test)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size)
        return trainloader, testloader