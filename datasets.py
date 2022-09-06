from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder
import random
import os
from PIL import ImageFilter, ImageOps, Image

import os

from dataloaders.TinyImageNet import TinyImageNetDataset
from dataloaders.CIFAR import CIFAR10, CIFAR100
from dataloaders.STL10 import STL10
from dataloaders.ImageNet import ImageNetDataset
from dataloaders.VisualGenome500 import VisualGenomeDataset500
from dataloaders.Cars_stanford import Cars
from dataloaders.Flowers_stanford import Flowers
from dataloaders.PASCALVOC import VocDataset
from dataloaders.MSCOCO import MSCOCO80Dataset
from dataloaders.MRNet import MRNetDataset
from dataloaders.AirCraft import Aircraft
from dataloaders.Pets import pets
from dataloaders.MNIST import MNIST
from dataloaders.CUB import Cub2011


def build_dataset(is_train, args):
    trnsfrm = build_transform(is_train, args.input_size)

    if args.data_set == 'CIFAR10':
        dataset = CIFAR10(os.path.join(args.data_location, 'CIFAR10_dataset'), 
                          download=True, train=is_train, transform=trnsfrm)
        nb_classes = 10

    
    elif args.data_set == 'CIFAR100':
        dataset = CIFAR100(os.path.join(args.data_location, 'CIFAR100_dataset'), 
                           download=True, train=is_train, transform=trnsfrm)
        nb_classes = 100
        
    
    elif args.data_set == 'Aircraft':
        dataset = Aircraft(os.path.join(args.data_location, 'Aircraft_dataset'), train=is_train, transform=trnsfrm)
        
        nb_classes = 100
    
    elif args.data_set == 'CUB':
        dataset = Cub2011(os.path.join(args.data_location, 'CUB_dataset'), train=is_train, transform=trnsfrm)
        
        nb_classes = 200
        
    elif args.data_set == 'Pets':
        split = 'trainval' if is_train else 'test'
        dataset = pets(os.path.join(args.data_location, 'Pets_dataset'), split=split, transform=trnsfrm)
        
        nb_classes = 37


    elif args.data_set == 'Cars':
        file_root = os.path.join(args.data_location, 'carsStanford/car_data/car_data/')
        
        if is_train:
            datafiles = file_root+'/train' #'TrainFiles_50Samples.csv'
        else:
            datafiles = file_root+'/test'
        dataset = Cars(datafiles, transform=trnsfrm)
        
        nb_classes = 196

    elif args.data_set == 'Flowers':
        file_root = os.path.join(args.data_location, 'Flowers')
        
        if is_train:
            datafiles = file_root+'/train' #'TrainFiles_50Samples.csv'
        else:
            datafiles = file_root+'/test'
        dataset = Flowers(datafiles, transform=trnsfrm)
        
        nb_classes = 102
   
    elif args.data_set == 'TinyImageNet':
        mode='train' if is_train else 'val'
        root_dir = os.path.join(args.data_location, 'TinyImageNet/tiny-imagenet-200/')
        dataset = TinyImageNetDataset(root_dir=root_dir, mode=mode, transform=trnsfrm)
        nb_classes = 200
        
    elif args.data_set == 'ImageNet':
        file_root = 'datasets/ImageNet_files/'

        if is_train:
            datafiles = file_root + 'TrainFiles_1300Samples_shuffled_abs.csv'
        else:
            datafiles = file_root + 'INet_val.csv'
            
        file_loc = os.path.join(args.data_location, 'still/ImageNet/ILSVRC2012')
        dataset = ImageNetDataset(datafiles, dataset_path=file_loc, transform=trnsfrm)
        
        nb_classes = 1000
        
    
    return dataset, nb_classes
    


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
    
    
def build_transform(is_train, input_size):

    if is_train:
        return transforms.Compose([
                transforms.RandomResizedCrop(input_size, scale=(0.08, 1.), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.01)],
                    p=0.5
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomChoice([Solarization(p=0.5),
                                         GaussianBlur(p=0.5)]),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
                

    # for test
    t = []
    size = int((256 / 224) * input_size)
    t.append(transforms.Resize(size, interpolation=Image.BICUBIC))
    t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    return transforms.Compose(t)
