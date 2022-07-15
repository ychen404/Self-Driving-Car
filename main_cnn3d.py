from skimage.transform import resize
from itertools import product
from collections import OrderedDict
from tqdm.notebook import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils, models
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import pdb

from Run import RunBuilder as RB
from Run import RunManager as RM

from DataLoading import UdacityDataset as UD
from DataLoading import ConsecutiveBatchSampler as CB
from model import Convolution3D as CNN3D

device = torch.device("cuda")

print("Setting up parameters")

parameters = OrderedDict(
    file = ['3DCNN_Paper'], # used to mark specific files in case that we want to check them on tensorboard
    learning_rate = [0.001],
    # batch_size = [5],
    batch_size = [5],
    seq_len = [5],
    num_workers = [4],
)
m = RM.RunManager()

print("Setting up model")

for run in RB.RunBuilder.get_runs(parameters):
    print(f"run: {run}")
    network = CNN3D.Convolution3D().to(device)

    optimizer = optim.Adam(network.parameters(),lr = run.learning_rate,betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)

    udacity_dataset = UD.UdacityDataset(csv_file='/workspace/Code/Ch2_002_export_HMB_1/interpolated.csv',
                                     root_dir='/workspace/Code/Ch2_002_export_HMB_1/',
                                     transform=transforms.Compose([transforms.ToTensor()]),
                                     select_camera='center_camera')

    dataset_size = int(len(udacity_dataset))
    print(f"Dataset size: {dataset_size}")
    del udacity_dataset
    split_point = int(dataset_size * 0.8)


    training_set = UD.UdacityDataset(csv_file='/workspace/Code/Ch2_002_export_HMB_1/interpolated.csv',
                                     root_dir='/workspace/Code/Ch2_002_export_HMB_1/',
                                     transform=transforms.Compose([transforms.ToTensor()]),
                                     select_camera='center_camera',
                                     select_range=(0,split_point))

    validation_set = UD.UdacityDataset(csv_file='/workspace/Code/Ch2_002_export_HMB_1/interpolated.csv',
                                     root_dir='/workspace/Code/Ch2_002_export_HMB_1/',
                                     transform=transforms.Compose([transforms.ToTensor()]),
                                     select_camera='center_camera',
                                     select_range=(split_point,dataset_size))

    training_cbs = CB.ConsecutiveBatchSampler(data_source=training_set, batch_size=run.batch_size, shuffle=True, drop_last=False, seq_len=run.seq_len)
    training_loader = DataLoader(training_set, sampler=training_cbs, num_workers=run.num_workers, pin_memory=True, collate_fn=(lambda x: x[0]))
    
    validation_cbs = CB.ConsecutiveBatchSampler(data_source=validation_set, batch_size=run.batch_size, shuffle=True, drop_last=False, seq_len=run.seq_len)
    validation_loader = DataLoader(validation_set, sampler=validation_cbs, num_workers=run.num_workers, collate_fn=(lambda x: x[0]))
    
    m.begin_run(run,network,[run.batch_size,3,run.seq_len,120,320])
    for epoch in range(10):
        m.begin_epoch()
        print(f"Epoch: {epoch}")
        # Calculation on Training Loss
        # for training_sample in tqdm(training_loader, total=int(len(training_set)/run.batch_size/run.seq_len)):
        for training_sample in training_loader:

            training_sample['image'] = torch.Tensor(resize(training_sample['image'], (run.batch_size,run.seq_len,3,120,320),anti_aliasing=True))
            training_sample['image'] = training_sample['image'].permute(0,2,1,3,4)
            
            param_values = [v for v in training_sample.values()]
            image,angle = param_values[0],param_values[3]
            
            image = image.to(device, non_blocking=True)
            prediction = network(image)
            prediction = prediction.squeeze().permute(1,0).to(device)

            # The angle data is float64 when first loaded somehow
            labels = angle.to(device, dtype=torch.float32, non_blocking=True)
            del param_values, image, angle
            if labels.shape[0]!=prediction.shape[0]:
                prediction = prediction[-labels.shape[0],:]
            training_loss_angle = F.mse_loss(prediction,labels,size_average=None, reduce=None, reduction='mean')
            print(f"training_loss_angle: {training_loss_angle.item()}")
            optimizer.zero_grad()# zero the gradient that are being held in the Grad attribute of the weights
            
            training_loss_angle.backward() # calculate the gradients
            optimizer.step() # finishing calculation on gradient
            
        print("Done")
        # Calculation on Validation Loss

        with torch.no_grad():    
            # for Validation_sample in tqdm(validation_loader, total=int(len(validation_set)/run.batch_size/run.seq_len)):
            for Validation_sample in validation_loader:
                Validation_sample['image'] = torch.Tensor(resize(Validation_sample['image'], (run.batch_size,run.seq_len,3,120,320),anti_aliasing=True))
                Validation_sample['image'] = Validation_sample['image'].permute(0,2,1,3,4)

                param_values = [v for v in Validation_sample.values()]
                image,angle = param_values[0],param_values[3]
                image = image.to(device)
                prediction = network(image)
                prediction = prediction.squeeze().permute(1,0).to(device)
                curr_batch_size = len(image)
                labels = angle.to(device)
                del param_values, image, angle
                if labels.shape[0]!=prediction.shape[0]:
                    prediction = prediction[-labels.shape[0],:]
                validation_loss_angle = F.mse_loss(prediction,labels,size_average=None, reduce=None, reduction='mean')                
                m.track_loss(validation_loss_angle, curr_batch_size)
                m.track_num_correct(prediction,labels) 
        # m.end_epoch(validation_set)
        m.end_epoch(validation_loader)
        torch.save(network.state_dict(), "saved_models/CNN3D/epoch-{}".format(epoch))
    m.end_run()
m.save('result')