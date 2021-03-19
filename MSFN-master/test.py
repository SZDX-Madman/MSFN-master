
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader,Dataset
from time import strftime, gmtime
import pytorch_lightning as pl
from argparse import ArgumentParser
from tqdm import tqdm
from loguru import logger
from PIL import Image
from models.import_helper import get_model
import numpy as np
import os
import time
import glob
from torchvision.transforms import ToTensor
from albumentations.augmentations.transforms import RandomCrop

model_dir='weights/MSFN/epoch=190.ckpt'


class TestSet(Dataset):
    def __init__(self, test_dir, mode='test'):
        self.input_dir = f"{test_dir}/{mode}/{mode}_blur_jpeg"
        # self.input_dir = '/data/ntire/rawdata/val_input'
        self.input_list = glob.glob(self.input_dir + '/*/*.jpg')
        for i in range(len(self.input_list)):
            self.input_list[i] = self.input_list[i].replace('\\', '/')
        self.len = len(self.input_list)
        self.totensor = ToTensor()
        logger.info(f'Loaded TestSet, len:{self.len}')

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # idx = idx % self.len
        input_name = self.input_list[idx]
        input = np.array(Image.open(input_name))

        # input = RandomCrop(128, 320)(image=input)['image']
        input = self.totensor(input)
        dir = os.path.dirname(input_name)
        dir = os.path.basename(dir)
        file_name = os.path.basename(input_name)
        file_name = file_name.replace('jpg', 'png')
        savedir = os.path.join(dir, file_name)
        data = {'input': input, 'name': savedir}
        return data
class NTIRE21Model(pl.LightningModule):
    def __init__(self, hparams):
        self.hparams = hparams
        # logger
        logger.info(str(hparams))
        self.res_folder = f'./res/{self.hparams.timestamp}'
        super(NTIRE21Model, self).__init__()
        self.ensemble=hparams.self_ensemble
        self.model = get_model(f'./models/MSFN.py')

    def forward(self, x):
        return self.model(x)
    def test_step(self, batch, batch_idx):
        # OPTIONAL
        data = batch
        t1=time.time()
        if(self.hparams.self_ensemble==True):
            image = data['input']
            B,C,H,W=image.shape
            new_image=torch.zeros((B*4,C,H,W),dtype=image.dtype,device=image.device)
            new_image[0:B]=image
            new_image[B:B * 2] = torch.fliplr(image)
            image=torch.rot90(image,2,[2,3])
            new_image[B*2:B*3]=image
            new_image[B*3:B*4]=torch.fliplr(image)
            new_image2=torch.rot90(new_image, 1, [2, 3])
            data['input'] = new_image
            out1=self.forward(data)['image']
            data['input'] = new_image2
            out2=self.forward(data)['image']
            out2=torch.rot90(out2,3,[2,3])
            tempout = (out1 + out2) / 2
            tempout[B:B * 2] = torch.fliplr(tempout[B:B * 2])
            tempout[B * 2:B * 3] = torch.rot90(tempout[B * 2:B * 3], 2, [2, 3])
            tempout[B * 3:B * 4] = torch.rot90(torch.fliplr(tempout[B * 3:B * 4]), 2, [2, 3])
            for i in range(B):
                image[i] = torch.mean(tempout[i::B], 0)
            out = image
        else:
            out=self.forward(data)['image']
        t2 = time.time()
        return {'image':out,'name':data['name'],'t':t2-t1}
    def test_step_end(self, outputs):
        res = outputs['image']
        name=outputs['name']
        res=res.clamp(min=0, max=1).cpu().numpy()
        for idex,img_ in enumerate(res):
            if (img_.shape[0] == 3):
                img = Image.fromarray(
                    ((img_.transpose((1, 2, 0))) * 255).astype(np.uint8))
                if(torch.cuda.device_count()>1):
                    filename = os.path.join(self.res_folder, name[idex][0])
                else:
                    filename = os.path.join(self.res_folder, name[idex])
                file_folder = os.path.dirname(filename)
                os.makedirs(file_folder, exist_ok=True)
                logger.info(f"{filename} time:{outputs['t']}s")
                img.save(filename)
    def test_dataloader(self):
        # OPTIONAL
        dataset=TestSet(test_dir=self.hparams.test_dir)
        return DataLoader(dataset, batch_size=torch.cuda.device_count(),num_workers=torch.cuda.device_count()*4)



def main(hparams):
    logger.remove()
    os.makedirs(f'./logs/{hparams.modelname}', exist_ok=True)
    log_file = f'./logs/{hparams.modelname}/{hparams.timestamp}.txt'
    logger.add(lambda msg: tqdm.write(msg, end=""),
               format="{time:HH:mm:ss} {message}")
    logger.add(log_file, rotation="20 MB", backtrace=True, diagnose=True)
    # init module
    # Load previous params
    # embed()
    model = NTIRE21Model(hparams)
    checkpoint=torch.load(model_dir)
    state=checkpoint['state_dict']
    compatible_state_dict = {}
    dict=model.state_dict()
    key1=list(state.keys())
    key2=list(dict.keys())
    for i in range(len(key1)):
        compatible_state_dict[key2[i]]=state[key1[i]]
    model.load_state_dict(compatible_state_dict)
    # model.load_from_checkpoint(hparams.checkpoint)
    logger.info(f'Trigger whole load from {model_dir}')
    logger.info(f'Loading weights from {model_dir}')
    model.freeze()
    trainer = Trainer(
        gpus=hparams.gpus,
        # nb_gpu_nodes=hparams.nodes,
        distributed_backend='dp',
    )
    os.makedirs(f'./res/{hparams.timestamp}', exist_ok=True)
    if hparams.self_ensemble:
        logger.info("Self ensembling enabled")
    else:
        logger.info("Self ensembling disabled")
    trainer.test(model)
if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--self_ensemble', default=True,
                        help='enable naive test time augmentation')
    parser.add_argument('--gpus', type=str, default='-1')
    parser.add_argument('-M', '--modelname', type=str,default='MSFN',
                         help='indicate model module')
    # specify the timestamp to load previous models
    parser.add_argument('--timestamp', type=str,
                        default=strftime("%m-%d_%H-%M-%S", gmtime()), help='exp timestamp')
    parser.add_argument('--test_dir',type=str,default='E:/CVPR2021/datasets/REDS',
                        help='the test dataset dir including blur ')
    hparams = parser.parse_args()
    main(hparams)