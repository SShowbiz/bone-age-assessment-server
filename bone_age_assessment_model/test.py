# import
import torch
import torch.nn as nn
from torchvision import transforms
from .model import *
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import base64
import numpy as np
import re
from PIL import Image
from io import StringIO
from io import BytesIO

checkpoint_dir = "./bone_age_assessment_model/checkpoint"

def init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def load(checkpoint_dir, classificationModel, optim):
    if not os.path.exists(checkpoint_dir):
        epoch = 0
        return classificationModel, optim, epoch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_lst = os.listdir(checkpoint_dir)
    ckpt_lst = [f for f in ckpt_lst if f.endswith('pth')]
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load(
        '%s/%s' % (checkpoint_dir, ckpt_lst[-1]), map_location=device)

    classificationModel.load_state_dict(dict_model['classificationModel'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return classificationModel, optim, epoch


def test(imagePath):
    imageObject = Image.open(BytesIO(base64.b64decode(
        re.sub("data:image/png;base64", '', imagePath)))).convert('L')
    imageObject.seek(0)
    image = np.array(imageObject)
    imageTransforms = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize(size=(260, 200))])

    transFormedImage = imageTransforms(image)
    imageTransformsTensor = transforms.Compose([transforms.ToPILImage(
    ), transforms.Resize(size=(260, 200)), transforms.ToTensor()])
    transformedImageTensor = imageTransformsTensor(
        (image[:, :, np.newaxis]/255.0).astype(np.float32)).unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classificationModel = ClassificationModel().to(device)
    init_weights(classificationModel)
    MAELoss = nn.L1Loss().to(device)
    optim = torch.optim.Adam(
        classificationModel.parameters(), lr=2e-5, betas=(0.5, 0.999))
    classificationModel, optim, startEpoch = load(
        checkpoint_dir=checkpoint_dir, classificationModel=classificationModel, optim=optim)

   # 결과 출력하기
    with torch.no_grad():
        classificationModel.eval()
        output = classificationModel(transformedImageTensor)
        predictBoneAgeForYear = float(output)/12

    return predictBoneAgeForYear
