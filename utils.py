# coding:utf-8
from PIL import Image
import mt_net_crosstrain as mtn
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import math
import os

basedir = os.path.abspath(os.path.dirname(__file__))

def initialize_model():
    trained_params_dir = r'static/models/model_for_Dachuang.pkl'
    trained_params_path = os.path.join(basedir,trained_params_dir)
    trained_dict = torch.load(trained_params_path, map_location=torch.device('cpu'))['state_dict']
    model = mtn.VGGNet()
    model.load_state_dict(trained_dict)
    model.eval()
    return model

def judge(model,img):
    result={}
    with torch.no_grad():
        out = model(img)
        result['label'] = torch.max(out[0],1)[1].numpy()
        result['shape'] = torch.max(out[1],1)[1].numpy()
        result['thickness'] = torch.max(out[2],1)[1].numpy()
        result['echo'] = torch.max(out[3],1)[1].numpy()
        #_, res = torch.max(out, 1)
        #_, res = torch.max(out.data, 1)
    #print(out.data)
    #print(res, torch.max(out, 1))
    #print(res, torch.max(out.data, 1))
    return result

def run(model, imgpath):
    model = model
    img = Image.open(imgpath)
    img = img.convert('RGB')
    if img is None:
        return torch.tensor([-1])
    transform = transforms.Compose([
        # transforms.CenterCrop(256),
        transforms.Lambda(lambda im: pad_img(im)),
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    img = transform(img).unsqueeze(0)
    result = judge(model, img)
    return result

def pad_img(img):
    w, h = img.size
    pading_size = (0, 0, 0, 0)
    if w > h:
        # left, top, right and bottom
        pading_size = (0, int(math.ceil((w-h)/2.0)), 0, int(math.floor((w-h)/2.0)))
    elif w < h:
        pading_size = (int(math.ceil((h-w)/2.0)), 0, int(math.floor((h-w)/2.0)), 0)
    else:
      pass
    return TF.pad(img, pading_size)

'''
if __name__ == '__main__':
    model = initialize_model()
    test = r'static/uploads/0.png'
    testpath = os.path.join(basedir,test)
    result = run(model, testpath)
    print('result:')
    for item in result:
        print(result[item][0])
    #print(model)
'''