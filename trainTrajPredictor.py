import sys

sys.path.append('../social_lstm')
from opTrajData import OpTrajData
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
# from social_lstm.model import SocialModel
import torch
from args import getArgs
from models import SocialTransformer, CoordLSTM, SocialLSTM, BGNLLLoss
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from utils import world2image, computeGlobalGroups, processGroups


def train(epochs, device, loss, dloader):
    model = CoordLSTM(2)  # SocialTransformer(2)#SocialModel(args)
    model = model.to(device)
    model = model.double()
    opt = optim.RMSprop(model.parameters(), lr=5e-4)
    for e in range(epochs):
        print("Epoch:", e)
        totalLoss = 0
        trackLoss = []
        totLen = len(dloader)
        for peopleIDs, pos, target, ims in tqdm(dloader):
            # import pdb; pdb.set_trace()
            if pos.size(1) > 0 and target.size(1) == pos.size(1):
                # outputs=model(pos.double(),target.double())
                gblGroups = []
                for p in pos:
                    gblGroups.append(computeGlobalGroups(world2image(p, np.linalg.inv(data.H)), model.numGrids, model.gridSize))
                groupedFeatures = processGroups(gblGroups, pos, 'coords')
                coeffs = model(peopleIDs, pos.double(), torch.stack(groupedFeatures))
                outputs, params = model.getCoords(coeffs)
                l = loss(target,params)
                # import pdb; pdb.set_trace()
                opt.zero_grad()
                l.backward()
                opt.step()
                totalLoss += l.item()
            else:
                totLen -= 1
        print('Loss:', totalLoss / totLen, 'totLen:', totLen)
        trackLoss.append(totalLoss / totLen)
    torch.save(model.state_dict(), 'coordLSTMweights.pt')
    plt.plot(trackLoss)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Bivariate Gaussian NLL')
    plt.show()


def test(device, loss, dloader, save_path):
    # test loop
    model = CoordLSTM(2)  # SocialTransformer(2)#SocialModel(args)
    model.load_state_dict(torch.load(save_path))
    model.eval()
    model = model.to(device)
    model = model.double()
    totalLoss = 0
    totLen = len(dloader)
    for peopleIDs, pos, target, ims in tqdm(dloader):
        # import pdb; pdb.set_trace()
        if pos.size(1) > 0 and target.size(1) == pos.size(1):
            # outputs=model(pos.double(),target.double())
            gblGroups = []
            for p in pos:
                gblGroups.append(computeGlobalGroups(world2image(p, np.linalg.inv(data.H)), model.numGrids, model.gridSize))
            groupedFeatures = processGroups(gblGroups, pos, 'coords')
            coeffs = model(peopleIDs, pos.double(), torch.stack(groupedFeatures))
            outputs, params = model.getCoords(coeffs)
            l = loss(target, params)
            totalLoss += l.item()
        else:
            totLen -= 1
    print('Loss:', totalLoss / totLen, 'totLen:', totLen)


# args=getArgs()
data = OpTrajData('ETH', 'by_frame', 'mask')
dloader = DataLoader(data, batch_size=1, shuffle=True, drop_last=False)
loss = BGNLLLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 10

train(epochs, device, loss, dloader)

data = OpTrajData('UCY', 'by_frame', None)
dloader = DataLoader(data, batch_size=1, shuffle=True, drop_last=False)
test(device, loss, dloader, 'testTransfWeights.pt')
