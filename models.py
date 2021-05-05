import torch
import torch.nn as nn
from torch.nn import Conv2d, Conv1d, Dropout, ReLU, LSTM, Transformer, TransformerEncoder, Linear
import numpy as np

class HumanTrajNet(nn.Module):
    def __init__(self):
        super(HumanTrajNet,self).__init__()

    def forward(self,x):
        pass

class SocialTransformer(nn.Module):
    def __init__(self, infeats):
        super(SocialTransformer, self).__init__()
        self.transf=Transformer(infeats,1)

    def forward(self,x,y):
        return self.transf(x,y)

class SocialLSTM(nn.Module):
    def __init__(self, infeats):
        super(SocialLSTM,self).__init__()
        self.embed=Linear(infeats,64)
        self.lstm=LSTM(64,128)

    def forward(self,x):
        pass

class CoordLSTM(nn.Module):
    def __init__(self, infeats, device):
        super(CoordLSTM,self).__init__()
        self.coordEmbed=Linear(infeats,64)
        self.socialEmbed=Linear(64+infeats,64)
        self.outputEmbed=Linear(32,5)
        self.lstm=LSTM(64,32)
        self.relu=ReLU()
        self.h= {}
        self.device=device
        self.gridSize = (480, 640)
        self.numGrids = (3, 3)

        self.dropout=Dropout(.1)

    def getHidden(self,personIDs):
        h=[]
        c=[]
        for p in personIDs:
            temp = self.h.get(p,(torch.rand(32),torch.rand(32)))
            h.append(temp[0])
            c.append(temp[1])
        # import pdb;pdb.set_trace()
        return (torch.stack(h).unsqueeze(0).double().to(self.device),torch.stack(c).unsqueeze(0).double().to(self.device))

    def updateHidden(self,personIDs,h):
        # import pdb;
        # pdb.set_trace()
        for i,p in enumerate(personIDs):
            self.h[p.item()]=(h[0][0][i],h[1][0][i])

    def forward(self, peopleIDs, x, gblTensor):
        # import pdb; pdb.set_trace()
        x=self.relu(self.coordEmbed(x))
        x=self.relu(self.socialEmbed(torch.cat((x,gblTensor),-1)))#.reshape(1,64,-1)
        # x = x[0].t().unsqueeze(0)
        h=self.getHidden(peopleIDs)
        try:
            x, h=self.lstm(x,h)
        except Exception as e:
            print(e)
            import pdb;pdb.set_trace()
        self.updateHidden(peopleIDs,(h[0].detach(),h[1].detach()))
        x=self.outputEmbed(x)
        return x

    def getCoords(self,output):
        #modified from https://github.com/quancore/social-lstm
        mux, muy, sx, sy, corr = output[:, :, 0], output[:, :, 1], output[:, :, 2], output[:, :, 3], output[:, :,4]

        sx = torch.exp(sx)
        sy = torch.exp(sy)
        corr = torch.tanh(corr)

        #may not need this line
        o_mux, o_muy, o_sx, o_sy, o_corr = mux[0, :], muy[0, :], sx[0, :], sy[0, :], corr[0, :]

        numNodes = mux.size()[1]
        next_x = torch.zeros(numNodes)
        next_y = torch.zeros(numNodes)
        for node in range(numNodes):
            mean = [o_mux[node], o_muy[node]]
            cov = [[o_sx[node] * o_sx[node], o_corr[node] * o_sx[node] * o_sy[node]],
                   [o_corr[node] * o_sx[node] * o_sy[node], o_sy[node] * o_sy[node]]]

            mean = np.array(mean, dtype='float')
            cov = np.array(cov, dtype='float')
            next_values = np.random.multivariate_normal(mean, cov, 1)
            next_x[node] = next_values[0][0]
            next_y[node] = next_values[0][1]

        # import pdb; pdb.set_trace()
        return torch.cat((next_x.reshape(-1,1), next_y.reshape(-1,1)),-1), [mux.squeeze(0), muy.squeeze(0), sx.squeeze(0), sy.squeeze(0), corr.squeeze(0)]

class BGNLLLoss(nn.Module):
    def __init__(self):
        super(BGNLLLoss,self).__init__()

    def forward(self,targets,params):
        # modified from https://github.com/quancore/social-lstm
        mux, muy, sx, sy, corr = params[0].cpu(), params[1].cpu(), params[2].cpu(), params[3].cpu(), params[4].cpu()
        normx = targets[:, :, 0] - mux
        normy = targets[:, :, 1] - muy
        sxsy = sx * sy
        z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
        negRho = 1 - corr ** 2

        result = torch.exp(-z / (2 * negRho)) / (2 * np.pi * (sxsy * torch.sqrt(negRho)))
        epsilon = 1e-20
        result = -torch.log(torch.clamp(result, min=epsilon))

        #maybe loss shouldn't be summed over all people?
        loss = 0
        counter = 0
        # import pdb;
        # pdb.set_trace()
        for frame in range(targets.shape[0]):
            # nodeIDs = nodesPresent[framenum]
            # nodeIDs = [int(nodeID) for nodeID in nodeIDs]

            for person in range(targets.shape[1]):
                # nodeID = look_up[nodeID]
                loss = loss + result[frame, person]
                counter = counter + 1

        if counter != 0:
            return loss / counter
        else:
            return loss

