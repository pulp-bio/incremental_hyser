#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from scipy.io import loadmat
import numpy as np
from time import time
from sklearn.metrics import r2_score
import pickle

import torch
from torch.utils.data import SequentialSampler, DataLoader
import torch.nn as nn
from torchsummary import summary

import nemo
from tqdm import tqdm
from copy import deepcopy


# In[ ]:


v_Hz = 2000.0

S_abl = 10 # able-bodied subjects
S_amp =  2 # amputee subjects
S     = S_abl + S_amp

C_abl  = 16
C_amp  = [13, 12]
Cx_s   = [*[C_abl for s in range(S_abl)], *C_amp] # sEMG channels for each subject
Cy_DoF = 18                                       # glove channels
Cy_DoA =  5

reduce_to_DoAs = True
Cy = Cy_DoF if not reduce_to_DoAs else Cy_DoA


mix_A1_A2 = True

#do_refiltering = True
slide_s = 0.016
Tw_s    = 0.128
slide_S, Tw_S = round(v_Hz * slide_s), round(v_Hz * Tw_s)

E           = 10       # epochs
bstr, bsinf = 64, 8192 # minibatch sizes for training and inference
quantize    = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


filename = 'results_tcn_doa_unstd_%dms_8b.pkl' % (round(1000 * slide_s))


# In[ ]:


A_DoAs = np.array(
    [
       [ 0.639 ,  0.    ,  0.    ,  0.    ,  0.    ],
       [ 0.383 ,  0.    ,  0.    ,  0.    ,  0.    ],
       [ 0.    ,  1.    ,  0.    ,  0.    ,  0.    ],
       [-0.639 ,  0.    ,  0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.4   ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.6   ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ,  0.4   ,  0.    ],
       [ 0.    ,  0.    ,  0.    ,  0.6   ,  0.    ],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.1667],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.3333],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.1667],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.3333],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
       [-0.19  ,  0.    ,  0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
    ],
    dtype=np.float32,
)


# In[ ]:


class TEMPONet(nn.Module):
    
    def __init__(self, Cx, Cy):
        super().__init__()
        
        self.Cx, self.Cy = Cx, Cy
        
        self.b0_tcn0      = nn.Conv1d(self.Cx, 16, 3, dilation=2, padding=2, bias=False)
        self.b0_tcn0_BN   = nn.BatchNorm1d(16)
        self.b0_tcn0_ReLU = nn.ReLU()
        self.b0_tcn1      = nn.Conv1d(16, 16, 3, dilation=2, padding=2, bias=False)
        self.b0_tcn1_BN   = nn.BatchNorm1d(16)
        self.b0_tcn1_ReLU = nn.ReLU()
        self.b0_conv      = nn.Conv1d(16, 16, 5, stride=1, padding=2, bias=False)
        self.b0_conv_pool = torch.nn.AvgPool1d(2, stride=2, padding=0)
        self.b0_conv_BN   = nn.BatchNorm1d(16)
        self.b0_conv_ReLU = nn.ReLU()

        self.b1_tcn0      = nn.Conv1d(16, 32, 3, dilation=4, padding=4, bias=False)
        self.b1_tcn0_BN   = nn.BatchNorm1d(32)
        self.b1_tcn0_ReLU = nn.ReLU()
        self.b1_tcn1      = nn.Conv1d(32, 32, 3, dilation=4, padding=4, bias=False)
        self.b1_tcn1_BN   = nn.BatchNorm1d(32)
        self.b1_tcn1_ReLU = nn.ReLU()
        self.b1_conv      = nn.Conv1d(32, 32, 5, stride=2, padding=2, bias=False)
        self.b1_conv_pool = torch.nn.AvgPool1d(2, stride=2, padding=0)
        self.b1_conv_BN   = nn.BatchNorm1d(32)
        self.b1_conv_ReLU = nn.ReLU()

        self.b2_tcn0      = nn.Conv1d(32, 64, 3, dilation=8, padding=8, bias=False)
        self.b2_tcn0_BN   = nn.BatchNorm1d(64)
        self.b2_tcn0_ReLU = nn.ReLU()
        self.b2_tcn1      = nn.Conv1d(64, 64, 3, dilation=8, padding=8, bias=False)
        self.b2_tcn1_BN   = nn.BatchNorm1d(64)
        self.b2_tcn1_ReLU = nn.ReLU()
        self.b2_conv      = nn.Conv1d(64, 64, 5, stride=4, padding=2, bias=False)
        self.b2_conv_pool = torch.nn.AvgPool1d(2, stride=2, padding=0)
        self.b2_conv_BN   = nn.BatchNorm1d(64)
        self.b2_conv_ReLU = nn.ReLU()
        
        self.FC0         = nn.Linear(64 * 4, 64, bias=False)
        self.FC0_BN      = nn.BatchNorm1d(64)
        self.FC0_ReLU    = nn.ReLU()
        self.FC0_dropout = nn.Dropout(0.5)
        
        self.FC1         = nn.Linear(64, 32, bias=False)
        self.FC1_BN      = nn.BatchNorm1d(32)
        self.FC1_ReLU    = nn.ReLU()
        self.FC1_dropout = nn.Dropout(0.5)
        
        self.GwayFC = nn.Linear(32, self.Cy, bias=False)
        
    def forward(self, x):

        x = self.b0_tcn0_ReLU(self.b0_tcn0_BN(                  self.b0_tcn0(x) ))
        x = self.b0_tcn1_ReLU(self.b0_tcn1_BN(                  self.b0_tcn1(x) ))
        x = self.b0_conv_ReLU(self.b0_conv_BN(self.b0_conv_pool(self.b0_conv(x))))

        x = self.b1_tcn0_ReLU(self.b1_tcn0_BN(                  self.b1_tcn0(x) ))
        x = self.b1_tcn1_ReLU(self.b1_tcn1_BN(                  self.b1_tcn1(x) ))
        x = self.b1_conv_ReLU(self.b1_conv_BN(self.b1_conv_pool(self.b1_conv(x))))
        
        x = self.b2_tcn0_ReLU(self.b2_tcn0_BN(                  self.b2_tcn0(x) ))
        x = self.b2_tcn1_ReLU(self.b2_tcn1_BN(                  self.b2_tcn1(x) ))
        x = self.b2_conv_ReLU(self.b2_conv_BN(self.b2_conv_pool(self.b2_conv(x))))
        
        x = x.flatten(1)
        x = self.FC0_dropout(self.FC0_ReLU(self.FC0_BN(self.FC0(x))))
        x = self.FC1_dropout(self.FC1_ReLU(self.FC1_BN(self.FC1(x))))
        y = self.GwayFC(x)
        
        return y



summary(TEMPONet(Cx_s[0], Cy), input_size=(Cx_s[0], Tw_S), device='cpu')



def do_quantization(model, ds_calib, bsinf):
    
    Q = 8 # bits of precision
 
    model.eval()
    
    # tranformation from PyTorch model to NeMO model
    model = nemo.transform.quantize_pact(
        deepcopy(model), W_bits=Q - 1, x_bits=Q,
        dummy_input=torch.randn((1, Cx_s[s], Tw_S)).to(device), remove_dropout=True,
    )
    
    # precision dictionary
    prec_dict = {
        'b0_tcn0': {'W_bits': Q - 1},
        'b0_tcn1': {'W_bits': Q - 1},
        'b0_conv': {'W_bits': Q - 1},
        'b1_tcn0': {'W_bits': Q - 1},
        'b1_tcn1': {'W_bits': Q - 1},
        'b1_conv': {'W_bits': Q - 1},
        'b2_tcn0': {'W_bits': Q - 1},
        'b2_tcn1': {'W_bits': Q - 1},
        'b2_conv': {'W_bits': Q - 1},
        'FC0': {'W_bits': Q - 1},
        'FC1': {'W_bits': Q - 1},
        'GwayFC': {'W_bits': Q - 1}, # this maybe keep fp32
        
        'b0_tcn0_ReLU': {'x_bits': Q},
        'b0_tcn1_ReLU': {'x_bits': Q},
        'b0_conv_ReLU': {'x_bits': Q},
        'b1_tcn0_ReLU': {'x_bits': Q},
        'b1_tcn1_ReLU': {'x_bits': Q},
        'b1_conv_ReLU': {'x_bits': Q},
        'b2_tcn0_ReLU': {'x_bits': Q},
        'b2_tcn1_ReLU': {'x_bits': Q},
        'b2_conv_ReLU': {'x_bits': Q},
        'FC0_ReLU': {'x_bits': Q},
        'FC1_ReLU': {'x_bits': Q},
    }
    
    # application of the precision dictionary
    model.change_precision(bits=1, scale_weights=True, scale_activations=True, min_prec_dict=prec_dict, verbose=True)
    
    # calibration on calibration set
    model.eval()
    with model.statistics_act():
        do_inference(model, criterion, device, ds_calib, bsinf)
    model.reset_alpha_act()
    
    return model



def do_inference(model, criterion, device, ds, bsinf):

    model.eval()
    
    dataloader = DataLoader(ds, batch_size=bsinf, drop_last=False, shuffle=False, sampler=SequentialSampler(ds))

    with torch.no_grad():
        Y, YHat = torch.zeros((0, model.Cy), device=device), torch.zeros((0, model.Cy), device=device)
        for Xb, Yb in dataloader:
            Y = torch.cat((Y, Yb.to(device)))
            YHat = torch.cat((YHat, model(Xb.to(device))))
            
        l    = criterion(YHat, Y).item()
        Y, YHat = Y.cpu().numpy(),YHat.cpu().numpy()
        r2mv =    r2multivariate(Y, YHat)
        rmse = rmse_multivariate(Y, YHat)
        mae  =  mae_multivariate(Y, YHat)

    return YHat, l, r2mv, rmse, mae



def do_fit(model, criterion, device, ds_tr, ds_va, bstr, bsinf, optimizer, E):

    model.train()
    
    loader_tr = DataLoader(ds_tr, batch_size=bstr, drop_last=True, shuffle=True)

    ltr, r2mvtr, rmsetr, maetr, lva, r2mvva, rmseva, maeva = np.zeros(E), np.zeros(E), np.zeros(E), np.zeros(E), np.zeros(E), np.zeros(E), np.zeros(E), np.zeros(E)
    print('\nEPOCH\tLtr\tR2tr\tLva\tR2va\tTime (s)\n')
    for e in range(E):

        if quantize and 1 + e == E: # only quantized the last epoch
            model = do_quantization(model, ds_tr, bsinf)
            optimizer = torch.optim.Adam(model.parameters())
            print('Quantization from now on')

        t0 = time()
        for Xb, Yb in loader_tr:
            optimizer.zero_grad()
            criterion(model(Xb.to(device)), Yb.to(device)).backward()
            optimizer.step()
        _, ltr[e], r2mvtr[e], rmsetr[e], maetr[e] = do_inference(model, criterion, device, ds_tr, bsinf)
        _, lva[e], r2mvva[e], rmseva[e], maeva[e] = do_inference(model, criterion, device, ds_va, bsinf)
        print('%d/%d\t%.3f\t%.4f\t%.3f\t%.4f\t%.1f%s' % (1 + e, E, ltr[e], r2mvtr[e], lva[e], r2mvva[e], time() - t0, (1 + e == E) * 3 * '\n'))

    if quantize:
        model.qd_stage(eps_in=1.0)
        model.id_stage()
        torch.save(model.state_dict(), 'model_subj%s.sd.pt' % s)

    history = {'ltr': ltr, 'r2mvtr': r2mvtr, 'lva': lva, 'r2mvva': r2mvva, 'rmseva': rmseva, 'maeva': maeva}
    
    return history


# In[ ]:


class WindowedSession():
    
    def __init__(self, XY, v_Hz=v_Hz, slide_s=slide_s, Tw_s=Tw_s):
        self.X, self.Y = XY
        self.C, self.M = self.X.shape
        self.v_Hz = v_Hz
        self.slide_s, self.Tw_s = slide_s, Tw_s
        self.slide_S, self.Tw_S = round(self.v_Hz * self.slide_s), round(self.v_Hz * self.Tw_s)
        self.Mw = (self.M - self.Tw_S) // self.slide_S + 1
        
    def __len__(self):
        return self.Mw
        
    def __data_generation(self, index):
        stop = self.Tw_S + index * self.slide_S
        return self.X[:, stop - self.Tw_S :stop], self.Y[-1 + stop]
    
    def __getitem__(self, index):
        return self.__data_generation(index)



def load_and_arrange(s, mix_A1_A2, v_Hz=v_Hz, slide_s=slide_s, Tw_s=Tw_s):
    
    
    MATLAB_workspace = loadmat('/home/zanghieri/work/NinaPro/NinaPro_DB8/Data/downloaded_mat/S%s_E1_A1.mat' % (1 + s))
    X_A1, Y_A1 = MATLAB_workspace['emg'][:, :Cx_s[s]], MATLAB_workspace['glove']
    del MATLAB_workspace
    
    MATLAB_workspace = loadmat('/home/zanghieri/work/NinaPro/NinaPro_DB8/Data/downloaded_mat/S%s_E1_A2.mat' % (1 + s))
    X_A2, Y_A2 = MATLAB_workspace['emg'][:, :Cx_s[s]], MATLAB_workspace['glove']
    del MATLAB_workspace
    
    MATLAB_workspace = loadmat('/home/zanghieri/work/NinaPro/NinaPro_DB8/Data/downloaded_mat/S%s_E1_A3.mat' % (1 + s))
    X_A3, Y_A3 = MATLAB_workspace['emg'][:, :Cx_s[s]], MATLAB_workspace['glove']
    del MATLAB_workspace

    
    #if do_refiltering:
    #    from scipy.signal import butter, sosfilt
    #    sos = butter(4, (10.0, 500.0), 'bp', fs=2000.0, output='sos')
    #    X_A1 = sosfilt(sos, X_A1, 0).astype(np.float32)
    #    X_A2 = sosfilt(sos, X_A2, 0).astype(np.float32)
    #    X_A3 = sosfilt(sos, X_A3, 0).astype(np.float32)
    
    
    if not mix_A1_A2:
        XTrain, YTrain = X_A1, Y_A1
        XValid, YValid = X_A2, Y_A2
    else:
        halfslide_S = round(v_Hz * slide_s / 2)
        XTrain, YTrain = np.concatenate((X_A1, X_A2))[:-halfslide_S], np.concatenate((Y_A1, Y_A2))[:-halfslide_S]
        XValid, YValid = np.concatenate((X_A1, X_A2))[ halfslide_S:], np.concatenate((Y_A1, Y_A2))[ halfslide_S:]
    del X_A1, Y_A1, X_A2, Y_A2
    XTest, YTest = X_A3, Y_A3
    del X_A3, Y_A3
    
    
    if reduce_to_DoAs:
        YTrain = np.dot(YTrain, A_DoAs)
        YValid = np.dot(YValid, A_DoAs)
        YTest  = np.dot(YTest , A_DoAs)
    
    
    # Signal quantization
    q = 0.01
    Q = 8
    x_extr  = np.quantile(np.abs(XTrain), 1 - q, axis=0).astype(np.float32)
    
    XTrain = np.floor(np.clip(XTrain, -x_extr, +x_extr) / (2 * x_extr) * 2**Q)
    XTrain[XTrain == 2**Q] -= 1
    XValid = np.floor(np.clip(XValid, -x_extr, +x_extr) / (2 * x_extr) * 2**Q)
    XValid[XTrain == 2**Q] -= 1
    XTest  = np.floor(np.clip(XTest , -x_extr, +x_extr) / (2 * x_extr) * 2**Q)
    XTest [XTest  == 2**Q] -= 1

    
    XTrain, XValid, XTest = XTrain.T, XValid.T, XTest.T
 
    setTrain = WindowedSession((XTrain, YTrain))
    setValid = WindowedSession((XValid, YValid))
    setTest  = WindowedSession((XTest , YTest ))
    
    return setTrain, setValid, setTest



def r2multivariate(Y, YHat):
    # Y and YHat are in format (T, C)
    return 1 - ((YHat - Y)**2).sum() / ((Y - Y.mean())**2).sum()

def rmse_multivariate(Y, YHat):
    # Y and YHat are in format (T, C)
    return ((YHat - Y)**2).mean()**0.5

def mae_multivariate(Y, YHat):
    # Y and YHat are in format (T, C)
    return np.abs(YHat - Y).mean()



def EMA_pass(x, alpha):
    M = x.shape[0] # x is either in format (T,) or (T, C)
    y = np.zeros(x.shape)
    y[0] = (1 - alpha) * x.mean(0) + alpha * x[0]
    for m in range(1, M):
        y[m] = (1 - alpha) * y[m - 1] + alpha * x[m]
    return y



def tune_alpha_EMA(xnoisy, x):
    
    N = 50 # N = 100 takes almost 2min/subject
    alphas = np.arange(N + 1) / N
    
    C = x.shape[1] # x is in format (T, C), not squeezed (T,)
    r2uv = np.zeros((C, N + 1))
    for c in range(C):
        for na in range(N + 1):
            r2uv[c, na] = r2_score(x[:, c], EMA_pass(xnoisy[:, c], alphas[na]))
    
    alpha0 = alphas[r2uv.argmax(1)]
    
    xema0 = EMA_pass(xnoisy, alpha0)
    r2mv0 =    r2multivariate(x, xema0)
    rmse0 = rmse_multivariate(x, xema0)
    mae0  =  mae_multivariate(x, xema0)

    return alpha0, xema0, r2mv0, rmse0, mae0


# In[ ]:


# Results structure
results = {
    
    'v_Hz'   : v_Hz,
    'slide_s': slide_s,
    'Tw_s'   : Tw_s,
    
    'ytr': [None for s in range(S)],
    'yva': [None for s in range(S)],
    'yte': [None for s in range(S)],

    'history'  : [{} for s in range(S)],
    'rgr'      : [None for s in range(S)],
    'alpha_ema': np.full((S, Cy), None),

    'ytrhat': [None for s in range(S)],
    'yvahat': [None for s in range(S)],
    'ytehat': [None for s in range(S)],
    'r2mvtr': np.full(S, None),
    'r2mvva': np.full(S, None),
    'r2mvte': np.full(S, None),
    'rmsetr': np.full(S, None),
    'rmseva': np.full(S, None),
    'rmsete': np.full(S, None),
    'maetr' : np.full(S, None),
    'maeva' : np.full(S, None),
    'maete' : np.full(S, None),
    
    'ytrhat_ema': [None for s in range(S)],
    'yvahat_ema': [None for s in range(S)],
    'ytehat_ema': [None for s in range(S)],
    'r2mvtr_ema': np.full(S, None),
    'r2mvva_ema': np.full(S, None),
    'r2mvte_ema': np.full(S, None),
    'rmsetr_ema': np.full(S, None),
    'rmseva_ema': np.full(S, None),
    'rmsete_ema': np.full(S, None),
    'maetr_ema' : np.full(S, None),
    'maeva_ema' : np.full(S, None),
    'maete_ema' : np.full(S, None),
    
}


# In[ ]:


for s in range(S):
    print('SUBJECT %d/%d' % (1 + s, S))
    
    
    # Datasets
    setTrain, setValid, setTest = load_and_arrange(s, mix_A1_A2)
    YTrain = np.array([setTrain[m][1] for m in range(setTrain.Mw)])
    YValid = np.array([setValid[m][1] for m in range(setValid.Mw)])
    YTest  = np.array([setTest [m][1] for m in range(setTest.Mw )])
    
    
    # Training
    model     = TEMPONet(Cx_s[s], Cy).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    history   = do_fit(model, criterion, device, setTrain, setTest, bstr, bsinf, optimizer, E)
        
    YTrainHat, _, r2mvTrain, rmseTrain, maeTrain = do_inference(model, criterion, device, setTrain, bsinf)
    YValidHat, _, r2mvValid, rmseValid, maeValid = do_inference(model, criterion, device, setValid, bsinf)
    YTestHat , _, r2mvTest , rmseTest , maeTest  = do_inference(model, criterion, device, setTest , bsinf)
    model_sd = model.cpu().state_dict()
    del model
    
    
    # Tuning of EMA's alpha
    t0_s = time()
    alpha, YTrainHat_ema, r2mvTrain_ema, rmseTrain_ema, maeTrain_ema = tune_alpha_EMA(YTrainHat, YTrain)
    print('Tuning time for alpha of EMA: %.0fs' % round(time() - t0_s, 3))
    YValidHat_ema = EMA_pass(YValidHat, alpha)
    YTestHat_ema  = EMA_pass(YTestHat , alpha)
    r2mvValid_ema =    r2multivariate(YValid, YValidHat_ema)
    r2mvTest_ema  =    r2multivariate(YTest , YTestHat_ema )
    rmseValid_ema = rmse_multivariate(YValid, YValidHat_ema)
    rmseTest_ema  = rmse_multivariate(YTest , YTestHat_ema )
    maeValid_ema  =  mae_multivariate(YValid, YValidHat_ema)
    maeTest_ema   =  mae_multivariate(YTest , YTestHat_ema )
    
    
    print('\nR2_multivariate')
    print(round(r2mvTrain    , 3), round(r2mvValid    , 3), round(r2mvTest    , 3))
    print(round(r2mvTrain_ema, 3), round(r2mvValid_ema, 3), round(r2mvTest_ema, 3))
    print('\nRMSE')
    print(round(rmseTrain    , 3), round(rmseValid    , 3), round(rmseTest    , 3))
    print(round(rmseTrain_ema, 3), round(rmseValid_ema, 3), round(rmseTest_ema, 3))
    print('\nMAE')
    print(round(maeTrain     , 3), round(maeValid     , 3), round(maeTest     , 3))
    print(round(maeTrain_ema , 3), round(maeValid_ema , 3), round(maeTest_ema , 3))
    print('\n\n\n')
    
    
    
    # Store results
    results['ytr'][s] = YTrain
    results['yva'][s] = YValid
    results['yte'][s] = YTest
    
    results['history'  ][s] = history
    results['rgr'      ][s] = model_sd
    results['alpha_ema'][s] = alpha
    
    results['ytrhat'][s] = YTrainHat
    results['yvahat'][s] = YValidHat
    results['ytehat'][s] = YTestHat
    results['r2mvtr'][s] = r2mvTrain
    results['r2mvva'][s] = r2mvValid
    results['r2mvte'][s] = r2mvTest
    results['rmsetr'][s] = rmseTrain
    results['rmseva'][s] = rmseValid
    results['rmsete'][s] = rmseTest
    results['maetr' ][s] = maeTrain
    results['maeva' ][s] = maeValid
    results['maete' ][s] = maeTest
    
    results['ytrhat_ema'][s] = YTrainHat_ema
    results['yvahat_ema'][s] = YValidHat_ema
    results['ytehat_ema'][s] = YTestHat_ema
    results['r2mvtr_ema'][s] = r2mvTrain_ema
    results['r2mvva_ema'][s] = r2mvValid_ema
    results['r2mvte_ema'][s] = r2mvTest_ema
    results['rmsetr_ema'][s] = rmseTrain_ema
    results['rmseva_ema'][s] = rmseValid_ema
    results['rmsete_ema'][s] = rmseTest_ema
    results['maetr_ema' ][s] = maeTrain_ema
    results['maeva_ema' ][s] = maeValid_ema
    results['maete_ema' ][s] = maeTest_ema



    # Save results after each subject
    pickle.dump({'results': results}, open(filename, 'wb'))


# In[ ]:


# Show a summary of the results


def print_metric_summary(metric_name):
    
    mtr     = results[metric_name + 'tr'    ]
    mva     = results[metric_name + 'va'    ]
    mte     = results[metric_name + 'te'    ]
    mtr_ema = results[metric_name + 'tr_ema']
    mva_ema = results[metric_name + 'va_ema']
    mte_ema = results[metric_name + 'te_ema']
    
    print('\n' + m + '\n')
    print('Average\tSE\tStd')
    print('Before EMA')
    print('Average\tSE\tStd')
    print('%.3f\t%.3f\t%.3f' % (mtr.mean()    , mtr.std(ddof=1)     / S**0.5, mtr.std(ddof=1)    ))
    print('%.3f\t%.3f\t%.3f' % (mva.mean()    , mva.std(ddof=1)     / S**0.5, mva.std(ddof=1)    ))
    print('%.3f\t%.3f\t%.3f' % (mte.mean()    , mte.std(ddof=1)     / S**0.5, mte.std(ddof=1)    ))
    print('After EMA')
    print('%.3f\t%.3f\t%.3f' % (mtr_ema.mean(), mtr_ema.std(ddof=1) / S**0.5, mtr_ema.std(ddof=1)))
    print('%.3f\t%.3f\t%.3f' % (mva_ema.mean(), mva_ema.std(ddof=1) / S**0.5, mva_ema.std(ddof=1)))
    print('%.3f\t%.3f\t%.3f' % (mte_ema.mean(), mte_ema.std(ddof=1) / S**0.5, mte_ema.std(ddof=1)))
    print('\n')



# Classification metrics
metric_names = ['r2mv', 'rmse', 'mae']
for m in metric_names:
    print_metric_summary(m)

