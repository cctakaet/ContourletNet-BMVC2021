import os, subprocess
import torch
import torch.nn.functional as F

import numpy as np
import cv2

from time import time
data_format = 'NHWC'

class DepthToSpace(torch.nn.Module):

    def __init__(self, h_factor=2, w_factor=2):
        super().__init__()
        self.h_factor, self.w_factor = h_factor, w_factor
    
    def forward(self, x):
        return pixelShuffle(x, self.h_factor, self.w_factor)

class SpaceToDepth(torch.nn.Module):

    def __init__(self, h_factor=2, w_factor=2):
        super().__init__()
        self.h_factor, self.w_factor = h_factor, w_factor
    
    def forward(self, x):
        return inv_pixelShuffle(x, self.h_factor, self.w_factor)

class ContourDec(torch.nn.Module):

    def __init__(self, nlevs):
        super().__init__()
        self.nlevs = nlevs
    
    def forward(self, x):
        return pdfbdec_layer(x, self.nlevs)

class ContourRec(torch.nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return pdfbrec_layer(x[0], x[1])

def dup(x, step=[2,2]):
    N,C,H,W = x.shape
    y = torch.zeros((N,C,H*step[0],W*step[1]), device=x.device)
    y[...,::step[0],::step[1]]=x
    return y
    
def conv2(x, W, C=1, strides=[1, 1, 1, 1], padding=0):
    return F.conv2d(x, W, padding=padding, groups=C)

def extend2_layer(x, ru, rd, cl, cr, extmod):
    # rx, cx = x.get_shape().as_list()[1:3]
    rx, cx = x.shape[2:]
    if extmod == 'per':

        y = torch.cat([x[..., rx-ru:rx,:],x,x[..., :rd,:]], dim=2)
        y = torch.cat([y[..., cx-cl:cx],y,y[..., :cr]], dim=3)

    elif extmod == 'qper_row':
        raise ValueError
        rx2 = round(rx / 2)
        y1 = K.concatenate([x[:,rx2:rx, cx-cl:cx,:], x[:,:rx2, cx-cl:cx,:]],axis=1)
        y2=K.concatenate([x[:,rx2:rx, :cr,:], x[:,:rx2, :cr,:]],axis=1)
        y=K.concatenate([y1,x,y2], axis=1)
        
        y=K.concatenate([y[:,rx-ru:rx,:,:],y,y[:,:rd,:,:]], axis=1)

    elif extmod == 'qper_col':
        
        cx2 = round(cx / 2)
        y1 = torch.cat([x[..., rx-ru:rx, cx2:cx], x[..., rx-ru:rx, :cx2]],dim=3)
        y2 = torch.cat([x[..., :rd, cx2:cx], x[..., :rd, :cx2]],dim=3)
        y = torch.cat([y1,x,y2], dim=2)
        
        y = torch.cat([y[..., cx-cl:cx],y,y[..., :cr]],dim=3)
    return y

def sefilter2_layer(x, f1, f2, extmod='per',shift=[0,0]):
    # Periodized extension
    f1_len = len(f1)
    f2_len = len(f2)
    lf1 = (f1_len - 1) / 2
    lf2 = (f2_len - 1) / 2
    
    y = extend2_layer(x, int(np.floor(lf1) + shift[0]), int(np.ceil(lf1) - shift[0]), \
         int(np.floor(lf2) + shift[1]), int(np.ceil(lf2) - shift[1]),extmod)

    # Seperable filter
    ch = y.shape[1]
    # f3=np.zeros((f1_len, 1, ch, ch), dtype = np.float32)
    f3 = torch.zeros((ch, ch, f1_len, 1), device=x.device)
    for i in range(ch):
        # f3[:,0,i,i] = f1
        f3[i,i,:,0] = f1
        
    # f4=np.zeros((1, f2_len, ch, ch), dtype = np.float32)
    f4 = torch.zeros((ch, ch, 1, f2_len), device=x.device)
    for i in range(ch):
        # f4[0,:,i,i] = f2
        f4[i,i,0,:] = f2
    
    y = conv2(y, f3)
    y = conv2(y, f4)
    
    return y

def lap_filter(device, dtype):
    # use '9-7' filter for the Laplacian pyramid
    h = np.array([0.037828455506995, -0.02384946501938, -0.11062440441842, 0.37740285561265], dtype = np.float32)
    h = np.concatenate((h,[0.8526986790094],h[::-1]))
        
    g = np.array([-0.064538882628938, -0.040689417609558, 0.41809227322221], dtype = np.float32)
    g = np.concatenate((g,[0.78848561640566],g[::-1]))
    h, g = torch.from_numpy(h), torch.from_numpy(g)
    h, g = h.to(device), g.to(device)
    return h, g

def lpdec_layer(x):
    h, g = lap_filter(x.device, x.dtype)

    # Lowpass filter and downsample
    xlo = sefilter2_layer(x,h,h,'per')
    c = xlo[:,:,::2,::2]

    # Compute the residual (bandpass) image by upsample, filter, and subtract
    # Even size filter needs to be adjusted to obtain perfect reconstruction
    adjust = (len(g) + 1)%2
    
    # d = insert_zero(xlo) # d = Lambda(insert_zero)(xlo)
    d = dup(c) # d = Lambda(insert_zero)(xlo)

    d = sefilter2_layer(d,g,g,'per',adjust*np.array([1,1], dtype = np.float32))
    
    d = x-d # d = Subtract()([x,d])

    return c,d

def lprec_layer(c,d):
    h, g = lap_filter(c.device, c.dtype)

    xhi = sefilter2_layer(d,h,h,'per')
    xhi = xhi[...,::2,::2] # xhi = Lambda(lambda x: x[:,::2,::2,:])(xhi)

    xlo = c - xhi # xlo = Subtract()([c, xhi])
    xlo = dup(xlo) # xlo = Lambda(dup)(xlo)
    
    # Even size filter needs to be adjusted to obtain 
    # perfect reconstruction with zero shift
    adjust = (len(g) + 1)%2
    
    xlo = sefilter2_layer(xlo,g,g,'per',adjust*np.array([1,1]))
    
    # Final combination
    x = xlo + d # x = Add()([xlo, d])
    
    return x

def pdfbdec_layer(x, nlevs, pfilt=None, dfilt=None):
    if nlevs != 0:
        # Laplacian decomposition
        xlo, xhi = lpdec_layer(x)

        # Use the ladder structure (whihc is much more efficient)
        xhi = dfbdec_layer(xhi, dfilt, nlevs)

    return xlo, xhi

def dfb_filter(device):
    # length 12 filter from Phoong, Kim, Vaidyanathan and Ansari
    v = np.array([0.63,-0.193,0.0972,-0.0526,0.0272,-0.0144], dtype = np.float32)
    # Symmetric impulse response
    f = np.concatenate((v[::-1],v))
    # Modulate f
    f[::2] = -f[::2]
    f = torch.from_numpy(f)
    f = f.to(device)
    return f

def new_fbdec_layer(x, f_, type1, type2, extmod='per'):

    sample = len(x)
    # x = tf.concat(x, axis=-1)
    x = torch.cat(x, dim=1)
    ch = x.shape[1] // sample

    # Polyphase decomposition of the input image
    if type1 == 'q':
        # Quincunx polyphase decomposition
        p0,p1 = qpdec_layer(x,type2)
        
    elif type1 == 'p':
        # Parallelogram polyphase decomposition
        p0,p1 = ppdec_layer(x,type2)

    # Ladder network structure
    y0 = 1 / (2**0.5) * (p0 - sefilter2_layer(p1,f_,f_,extmod,[1,1]))
    y1 = (-2**0.5)*p1 - sefilter2_layer(y0,f_,f_,extmod)
    
    # return [y0, y1]
    return [y0[:,i*ch:(i+1)*ch] for i in range(sample)], [y1[:,i*ch:(i+1)*ch] for i in range(sample)]

### TODO DFB
def dfbdec_layer(x, f, n):
    f = dfb_filter(x.device)

    if n == 1:
        y = [None] * 2
        # Simplest case, one level
        y[0], y[1] = fbdec_layer(x, f, 'q', '1r', 'qper_col')
    elif n >= 2:
        y = [None] * 4
        x0, x1 = fbdec_layer(x, f, 'q', '1r', 'qper_col')
        # y[1], y[0] = fbdec_layer(x0, f, 'q', '2c', 'per')
        # y[3], y[2] = fbdec_layer(x1, f, 'q', '2c', 'per')
        odd_list, even_list = new_fbdec_layer([x0, x1], f, 'q', '2c', 'per')
        # y[1], y[2] = odd_list
        # y[0], y[3] = even_list
        for ix in range(len(odd_list)):
            y[ix*2+1], y[ix*2] = odd_list[ix], even_list[ix]
        
        # Now expand the rest of the tree
        for l in range(3,n+1):
            # Allocate space for the new subband outputs
            y_old = y.copy()
            y = [None] * (2**l)
            
            # The first half channels use R1 and R2
            # for k in range(1, 2 ** (l - 2)+1):
            #     i = (k - 1) % 2 + 1
            #     y[2*k-1], y[2*k-2] = fbdec_layer(y_old[k-1], f, 'p', i, 'per')
            odd = np.arange(1, 2 ** (l - 2)+1, 2)
            even = np.arange(2, 2 ** (l - 2)+1, 2)
            odd_list, even_list = new_fbdec_layer([y_old[k-1] for k in odd], f, 'p', 1, 'per')
            for ix, k in enumerate(odd):
                y[2*k-1], y[2*k-2] = odd_list[ix], even_list[ix]
            odd_list, even_list = new_fbdec_layer([y_old[k-1] for k in even], f, 'p', 2, 'per')
            for ix, k in enumerate(even):
                y[2*k-1], y[2*k-2] = odd_list[ix], even_list[ix]
                
            # The second half channels use R3 and R4
            # for k in range(2 ** (l - 2) + 1,2 ** (l - 1) + 1):
            #     i = (k - 1) % 2 + 3
            #     y[2*k-1], y[2*k-2] = fbdec_layer(y_old[k-1], f, 'p', i, 'per')
            odd += 2 ** (l - 2)
            even += 2 ** (l - 2)
            odd_list, even_list = new_fbdec_layer([y_old[k-1] for k in odd], f, 'p', 3, 'per')
            for ix, k in enumerate(odd):
                y[2*k-1], y[2*k-2] = odd_list[ix], even_list[ix]
            odd_list, even_list = new_fbdec_layer([y_old[k-1] for k in even], f, 'p', 4, 'per')
            for ix, k in enumerate(even):
                y[2*k-1], y[2*k-2] = odd_list[ix], even_list[ix]
    
    # Backsampling
    def backsamp(y=None):
        n = np.log2(len(y))
        
        assert not (n != round(n) or n < 1), 'Input must be a cell vector of dyadic length'
        n=int(n)
        if n == 1:
            # One level, the decomposition filterbank shoud be Q1r
            # Undo the last resampling (Q1r = R2 * D1 * R3)
            for k in range(2):
                y[k]=resamp(y[k],4)
                y[k][..., ::2]=resamp(y[k][..., ::2], 1)
                y[k][..., 1::2]=resamp(y[k][..., 1::2],1)
        
        if n > 2:
            N=2 ** (n - 1)
                
            for k in range(1, 2 ** (n - 2) +1):
                shift = 2 * k - (2 ** (n - 2) + 1)
                
                # The first half channels
                # y[2*k - 2]=resamp(y[2*k - 2],3,shift)
                # y[2*k - 1]=resamp(y[2*k - 1],3,shift)
                y[2*k - 2], y[2*k - 1] = new_resamp([y[2*k - 2], y[2*k - 1]],3,shift)
                
                # The second half channels
                # y[2*k - 2 + N]=resamp(y[2*k - 2 + N],1,shift)
                # y[2*k - 1 + N]=resamp(y[2*k - 1 + N],1,shift)
                y[2*k - 2 + N], y[2*k - 1 + N] = new_resamp([y[2*k - 2 + N], y[2*k - 1 + N]],1,shift)

        return y
    y=backsamp(y)
    
    # Flip the order of the second half channels
    y[2 ** (n - 1):]=y[-1:2 ** (n - 1)-1:-1]

    return y

### TODO DFB
def new_resamp(y, type_, shift=1):
    sample = len(y)
    y = torch.stack(y)
    # print(y.get_shape().as_list())
    if type_ in [3,4]:
        y = torch.transpose(y, 3, 4)

    # m,n,c=y.get_shape().as_list()[-3:]
    N,c,m,n = y.shape[1:]

    # y = tf.reshape(y, [sample, -1, m*n, c])
    y = torch.reshape(y, [sample, -1, c, m*n])

    z=np.zeros((m,n), dtype=np.int64)
    for j in range(n):
        if type_ in [1,3]:
            k= (shift * j) % m
            
        else:
            k= (-shift * j) % m
            
        if k < 0:
            k=k + m
        t1 = np.arange(k, m)
        t2 = np.arange(k)
        z[:,j] = np.concatenate([t1, t2]) * n + j
        
    z = z.reshape(-1)
    z = torch.from_numpy(z) # LongTensor int64
    z = z.to(y.device)

    z = torch.reshape(z, (1,1,1,-1))
    y = torch.gather(y, 3, z.expand(sample,N,c,-1))

    # y = tf.gather(y, z.astype(int), axis=-2)
    # y = Reshape((m,n,c))(y)
    # y = tf.reshape(y, [sample, -1, m, n, c])
    y = torch.reshape(y, [sample, -1, c, m, n])

    if type_ in [3,4]:
        y = torch.transpose(y, 3, 4)
    y = [y[i] for i in range(sample)]
    return y

def fbdec_layer(x, f_, type1, type2, extmod='per'):

    # Polyphase decomposition of the input image
    if type1 == 'q':
        # Quincunx polyphase decomposition
        p0,p1 = qpdec_layer(x,type2)
        
    elif type1 == 'p':
        # Parallelogram polyphase decomposition
        p0,p1 = ppdec_layer(x,type2)
    
    # Ladder network structure
    y0 = 1 / (2**0.5) * (p0 - sefilter2_layer(p1,f_,f_,extmod,[1,1]))
    y1 = (-2**0.5)*p1 - sefilter2_layer(y0,f_,f_,extmod)
    
    return [y0, y1]

def qpdec_layer(x, type_='1r'):

    if type_ == '1r':   # Q1 = R2 * D1 * R3
        y = resamp(x, 2)

        # p0 = resamp(y[:,::2,:,:], 3)
        
        # inv(R2) * [0; 1] = [1; 1]
        # p1 = resamp(y(2:2:end, [2:end, 1]), 3)
        p1 = torch.cat([y[..., 1::2,1:], y[..., 1::2,0:1]], dim=3)
        # p1 = resamp(p1, 3)
        p0, p1 = new_resamp([y[...,::2,:], p1], 3)

    elif type_ == '1c': # Q1 = R3 * D2 * R2
        # TODO
        y=resamp(x,3)
        
        # p0=resamp(y[:,:,::2,:],2)
        p0=resamp(y[...,::2],2)
        
        # inv(R3) * [0; 1] = [0; 1]
        # p1=resamp(y[:,:,1::2,:],2)
        p1=resamp(y[...,1::2],2)
            
    elif type_ == '2r': # Q2 = R1 * D1 * R4
        # TODO
        y=resamp(x,1)
        
        p0=resamp(y[...,::2,:],4)
        
        # inv(R1) * [1; 0] = [1; 0]
        p1=resamp(y[...,1::2,:],4)
        
    elif type_ == '2c': # Q2 = R4 * D2 * R1
        y = resamp(x,4)
        
        # p0=resamp(y[:,:,::2,:],1)
        
        # inv(R4) * [1; 0] = [1; 1]
        # p1 = resamp(y([2:end, 1], 2:2:end), 1)
        p1 = torch.cat([y[...,1:,1::2], y[...,0:1,1::2]], dim=2)
        # p1 = resamp(p1,1)
        # p0, p1 = new_resamp([y[:,:,::2,:], p1], 1)
        p0, p1 = new_resamp([y[...,::2], p1], 1)
        
    else:
        raise ValueError('Invalid argument type')

    return p0, p1

def ppdec_layer(x, type_):
    # TODO
    if type_ == 1:      # P1 = R1 * Q1 = D1 * R3
        # p0=resamp(x[:,::2,:,:],3)

        # R1 * [0; 1] = [1; 1]
        #p1=resamp(np.roll(x[1::2,:],-1,axis=1),3)
        p1 = torch.cat([x[...,1::2,1:], x[...,1::2,0:1]], dim=3)
        # p1=resamp(p1, 3)
        # p0, p1 = new_resamp([x[:,::2,:,:], p1], 3)
        p0, p1 = new_resamp([x[...,::2,:], p1], 3)
        
    elif type_ == 2:    # P2 = R2 * Q2 = D1 * R4
        # p0=resamp(x[:,::2,:,:],4)
        
        # R2 * [1; 0] = [1; 0]
        # p1=resamp(x[:,1::2,:,:],4)
        # p0, p1 = new_resamp([x[:,::2,:,:], x[:,1::2,:,:]], 4)
        p0, p1 = new_resamp([x[...,::2,:], x[...,1::2,:]], 4)
        
    elif type_ == 3:    # P3 = R3 * Q2 = D2 * R1
        # p0=resamp(x[:,:,::2,:],1)
        
        # R3 * [1; 0] = [1; 1]
        #p1=resamp(np.roll(x[:,1::2],-1,axis=0),1)
        # p1 = torch.cat([x[:,1:,1::2,:], x[:,0:1,1::2,:]], dim=1)
        p1 = torch.cat([x[...,1:,1::2], x[...,0:1,1::2]], dim=2)
        # p1=resamp(p1, 1)
        # p0, p1 = new_resamp([x[:,:,::2,:], p1], 1)
        p0, p1 = new_resamp([x[...,::2], p1], 1)
        
    elif type_ == 4:    # P4 = R4 * Q1 = D2 * R2
        # p0=resamp(x[:,:,::2,:],2)
        
        # R4 * [0; 1] = [0; 1]
        # p1=resamp(x[:,:,1::2,:],2)
        # p0, p1 = new_resamp([x[:,:,::2,:], x[:,:,1::2,:]], 2)
        p0, p1 = new_resamp([x[...,::2], x[...,1::2]], 2)
        
    else:
        raise ValueError('Invalid argument type')
    
    return p0, p1

def resamp(x, type_, shift=1,extmod='per'):
    if type_ in [1,2]:
        y=resampm(x,type_,shift)
        
    elif type_ in [3,4]:
        y = torch.transpose(x, 2, 3)
        y = resampm(y, type_-2, shift)
        y = torch.transpose(y, 2, 3)
        
    else:
        raise ValueError('The second input (type) must be one of {1, 2, 3, 4}')

    return y

total = 0
def resampm(x, type_, shift=1):
    tic = time()
    N,c,m,n=x.shape

    x = torch.reshape(x, [-1, c, m*n])

    z=np.zeros((m,n), dtype=np.int64)
    for j in range(n):
        if type_ == 1:
            k= (shift * j) % m
            
        else:
            k= (-shift * j) % m
            
        if k < 0:
            k=k + m
        t1 = np.arange(k, m)
        t2 = np.arange(k)
        z[:,j] = np.concatenate([t1, t2]) * n + j
        
    z = z.reshape(-1)
    z = torch.from_numpy(z)
    z = z.to(x.device)

    # y = tf.gather(x, z.astype(int), axis=1)
    # y = tf.reshape(y, [-1, m, n, c])
    z = z.reshape((1,1,-1))
    y = torch.gather(x, 2, z.expand(N,c,-1))
    y = torch.reshape(y, [-1, c, m, n])
    
    toc = time()
    global total
    total += (toc-tic)
    # print('This resamp takes:', toc-tic, 'sec. Current time cost on resamp:', total)
    return y

def pdfbrec_layer(xlo, xhi, pfilt=None, dfilt=None):
    
    xhi = dfbrec_layer(xhi)

    x = lprec_layer(xlo, xhi)
    return x

def new_fbrec_layer(y0,y1,f_,type1,type2,extmod='per'):

    sample = len(y0)
    y0 = torch.cat(y0, axis=1)
    y1 = torch.cat(y1, axis=1)
    ch = y0.shape[1] // sample

    p1 = -1 / (2**0.5) * (y1 + sefilter2_layer(y0,f_,f_,extmod))
    
    p0 = (2**0.5) * y0 + sefilter2_layer(p1,f_,f_,extmod,[1,1])
    
    # Polyphase reconstruction
    if type1 == 'q':
        # Quincunx polyphase reconstruction
        x = qprec_layer(p0,p1,type2)
        
    elif type1 == 'p':
        # Parallelogram polyphase reconstruction
        x = pprec_layer(p0,p1,type2)
        
    else:
        raise ValueError('Invalid argument type1')
    
    return [x[:,i*ch:(i+1)*ch] for i in range(sample)]

def dfbrec_layer(y):
    f = dfb_filter(y[0].device)

    if type(y) is not list:
        dir = y.shape[1] // 3
        y = [y[..., sp*3:(sp+1)*3] for sp in range(dir)]

    n = np.log2(len(y))
    n = int(n)

    # Flip back the order of the second half channels
    y[2 ** (n - 1):]=y[-1:2 ** (n - 1)-1:-1]

    # Undo backsampling
    def rebacksamp(y=None):
        # Number of decomposition tree levels
        n=np.log2(len(y))
        assert not (n != round(n) or n < 1), 'Input must be a cell vector of dyadic length'
        n=int(n)
        if n == 1:
            # One level, the reconstruction filterbank shoud be Q1r
            # Redo the first resampling (Q1r = R2 * D1 * R3)
            for k in range(2):
                y[k][...,::2]=resamp(y[k][...,::2],2)
                y[k][...,1::2]=resamp(y[k][...,1::2],2)
                y[k]=resamp(y[k],3)
        if n > 2:
            N=2 ** (n - 1)
            
            for k in range(1,2 ** (n - 2)+1):
                shift=2*k - (2 ** (n - 2) + 1)
                
                # The first half channels
                # y[2*k - 2]=resamp(y[2*k - 2],3,-shift)
                # y[2*k - 1]=resamp(y[2*k - 1],3,-shift)
                y[2*k - 2], y[2*k - 1] = new_resamp([y[2*k - 2], y[2*k - 1]],3,-shift)
                
                # The second half channels
                # y[2*k - 2 + N]=resamp(y[2*k - 2 + N],1,-shift)
                # y[2*k - 1 + N]=resamp(y[2*k - 1 + N],1,-shift)
                y[2*k - 2 + N], y[2*k - 1 + N] = new_resamp([y[2*k - 2 + N], y[2*k - 1 + N]],1,-shift)
        
        return y
    y=rebacksamp(y)
    
    if n == 1:
        # Simplest case, one level
        x=fbrec_layer(y[0],y[1],f,'q','1r','qper_col')

    elif n >= 2:
        for l in range(n,2,-1):
            y_old=y.copy()
            y=[None] * (2 ** (l - 1))
            
            # The first half channels use R1 and R2
            # for k in range(1,2 ** (l - 2)+1):
            #     i=(k - 1) % 2 + 1
            #     y[k-1]=fbrec_layer(y_old[2*k-1],y_old[2*k-2],f,'p',i,'per')
            odd = np.arange(1, 2 ** (l - 2)+1, 2)
            even = np.arange(2, 2 ** (l - 2)+1, 2)
            tmp = new_fbrec_layer([y_old[2*k-1] for k in odd],[y_old[2*k-2] for k in odd],f,'p',1,'per')
            for idx, k in enumerate(odd):
                y[k-1] = tmp[idx]

            tmp = new_fbrec_layer([y_old[2*k-1] for k in even],[y_old[2*k-2] for k in even],f,'p',2,'per')
            for idx, k in enumerate(even):
                y[k-1] = tmp[idx]

            # The second half channels use R3 and R4
            # for k in range(2 ** (l - 2) + 1,2 ** (l - 1)+1):
            #     i=(k - 1) % 2 + 3
            #     y[k-1]=fbrec_layer(y_old[2*k-1],y_old[2*k-2],f,'p',i,'per')
            odd += 2 ** (l - 2)
            even += 2 ** (l - 2)
            tmp = new_fbrec_layer([y_old[2*k-1] for k in odd],[y_old[2*k-2] for k in odd],f,'p',3,'per')
            for idx, k in enumerate(odd):
                y[k-1] = tmp[idx]

            tmp = new_fbrec_layer([y_old[2*k-1] for k in even],[y_old[2*k-2] for k in even],f,'p',4,'per')
            for idx, k in enumerate(even):
                y[k-1] = tmp[idx]

        # x0 = fbrec_layer(y[1],y[0],f,'q','2c','per')
        # x1 = fbrec_layer(y[3],y[2],f,'q','2c','per')
        x0, x1 = new_fbrec_layer([y[1], y[3]],[y[0], y[2]],f,'q','2c','per')

        # First level
        x = fbrec_layer(x0,x1,f,'q','1r','qper_col')
    
    return x

def fbrec_layer(y0,y1,f_,type1,type2,extmod='per'):

    p1 = -1 / (2**0.5) * (y1 + sefilter2_layer(y0,f_,f_,extmod))
    
    p0 = (2**0.5) * y0 + sefilter2_layer(p1,f_,f_,extmod,[1,1])
    
    # Polyphase reconstruction
    if type1 == 'q':
        # Quincunx polyphase reconstruction
        x = qprec_layer(p0,p1,type2)
        
    elif type1 == 'p':
        # Parallelogram polyphase reconstruction
        x = pprec_layer(p0,p1,type2)
        
    else:
        raise ValueError('Invalid argument type1')
    
    return x
    
def slice_2c(y):
    idx = []
    n = y.shape[3]//2
    for i in range(n):
        idx.extend([i, i+n])
    # y = tf.gather(y, idx, axis=2)
    idx = torch.tensor(idx, dtype=torch.long, device=y.device)
    N,C,H,W = y.shape
    idx = torch.reshape(idx, (1,1,1,-1))
    y = torch.gather(y, 3, idx.expand(N,C,H,-1))
    return y

def slice_1r(y):
    idx = []
    m = y.shape[2]//2
    for i in range(m):
        idx.extend([i, i+m])
    # y = tf.gather(y, idx, axis=1)
    idx = torch.tensor(idx, dtype=torch.long, device=y.device)
    N,C,H,W = y.shape
    idx = torch.reshape(idx, (1,1,-1,1))
    y = torch.gather(y, 2, idx.expand(N,C,-1,W))
    return y

def qprec_layer(p0, p1, type_='1r'):
    m,n = p0.shape[1:3]
    
    if type_ == '1r':   # Q1 = R2 * D1 * R3
        # y1 = resamp(p0,4)
        # y2 = resamp(p1,4)
        y1, y2 = new_resamp([p0, p1], 4)
        y2 = torch.cat([y2[...,-1:], y2[...,:-1]], dim=3)
        y = torch.cat([y1, y2], dim=2)
        
        y = slice_1r(y)
        x = resamp(y,1)
        
    elif type_ == '1c': # Q1 = R3 * D2 * R2
        # TODO
        y=np.zeros((m,2*n))
        y[:,::2]=resamp(p0,1)
        y[:,1::2]=resamp(p1,1)
        
        x=resamp(y,4)

    elif type_ == '2r': # Q2 = R1 * D1 * R4
        # TODO
        y=np.zeros((2*m,n))
        y[::2,:]=resamp(p0,3)
        y[1::2,:]=resamp(p1,3)
        
        x=resamp(y,2)
        
    elif type_ == '2c': # Q2 = R4 * D2 * R1

        # y1 = resamp(p0,2)
        # y2 = resamp(p1,2)
        y1, y2 = new_resamp([p0, p1], 2)
        y2 = torch.cat([y2[:,:,-1:,:], y2[:,:,:-1,:]], dim=2)
        y = torch.cat([y1, y2], dim=3)

        y = slice_2c(y)
        x = resamp(y,3)
        
    else:
        raise ValueError('Invalid argument type')
    
    return x

def pprec_layer(p0, p1, type_):

    if type_ == 1:      # P1 = R1 * Q1 = D1 * R3
        '''x=np.zeros((2*m,n))
        x[::2,:]=resamp(p0,4)
        x[1::2,:]=np.roll(resamp(p1,4),1,axis=1)  # double check'''
        
        # x1 = resamp(p0,4)
        # x2 = resamp(p1,4)
        x1, x2 = new_resamp([p0, p1], 4)
        x2 = torch.cat([x2[...,-1:], x2[...,:-1]], dim=3)
        x = torch.cat([x1, x2], dim=2)
        
        # x = Lambda(lambda y: slice_1r(y))(x)
        x = slice_1r(x)
        
    elif type_ == 2:    # P2 = R2 * Q2 = D1 * R4
        '''x=np.zeros((2*m,n))
        x[::2,:]=resamp(p0,3)
        x[1::2,:]=resamp(p1,3)'''
        
        # x1 = resamp(p0,3)
        # x2 = resamp(p1,3)
        x1, x2 = new_resamp([p0, p1], 3)
        x = torch.cat([x1, x2], dim=2)

        x = slice_1r(x)
        
    elif type_ == 3:    # P3 = R3 * Q2 = D2 * R1
        '''x=np.zeros((m,2*n))
        x[:,::2]=resamp(p0,2)
        x[:,1::2]=np.roll(resamp(p1,2),1,axis=0)  # double check'''

        # x1 = resamp(p0,2)
        # x2 = resamp(p1,2)
        x1, x2 = new_resamp([p0, p1], 2)
        x2 = torch.cat([x2[:,:,-1:,:], x2[:,:,:-1,:]], dim=2)
        x = torch.cat([x1, x2], dim=3)

        x = slice_2c(x)
        
    elif type_ == 4:    # P4 = R4 * Q1 = D2 * R2
        '''x=np.zeros((m,2*n))
        x[:,::2]=resamp(p0,1)
        x[:,1::2]=resamp(p1,1)'''

        # x1 = resamp(p0,1)
        # x2 = resamp(p1,1)
        x1, x2 = new_resamp([p0, p1], 1)
        x = torch.cat([x1, x2], dim=3)

        x = slice_2c(x)
        
    else:
        raise ValueError('Invalid argument type')
    
    return x

def pixelShuffle(x, h_factor=2, w_factor=2):
    N, C, H, W = x.size()
    x = x.view(N, h_factor, w_factor, C // (h_factor * w_factor), H, W)  # (N, bs, bs, C//bs^2, H, W)
    x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
    x = x.view(N, C // (h_factor * w_factor), H * h_factor, W * w_factor)  # (N, C//bs^2, H * bs, W * bs)
    return x

def inv_pixelShuffle(x, h_factor=2, w_factor=2):
    N, C, H, W = x.size()
    x = x.view(N, C, H // h_factor, h_factor, W // w_factor, w_factor)  # (N, C, H//bs, bs, W//bs, bs)
    x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
    x = x.view(N, C * (h_factor * w_factor), H // h_factor, W // w_factor)  # (N, C*bs^2, H//bs, W//bs)
    return x

def show_coeff(show):
    plot = True
    if plot:
        import matplotlib.pyplot as plt
        # show = c.cpu().numpy()
        if isinstance(show, list):
            dir = len(show)
            sub = 2 if dir==2 else 4
            for idx in range(sub):
                x = show[idx]
                x = x.cpu().numpy()
                if show[0].ndim>=4:
                    x = x[0]
                if show[0].shape[1] in [1,3,4]:
                    x = np.transpose(x, [1,2,0]) * 10
                
                plt.subplot(2,sub>>1,idx+1)
                plt.imshow(x)
        
        else:
            x = show.cpu().numpy()
            if show.ndim>=4:
                x = x[0]
            C = 3 if show.shape[1] >= 3 else 1
            x = np.transpose(x[:C], [1,2,0]) * 2
            plt.imshow(x)
        plt.show()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    im = cv2.imread("contourlet_python/zoneplate.png")
    im = cv2.imread("contourlet_python/norain-99.png")
    im = cv2.resize(im, (480,320))
    im = np.expand_dims(im, axis=0)
    im = np.concatenate([im]*5, axis=0)

    im = np.transpose(im, [0,3,1,2])
    gt = im.copy()
    norm = 255
    im = np.float32(im) / norm

    im = torch.from_numpy(im)
    im = im.to(device)

    test = SpaceToDepth(w_factor=1)(im)
    test = DepthToSpace(w_factor=1)(test)
    print(torch.equal(test, im))

    tic = time()
    nlevs = 4
    xlo, xhi = ContourDec(nlevs)(im)
    xhi_npy = [x.cpu().numpy() for x in xhi]

    print(time()-tic)
    if len(xhi) > 4:
        def test_PS(hi_1, h_factor, w_factor):
            hi_1 = torch.cat(hi_1, dim=1)
            tmp = DepthToSpace(h_factor=h_factor, w_factor=w_factor)(hi_1)
            print(tmp.shape)
            show_coeff(tmp)
            tmp = SpaceToDepth(h_factor=h_factor, w_factor=w_factor)(tmp)
            print(torch.equal(hi_1, tmp))
        ix = len(xhi)>>2
        test_PS(xhi[0:ix], ix, 1)
        test_PS(xhi[ix:2*ix], ix, 1)
        test_PS(xhi[2*ix:3*ix], 1, ix)
        test_PS(xhi[3*ix:4*ix], 1, ix)
        
    show_coeff(xhi)
    tic = time()
    rec = ContourRec()([xlo, xhi])

    print(time()-tic)

    tmp = im.cpu().numpy()[0,0]
    
    cmp_np = False
    if cmp_np:
        import sys
        sys.path.append("contourlet_python")
        from pdfbdec import pdfbdec
        pfilter='9-7'      # Pyramidal filter
        dfilter='pkva'      # Directional filter
        
        coeffs=pdfbdec(tmp,pfilter,dfilter,[nlevs])

        for x in range(len(xhi_npy)):
            print(coeffs[1][x].shape, xhi_npy[x][0,0].shape)
            coeffs[1][x] = np.uint8(coeffs[1][x]*100)
            xhi_npy[x][0,0] = np.uint8(xhi_npy[x][0,0]*100)
            print((coeffs[1][x]==xhi_npy[x][0,0]).all())
            # print(np.isclose(coeffs[1][x], xhi_npy[x][0,0], rtol=1e-05, atol=1e-08, equal_nan=False).all())

    # show_coeff(xlo)

    rec = rec.cpu().numpy()
    rec = np.uint8(np.round(rec*norm))
    print((rec==gt).all())

    c, d = lpdec_layer(im)

    rec = lprec_layer(c, d)
    rec = rec.cpu().numpy()
    rec = np.uint8(np.round(rec*norm))
    print((rec==gt).all())

    