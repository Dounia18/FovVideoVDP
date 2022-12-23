# Decimated Laplacian Pyramid
import torch
import torch.nn.functional as Func
import numpy as np 
import os
import sys
import math
from torchvision.transforms import GaussianBlur, Resize

def ceildiv(a, b):
    return -(-a // b)

class fvvdp_lpyr_dec():

    def __init__(self, W, H, ppd, device):
        self.device = device
        self.ppd = ppd
        self.min_freq = 0.5
        self.W = W
        self.H = H

        max_levels = int(np.floor(np.log2(min(self.H, self.W))))-1

        bands = np.concatenate([[1.0], np.power(2.0, -np.arange(0.0,14.0)) * 0.3228], 0) * self.ppd/2.0 

        # print(max_levels)
        # print(bands)
        # sys.exit(0)

        invalid_bands = np.array(np.nonzero(bands <= self.min_freq)) # we want to find first non0, length is index+1

        if invalid_bands.shape[-2] == 0:
            max_band = max_levels
        else:
            max_band = invalid_bands[0][0]

        # max_band+1 below converts index into count
        self.height = np.clip(max_band+1, 0, max_levels) # int(np.clip(max(np.ceil(np.log2(ppd)), 1.0)))
        self.band_freqs = np.array([1.0] + [0.3228 * 2.0 **(-f) for f in range(self.height)]) * self.ppd/2.0

        self.pyr_shape = self.height * [None] # shape (W,H) of each level of the pyramid
        self.pyr_ind = self.height * [None]   # index to the elements at each level

        cH = H
        cW = W
        for ll in range(self.height):
            self.pyr_shape[ll] = (cH, cW)
            cH = ceildiv(H,2)
            cW = ceildiv(W,2)

    def get_freqs(self):
        return self.band_freqs

    def get_band_count(self):
        return self.height+1

    def get_band(self, bands, band):
        if band == 0 or band == (len(bands)-1):
            band_mul = 1.0
        else:
            band_mul = 2.0

        return bands[band] * band_mul

    def set_band(self, bands, band, data):
        if band == 0 or band == (len(bands)-1):
            band_mul = 1.0
        else:
            band_mul = 2.0

        bands[band] = data / band_mul

    def get_gband(self, gbands, band):
        return gbands[band]

    # def get_gband_count(self):
    #     return self.height #len(gbands)

    # def clear(self):
    #     for pyramid in self.P:
    #         for level in pyramid:
    #             # print ("deleting " + str(level))
    #             del level

    def decompose(self, image): 
        # assert len(image.shape)==4, "NCHW (C==1) is expected, got " + str(image.shape)
        # assert image.shape[-2] == self.H
        # assert image.shape[-1] == self.W

        # self.image = image

        return self.laplacian_pyramid_dec(image, self.height+1)

    def reconstruct(self, bands):
        img = bands[-1]

        for i in reversed(range(0, len(bands)-1)):
            img = self.gausspyr_expand(img, [bands[i].shape[-2], bands[i].shape[-1]])
            img += bands[i]

        return img

    def laplacian_pyramid_dec(self, image, levels = -1, kernel_a = 0.4):
        gpyr = self.gaussian_pyramid_dec(image, levels, kernel_a)

        height = len(gpyr)
        if height == 0:
            return []

        lpyr = []
        for i in range(height-1):
            layer = gpyr[i] - self.gausspyr_expand(gpyr[i+1], [gpyr[i].shape[-2], gpyr[i].shape[-1]], kernel_a)
            lpyr.append(layer)

        lpyr.append(gpyr[height-1])

        # print("laplacian pyramid summary:")
        # print("self.height = %d" % self.height)
        # print("height      = %d" % height)
        # print("len(lpyr)   = %d" % len(lpyr))
        # print("len(gpyr)   = %d" % len(gpyr))
        # sys.exit(0)

        return lpyr, gpyr

    def interleave_zeros_and_pad(self, x, exp_size, dim):
        new_shape = [*x.shape]
        new_shape[dim] = exp_size[dim]+4
        z = torch.zeros( new_shape, dtype=x.dtype, device=x.device)
        odd_no = (exp_size[dim]%2)
        if dim==-2:
            z[:,:,2:-2:2,:] = x
            z[:,:,0,:] = x[:,:,0,:]
            z[:,:,-2+odd_no,:] = x[:,:,-1,:]
        elif dim==-1:
            z[:,:,:,2:-2:2] = x
            z[:,:,:,0] = x[:,:,:,0]
            z[:,:,:,-2+odd_no] = x[:,:,:,-1]
        else:
            assert False, "Wrong dimension"

        return z

    def gaussian_pyramid_dec(self, image, levels = -1, kernel_a = 0.4):

        default_levels = int(np.floor(np.log2(min(image.shape[-2], image.shape[-1]))))

        if levels == -1:
            levels = default_levels
        if levels > default_levels:
            raise Exception("Too many levels (%d) requested. Max is %d for %s" % (levels, default_levels, image.shape))

        res = [image]

        for i in range(1, levels):
            res.append(self.gausspyr_reduce(res[i-1], kernel_a))

        return res


    def sympad(self, x, padding, axis):
        if padding == 0:
            return x
        else:
            beg = torch.flip(torch.narrow(x, axis, 0,        padding), [axis])
            end = torch.flip(torch.narrow(x, axis, -padding, padding), [axis])

            return torch.cat((beg, x, end), axis)

    def get_kernels( self, im, kernel_a = 0.4 ):

        ch_dim = len(im.shape)-2
        if hasattr(self, "K_horiz") and ch_dim==self.K_ch_dim:
            return self.K_vert, self.K_horiz

        K = torch.tensor([0.25 - kernel_a/2.0, 0.25, kernel_a, 0.25, 0.25 - kernel_a/2.0], device=im.device, dtype=im.dtype)
        self.K_vert = torch.reshape(K, (1,)*ch_dim + (K.shape[0], 1))
        self.K_horiz = torch.reshape(K, (1,)*ch_dim + (1, K.shape[0]))
        self.K_ch_dim = ch_dim
        return self.K_vert, self.K_horiz
        

    def gausspyr_reduce(self, x, kernel_a = 0.4):

        K_vert, K_horiz = self.get_kernels( x, kernel_a )

        B, C, H, W = x.shape
        y_a = Func.conv2d(x.view(-1,1,H,W), K_vert, stride=(2,1), padding=(2,0)).view(B,C,-1,W)

        # Symmetric padding 
        y_a[:,:,0,:] += x[:,:,0,:]*K_vert[0,0,1,0] + x[:,:,1,:]*K_vert[0,0,0,0]
        if (x.shape[-2] % 2)==1: # odd number of rows
            y_a[:,:,-1,:] += x[:,:,-1,:]*K_vert[0,0,3,0] + x[:,:,-2,:]*K_vert[0,0,4,0]
        else: # even number of rows
            y_a[:,:,-1,:] += x[:,:,-1,:]*K_vert[0,0,4,0]

        H = y_a.shape[-2]
        y = Func.conv2d(y_a.view(-1,1,H,W), K_horiz, stride=(1,2), padding=(0,2)).view(B,C,H,-1)

        # Symmetric padding 
        y[:,:,:,0] += y_a[:,:,:,0]*K_horiz[0,0,0,1] + y_a[:,:,:,1]*K_horiz[0,0,0,0]
        if (x.shape[-2] % 2)==1: # odd number of columns
            y[:,:,:,-1] += y_a[:,:,:,-1]*K_horiz[0,0,0,3] + y_a[:,:,:,-2]*K_horiz[0,0,0,4]
        else: # even number of columns
            y[:,:,:,-1] += y_a[:,:,:,-1]*K_horiz[0,0,0,4] 

        return y

    def gausspyr_expand_pad(self, x, padding, axis):
        if padding == 0:
            return x
        else:
            beg = torch.narrow(x, axis, 0,        padding)
            end = torch.narrow(x, axis, -padding, padding)

            return torch.cat((beg, x, end), axis)

    # This function is (a bit) faster
    def gausspyr_expand(self, x, sz = None, kernel_a = 0.4):
        if sz is None:
            sz = [x.shape[-2]*2, x.shape[-1]*2]

        K_vert, K_horiz = self.get_kernels( x, kernel_a )

        y_a = self.interleave_zeros_and_pad(x, dim=-2, exp_size=sz)

        B, C, H, W = y_a.shape
        y_a = Func.conv2d(y_a.view(-1,1,H,W), K_vert*2).view(B,C,-1,W)

        y   = self.interleave_zeros_and_pad(y_a, dim=-1, exp_size=sz)
        B, C, H, W = y.shape

        y   = Func.conv2d(y.view(-1,1,H,W), K_horiz*2).view(B,C,H,-1)

        return y

    def interleave_zeros(self, x, dim):
        z = torch.zeros_like(x, device=self.device)
        if dim==2:
            return torch.cat([x,z],dim=3).view(x.shape[0], x.shape[1], 2*x.shape[2],x.shape[3])
        elif dim==3:
            return torch.cat([x.permute(0,1,3,2),z.permute(0,1,3,2)],dim=3).view(x.shape[0], x.shape[1], 2*x.shape[3],x.shape[2]).permute(0,1,3,2)


# This pyramid computes and stores contrast during decomposition, improving performance and reducing memory consumption
class fvvdp_contrast_pyr(fvvdp_lpyr_dec):

    def decompose(self, image):
        levels = self.height+1
        kernel_a = 0.4
        gpyr = self.gaussian_pyramid_dec(image, levels, kernel_a)

        height = len(gpyr)
        if height == 0:
            return []

        lpyr = []
        L_bkg_pyr = []
        for i in range(height-1):
            glayer_ex = self.gausspyr_expand(gpyr[i+1], [gpyr[i].shape[-2], gpyr[i].shape[-1]], kernel_a)
            layer = gpyr[i] - glayer_ex 

            # Order: test-sustained, ref-sustained, test-transient, ref-transient
            # L_bkg is set to ref-sustained 
            L_bkg = torch.clamp(glayer_ex[...,1:2,:,:,:], min=0.1)
            contrast = torch.clamp(torch.div(layer, L_bkg), max=1000.0)

            lpyr.append(contrast)
            L_bkg_pyr.append(L_bkg)

        lpyr.append(gpyr[height-1]) 
        
        return lpyr, L_bkg_pyr


class fvvdp_spyr_dec(fvvdp_lpyr_dec):

    def __init__(self, W, H, ppd, device):
        self.device = device
        self.ppd = ppd
        self.min_freq = 0.5
        self.W = W
        self.H = H

        self.lo0filt, self.hi0filt, self.lofilt, self.bfilts, self.steermtx, self.harmonics = self.spFilter()

        max_levels = self.max_levels(min(W, H), self.lofilt.size())

        # max_band+1 below converts index into count
        self.height = min(math.ceil(math.log2(ppd)-2), max_levels) + 1  #The 1 added is the base band
        self.band_freqs = np.array([1.0] + [0.3228 * 2.0 ** (-f) for f in range(self.height)]) * self.ppd / 2.0

        self.pyr_shape = (self.height+1) * [None]  # shape (W,H) of each level of the pyramid
        self.pyr_ind = (self.height+1) * [None]  # index to the elements at each level

        cH = H
        cW = W
        self.pyr_shape[0] = (cH, cW)
        for ll in range(self.height):
            self.pyr_shape[ll+1] = (cH, cW)
            cH = cH // 2
            cW = cW // 2


    def decompose(self, image):

        hi0 = self.conv(image, self.hi0filt, type='corr')
        lo0 = self.conv(image, self.lo0filt, type='corr')

        pyr = self.buildLevels(lo0, self.height-1)

        pyr.insert(0, hi0)  # I'm still not sure if I can simply add the base band

        sigma = 10**(-0.781367) * self.ppd
        L_adapt = self.localAdapt(image, sigma)

        L_adapt = torch.mean(L_adapt, dim=0)

        return pyr, L_adapt

    def reconstruct(self, bands):
        levels = self.height

        image = self.reconstructLevels(bands, levels)

        image = self.conv(image, self.lo0filt, type='conv') + self.conv(bands[0], self.hi0filt, type='conv')

        return image

    def localAdapt(self, image, sigma):
        kernel_size = round(sigma, 6)
        kernel_size = int(kernel_size + 1-np.mod(kernel_size, 2))
        padding = self.get_pad([kernel_size, kernel_size])

        image = torch.log(torch.clamp(image, min=1e-6))

        #image = Func.pad(image, pad=padding, mode='reflect')

        filt = GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=sigma)

        image = filt(image)

        return torch.exp(image)

    def buildLevels(self, image, levels):

        bands = []
        bfiltsz = round(math.sqrt(self.bfilts.size()[0]))
        filt = torch.reshape(self.bfilts, (bfiltsz, bfiltsz))

        for _ in range(levels):
            band = self.conv(image, filt, type='corr')
            bands.append(band)

            image = self.conv(image, self.lofilt, step=2, type='corr')

        bands.append(image)

        return bands

    def reconstructLevels(self, pyr, levels):

        res = pyr[-1]

        bfiltsz = round(math.sqrt(self.bfilts.size()[0]))
        filt = torch.reshape(self.bfilts, (bfiltsz, bfiltsz))

        for level in reversed(range(levels - 1)):
            size = pyr[level + 1].size()
            res = self.conv(res, self.lofilt, step=2, type='conv', size=size)

            band = pyr[level + 1]
            res += self.conv(band, filt, type='conv')

        return res

    def get_gband(self, gbands, band):
        return Resize(self.pyr_shape[band])(gbands)

    '''
    Determine the maximum pyramid height for a given image and filter sizes
    '''

    def max_levels(self, image_size, filter_size):

        filter_size = filter_size[0]
        levels = 0
        while True:
            if image_size < filter_size:
                break
            image_size = math.floor(image_size / 2)
            levels += 1

        return levels

    def get_pad(self, size):
        padding = []
        for k in size:
            if k % 2 == 0:
                pad = [(k - 1) // 2, (k - 1) // 2 + 1]
            else:
                pad = [(k - 1) // 2, (k - 1) // 2]
            padding.extend(pad)
        return padding
    '''
    Compute the correlation/convolution between an image with a filter, followed by a downsampling/upsampling determined by the step.
    '''

    def conv(self, image, filt, step=1, type='corr', size=None):
        # We are supposing the input image as BCHW

        padding = self.get_pad(filt.size())

        if type == 'conv':
            image = image.unsqueeze(0)
            if step>1:  # We need to apply upsampling first
                im = image
                image = torch.tensor(np.zeros((im.size()[0], size[0], size[1]), dtype=np.float32), device=self.device)
                image[:, 0::step, 0::step] = im

        image = Func.pad(image, pad=padding, mode='reflect')

        if type == 'corr':
            filt = filt.flip([0, 1])
        filt = filt.unsqueeze(0)
        filt = filt.unsqueeze(0)

        output = Func.conv2d(image, filt, padding='valid')

        if type == 'corr' and step > 1:  # We need to apply downsampling
            output = output[:, :, 0::step, 0::step]

        return output

    '''
    Get the filters used in Speerable Pyramid Algorithm
    '''

    def spFilter(self, filter='filter0'):

        if filter == 'filter0':
                harmonics = torch.tensor([
                    [0]
                ], device=self.device)

                lo0filt = torch.tensor([
                    [-4.514000e-04, - 1.137100e-04, - 3.725800e-04, - 3.743860e-03, - 3.725800e-04, - 1.137100e-04,
                     - 4.514000e-04],
                    [- 1.137100e-04, - 6.119520e-03, - 1.344160e-02, - 7.563200e-03, - 1.344160e-02, - 6.119520e-03,
                     - 1.137100e-04],
                    [- 3.725800e-04, - 1.344160e-02, 6.441488e-02, 1.524935e-01, 6.441488e-02, - 1.344160e-02,
                     - 3.725800e-04],
                    [- 3.743860e-03, - 7.563200e-03, 1.524935e-01, 3.153017e-01, 1.524935e-01, - 7.563200e-03,
                     - 3.743860e-03],
                    [- 3.725800e-04, - 1.344160e-02, 6.441488e-02, 1.524935e-01, 6.441488e-02, - 1.344160e-02,
                     - 3.725800e-04],
                    [- 1.137100e-04, - 6.119520e-03, - 1.344160e-02, - 7.563200e-03, - 1.344160e-02, - 6.119520e-03,
                     - 1.137100e-04],
                    [- 4.514000e-04, - 1.137100e-04, - 3.725800e-04, - 3.743860e-03, - 3.725800e-04, - 1.137100e-04,
                     - 4.514000e-04]
                ], device=self.device)

                lofilt = torch.tensor([
                    [-2.257000e-04, - 8.064400e-04, - 5.686000e-05, 8.741400e-04, - 1.862800e-04, - 1.031640e-03,
                     - 1.871920e-03, - 1.031640e-03, - 1.862800e-04, 8.741400e-04, - 5.686000e-05, - 8.064400e-04,
                     - 2.257000e-04],
                    [- 8.064400e-04, 1.417620e-03, - 1.903800e-04, - 2.449060e-03, - 4.596420e-03, - 7.006740e-03,
                     - 6.948900e-03, - 7.006740e-03, - 4.596420e-03, - 2.449060e-03, - 1.903800e-04, 1.417620e-03,
                     - 8.064400e-04],
                    [- 5.686000e-05, - 1.903800e-04, - 3.059760e-03, - 6.401000e-03, - 6.720800e-03, - 5.236180e-03,
                     - 3.781600e-03, - 5.236180e-03, - 6.720800e-03, - 6.401000e-03, - 3.059760e-03, - 1.903800e-04,
                     - 5.686000e-05],
                    [8.741400e-04, - 2.449060e-03, - 6.401000e-03, - 5.260020e-03, 3.938620e-03, 1.722078e-02,
                     2.449600e-02, 1.722078e-02, 3.938620e-03, - 5.260020e-03, - 6.401000e-03, - 2.449060e-03,
                     8.741400e-04],
                    [- 1.862800e-04, - 4.596420e-03, - 6.720800e-03, 3.938620e-03, 3.220744e-02, 6.306262e-02,
                     7.624674e-02, 6.306262e-02, 3.220744e-02, 3.938620e-03, - 6.720800e-03, - 4.596420e-03,
                     - 1.862800e-04],
                    [- 1.031640e-03, - 7.006740e-03, - 5.236180e-03, 1.722078e-02, 6.306262e-02, 1.116388e-01,
                     1.348999e-01, 1.116388e-01, 6.306262e-02, 1.722078e-02, - 5.236180e-03, - 7.006740e-03,
                     - 1.031640e-03],
                    [- 1.871920e-03, - 6.948900e-03, - 3.781600e-03, 2.449600e-02, 7.624674e-02, 1.348999e-01,
                     1.576508e-01, 1.348999e-01, 7.624674e-02, 2.449600e-02, - 3.781600e-03, - 6.948900e-03,
                     - 1.871920e-03],
                    [- 1.031640e-03, - 7.006740e-03, - 5.236180e-03, 1.722078e-02, 6.306262e-02, 1.116388e-01,
                     1.348999e-01, 1.116388e-01, 6.306262e-02, 1.722078e-02, - 5.236180e-03, - 7.006740e-03,
                     - 1.031640e-03],
                    [- 1.862800e-04, - 4.596420e-03, - 6.720800e-03, 3.938620e-03, 3.220744e-02, 6.306262e-02,
                     7.624674e-02, 6.306262e-02, 3.220744e-02, 3.938620e-03, - 6.720800e-03, - 4.596420e-03,
                     - 1.862800e-04],
                    [8.741400e-04, - 2.449060e-03, - 6.401000e-03, - 5.260020e-03, 3.938620e-03, 1.722078e-02,
                     2.449600e-02, 1.722078e-02, 3.938620e-03, - 5.260020e-03, - 6.401000e-03, - 2.449060e-03,
                     8.741400e-04],
                    [- 5.686000e-05, - 1.903800e-04, - 3.059760e-03, - 6.401000e-03, - 6.720800e-03, - 5.236180e-03,
                     - 3.781600e-03, - 5.236180e-03, - 6.720800e-03, - 6.401000e-03, - 3.059760e-03, - 1.903800e-04,
                     - 5.686000e-05],
                    [- 8.064400e-04, 1.417620e-03, - 1.903800e-04, - 2.449060e-03, - 4.596420e-03, - 7.006740e-03,
                     - 6.948900e-03, - 7.006740e-03, - 4.596420e-03, - 2.449060e-03, - 1.903800e-04, 1.417620e-03,
                     - 8.064400e-04],
                    [- 2.257000e-04, - 8.064400e-04, - 5.686000e-05, 8.741400e-04, - 1.862800e-04, - 1.031640e-03,
                     - 1.871920e-03, - 1.031640e-03, - 1.862800e-04, 8.741400e-04, - 5.686000e-05, - 8.064400e-04,
                     - 2.257000e-04]
                ], device=self.device)

                mtx = torch.tensor([
                    [1.000000]
                ], device=self.device)

                hi0filt = torch.tensor([
                    [5.997200e-04, - 6.068000e-05, - 3.324900e-04, - 3.325600e-04, - 2.406600e-04, - 3.325600e-04,
                     - 3.324900e-04, - 6.068000e-05, 5.997200e-04],
                    [- 6.068000e-05, 1.263100e-04, 4.927100e-04, 1.459700e-04, - 3.732100e-04, 1.459700e-04,
                     4.927100e-04, 1.263100e-04, - 6.068000e-05],
                    [- 3.324900e-04, 4.927100e-04, - 1.616650e-03, - 1.437358e-02, - 2.420138e-02, - 1.437358e-02,
                     - 1.616650e-03, 4.927100e-04, - 3.324900e-04],
                    [- 3.325600e-04, 1.459700e-04, - 1.437358e-02, - 6.300923e-02, - 9.623594e-02, - 6.300923e-02,
                     - 1.437358e-02, 1.459700e-04, - 3.325600e-04],
                    [- 2.406600e-04, - 3.732100e-04, - 2.420138e-02, - 9.623594e-02, 8.554893e-01, - 9.623594e-02,
                     - 2.420138e-02, - 3.732100e-04, - 2.406600e-04],
                    [- 3.325600e-04, 1.459700e-04, - 1.437358e-02, - 6.300923e-02, - 9.623594e-02, - 6.300923e-02,
                     - 1.437358e-02, 1.459700e-04, - 3.325600e-04],
                    [- 3.324900e-04, 4.927100e-04, - 1.616650e-03, - 1.437358e-02, - 2.420138e-02, - 1.437358e-02,
                     - 1.616650e-03, 4.927100e-04, - 3.324900e-04],
                    [- 6.068000e-05, 1.263100e-04, 4.927100e-04, 1.459700e-04, - 3.732100e-04, 1.459700e-04,
                     4.927100e-04, 1.263100e-04, - 6.068000e-05],
                    [5.997200e-04, - 6.068000e-05, - 3.324900e-04, - 3.325600e-04, - 2.406600e-04, - 3.325600e-04,
                     - 3.324900e-04, - 6.068000e-05, 5.997200e-04]
                ], device=self.device)

                bfilts = torch.tensor([[
                    -9.066000e-05, - 1.738640e-03, - 4.942500e-03, - 7.889390e-03, - 1.009473e-02, - 7.889390e-03,
                    - 4.942500e-03, - 1.738640e-03, - 9.066000e-05,
                    - 1.738640e-03, - 4.625150e-03, - 7.272540e-03, - 7.623410e-03, - 9.091950e-03, - 7.623410e-03,
                    - 7.272540e-03, - 4.625150e-03, - 1.738640e-03,
                    - 4.942500e-03, - 7.272540e-03, - 2.129540e-02, - 2.435662e-02, - 3.487008e-02, - 2.435662e-02,
                    - 2.129540e-02, - 7.272540e-03, - 4.942500e-03,
                    - 7.889390e-03, - 7.623410e-03, - 2.435662e-02, - 1.730466e-02, - 3.158605e-02, - 1.730466e-02,
                    - 2.435662e-02, - 7.623410e-03, - 7.889390e-03,
                    - 1.009473e-02, - 9.091950e-03, - 3.487008e-02, - 3.158605e-02, 9.464195e-01, - 3.158605e-02,
                    - 3.487008e-02, - 9.091950e-03, - 1.009473e-02,
                    - 7.889390e-03, - 7.623410e-03, - 2.435662e-02, - 1.730466e-02, - 3.158605e-02, - 1.730466e-02,
                    - 2.435662e-02, - 7.623410e-03, - 7.889390e-03,
                    - 4.942500e-03, - 7.272540e-03, - 2.129540e-02, - 2.435662e-02, - 3.487008e-02, - 2.435662e-02,
                    - 2.129540e-02, - 7.272540e-03, - 4.942500e-03,
                    - 1.738640e-03, - 4.625150e-03, - 7.272540e-03, - 7.623410e-03, - 9.091950e-03, - 7.623410e-03,
                    - 7.272540e-03, - 4.625150e-03, - 1.738640e-03,
                    - 9.066000e-05, - 1.738640e-03, - 4.942500e-03, - 7.889390e-03, - 1.009473e-02, - 7.889390e-03,
                    - 4.942500e-03, - 1.738640e-03, - 9.066000e-05
                ]], device=self.device).t()

        return lo0filt, hi0filt, lofilt, bfilts, mtx, harmonics

# if __name__ == '__main__':

#     device = torch.device('cuda:0')

#     torch.set_printoptions(precision=2, sci_mode=False, linewidth=300)

#     image = torch.tensor([
#         [ 1,  2,  3,  4,  5,  6,  7,  8],
#         [11, 12, 13, 14, 15, 16, 17, 18],
#         [21, 22, 23, 24, 25, 26, 27, 28],
#         [31, 32, 33, 34, 35, 36, 37, 38],
#         [41, 42, 43, 44, 45, 46, 47, 48],
#         [51, 52, 53, 54, 55, 56, 57, 58],
#         [61, 62, 63, 64, 65, 66, 67, 68],
#         [71, 72, 73, 74, 75, 76, 77, 78],
#         ], dtype=torch.float32, device=device)

#     image = image.repeat((16, 16))
#     # image = torch.cat((image, image, image), axis = -1)
#     # image = torch.cat((image, image, image), axis = -2)

#     ppd = 50

#     im_tensor = image.view(1, 1, image.shape[-2], image.shape[-1])

#     lp = fvvdp_lpyr_dec_fast(im_tensor.shape[-2], im_tensor.shape[-1], ppd, device)
#     lp_old = fvvdp_lpyr_dec(im_tensor.shape[-2], im_tensor.shape[-1], ppd, device)

#     lpyr, gpyr = lp.decompose( im_tensor )
#     lpyr_2, gpyr_2 = lp_old.decompose( im_tensor )

#     for li in range(lp.get_band_count()):
#         E = Func.mse_loss(lp.get_band(lpyr, li), lp_old.get_band(lpyr_2, li))
#         print( "Level {}, MSE={}".format(li, E))

#     import torch.utils.benchmark as benchmark

#     t0 = benchmark.Timer(
#         stmt='lp.decompose( im_tensor )',
#         setup='',
#         globals={'im_tensor': im_tensor, 'lp': lp})

#     t1 = benchmark.Timer(
#         stmt='lp_old.decompose( im_tensor )',
#         setup='',
#         globals={'im_tensor': im_tensor, 'lp_old': lp_old})

#     print("New pyramid")
#     print(t0.timeit(30))

#     print("Old pyramid")
#     print(t1.timeit(30))

#     # print("----Gaussian----")
#     # for gi in range(lp.get_band_count()):
#     #     print(lp.get_gband(gpyr, gi))

#     # print("----Laplacian----")
#     # for li in range(lp.get_band_count()):
#     #     print(lp.get_band(lpyr, li))


