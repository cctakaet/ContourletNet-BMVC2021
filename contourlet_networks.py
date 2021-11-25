import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from contourlet_torch import DepthToSpace, SpaceToDepth, ContourDec, ContourRec, dfbrec_layer, lprec_layer

class ContourGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                    n_blocks=6, padding_type='reflect', nlevs=[4], pix_shuffle=False, refine=False):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ContourGenerator, self).__init__()
        print("init ContourGenerator")
        self.pix_shuffle = pix_shuffle
        param_dict = {
            'norm_layer': norm_layer,
            'use_dropout': use_dropout
        }
        self.sub_models = 5
        self.decomposition = nn.ModuleList([ContourDec(nlev) for nlev in nlevs])
        high_block = int(6/len(self.decomposition))
        print(high_block)
        if pix_shuffle:
            self.DepthToH = nn.ModuleList([DepthToSpace(h_factor=2**(nlev-2), w_factor=1) for nlev in nlevs])
            self.DepthToW = nn.ModuleList([DepthToSpace(h_factor=1, w_factor=2**(nlev-2)) for nlev in nlevs])

            self.refineL = LowGenerator(input_nc, output_nc, ngf, **param_dict, n_blocks=6)
            high_model = [
                HighGenerator(input_nc*7, output_nc*4, ngf, **param_dict, n_blocks=6)
            ]
            for i in range(1, len(self.decomposition)):
                high_model += [
                    HighGenerator(input_nc*7, output_nc*4, ngf, **param_dict, n_blocks=4)
                ]
            self.refineH = nn.ModuleList(high_model)

            self.HtoDepth = nn.ModuleList([SpaceToDepth(h_factor=2**(nlev-2), w_factor=1) for nlev in nlevs])
            self.WtoDepth = nn.ModuleList([SpaceToDepth(h_factor=1, w_factor=2**(nlev-2)) for nlev in nlevs])
        else:
            # raise NotImplementedError("Try pixel shuffle...")
            self.refineL = LowGenerator(input_nc, output_nc, ngf, **param_dict, n_blocks=6)
            high_model = [
                HighGenerator(input_nc<<(nlevs[0]-1), output_nc<<(nlevs[0]-1), ngf, **param_dict,
                                n_blocks=6, n_downsampling=0, stride=(2**(nlevs[0]-2), 1)),
                HighGenerator(input_nc<<(nlevs[0]-1), output_nc<<(nlevs[0]-1), ngf, **param_dict,
                                n_blocks=6, n_downsampling=0, stride=(1, 2**(nlevs[0]-2)))
            ]
            for i in range(1, len(self.decomposition)):
                high_model += [
                    HighGenerator(input_nc<<(nlevs[i]-1), output_nc<<(nlevs[i]-1), ngf, **param_dict,
                                n_blocks=4, n_downsampling=0, stride=(2**(nlevs[i]-2), 1)),
                    HighGenerator(input_nc<<(nlevs[i]-1), output_nc<<(nlevs[i]-1), ngf, **param_dict,
                                n_blocks=4, n_downsampling=0, stride=(1, 2**(nlevs[i]-2)))
            ]
            self.refineH = nn.ModuleList(high_model)
                
        self.reconstruct = ContourRec()
        self.refine = refine
        print("refine:", refine)
        if self.refine:
            self.enhancer1 = Enhancer(6, 3)
            self.enhancer2 = Enhancer(6, 3)
        # self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        xlo = input
        xlo_list, xhi_list = [], []
        refine_list = []
        for dec in self.decomposition:
            xlo, xhi = dec(xlo)
            xlo_list.append(xlo)
            xhi_list.append(xhi)
            refine_list.append([xlo])
                
        # for ix, toH in enumerate(self.DepthToH):
        for ix in range(len(self.decomposition)):
            toH = self.DepthToH[ix] if self.pix_shuffle else lambda x: x
            toW = self.DepthToW[ix] if self.pix_shuffle else lambda x: x
            combine = len(xhi_list[ix])>>2
            tmp1 = torch.cat(xhi_list[ix][:combine], dim=1)
            tmp1 = toH(tmp1)
            refine_list[ix].append(tmp1)
            tmp2 = torch.cat(xhi_list[ix][combine:2*combine], dim=1)
            tmp2 = toH(tmp2)
            refine_list[ix].append(tmp2)
            tmp3 = torch.cat(xhi_list[ix][2*combine:3*combine], dim=1)
            tmp3 = toW(tmp3)
            refine_list[ix].append(tmp3)
            tmp4 = torch.cat(xhi_list[ix][3*combine:4*combine], dim=1)
            tmp4 = toW(tmp4)
            refine_list[ix].append(tmp4)
        
        bands_copy = [None] * len(self.decomposition) # copy from graph for loss  
        lap_copy = [None] * len(self.decomposition) # copy from graph for loss  
        for lv in range(-1, -(len(self.decomposition)+1), -1):
            if lv == -1:
                refine_list[lv][0] = self.refineL(refine_list[lv][0])
            # refine H
            if self.pix_shuffle:
                refine_list[lv][1:] = self.refineH[lv](refine_list[lv][1:], [refine_list[lv][0], xlo_list[lv]])
            else:
                refine_list[lv][1:3] = self.refineH[lv*2](refine_list[lv][1:3], [refine_list[lv][0], xlo_list[lv]])
                refine_list[lv][3:] = self.refineH[lv*2+1](refine_list[lv][3:], [refine_list[lv][0], xlo_list[lv]])
            toH = self.HtoDepth[lv] if self.pix_shuffle else lambda x: x
            toW = self.WtoDepth[lv] if self.pix_shuffle else lambda x: x
            combine = len(xhi_list[lv])>>2
            tmp1 = toH(refine_list[lv][1])
            C = tmp1.shape[1] // combine
            # print(tmp1.shape, refine_list[lv][1].shape)
            # print(len(xhi_list[lv]), [xhi_list[lv][aaa].shape for aaa in range(len(xhi_list[lv]))])
            xhi_list[lv][:combine] = [tmp1[:,N*C:(N+1)*C] for N in range(combine)]
            
            tmp2 = toH(refine_list[lv][2])
            C = tmp2.shape[1] // combine
            xhi_list[lv][combine:2*combine] = [tmp2[:,N*C:(N+1)*C] for N in range(combine)]

            tmp3 = toW(refine_list[lv][3])
            C = tmp3.shape[1] // combine
            xhi_list[lv][2*combine:3*combine] = [tmp3[:,N*C:(N+1)*C] for N in range(combine)]
            
            tmp4 = toW(refine_list[lv][4])
            C = tmp4.shape[1] // combine
            xhi_list[lv][3*combine:4*combine] = [tmp4[:,N*C:(N+1)*C] for N in range(combine)]
            # print(len(xhi_list[lv]), [xhi_list[lv][aaa].shape for aaa in range(len(xhi_list[lv]))])
            # reconstruct
            # bands_copy[lv] = [refine_list[lv][0][:,:3].clone()]
            # bands_copy[lv] += [band[:,:3].clone() for band in xhi_list[lv]]
            bands_copy[lv] = [band[:,:3].clone() for band in xhi_list[lv]]
            xhi = dfbrec_layer(xhi_list[lv])
            lap_copy[lv] = [refine_list[lv][0][:,:3].clone(), xhi[:,:3].clone()]
            if lv == -len(self.decomposition):
                rec_img = self.reconstruct([refine_list[lv][0], xhi_list[lv]])
                # rain_streak = lprec_layer(refine_list[lv][0], xhi)
            else:
                refine_list[lv-1][0] = self.reconstruct([refine_list[lv][0], xhi_list[lv]])
                # refine_list[lv-1][0] = lprec_layer(refine_list[lv][0], xhi)
        # rec_img = torch.cat([input-rain_streak, rain_streak], dim=1)
        if self.refine:
            tmp = torch.cat([input, rec_img[:,:3]], dim=1)
            tmp = self.enhancer1(tmp)
            tmp = torch.cat([tmp, rec_img[:,:3]], dim=1)
            tmp = self.enhancer2(tmp)
            # rec_img = torch.cat([tmp, rec_img[:,3:]], dim=1)
            rec_img[:,:3] = tmp+rec_img[:,:3]
        return rec_img, bands_copy, lap_copy
        # return rec_img, lap_copy

class LowGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', high_pass=False):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(LowGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            # use_bias = norm_layer == nn.InstanceNorm2d
            use_bias = True # zzx (if no normalization, we need bias)

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.Upsample(scale_factor=2),
                      nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, padding=1, bias=use_bias),
                      # nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                      #                  kernel_size=3, stride=2,
                      #                  padding=1, output_padding=1,
                      #                  bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        # ZZX: We remove the nonlinear activation on the last layer of our separator since we
        # found it may slow-down the convergence.
        # model += [nn.Tanh()]
        # model += [nn.Sigmoid()]
        # model += [nn.ReLU()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class HighGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', n_downsampling=2, stride=1):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(HighGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            # use_bias = norm_layer == nn.InstanceNorm2d
            use_bias = True # zzx (if no normalization, we need bias)

        self.in_out_ratio = int(output_nc//input_nc)
        if stride != 1:
            print(input_nc, output_nc)
            self.ref_feature = nn.Conv2d(9, ngf>>1, kernel_size=7, stride=stride, padding=3)
            self.to_feature = nn.Conv2d(input_nc, ngf>>1, kernel_size=7, padding=3)
            input_nc = ngf

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.LeakyReLU(0.3, True)]

        # n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.LeakyReLU(0.3, True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.Upsample(scale_factor=2),
                      nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, padding=1, bias=use_bias),
                      # nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                      #                  kernel_size=3, stride=2,
                      #                  padding=1, output_padding=1,
                      #                  bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.LeakyReLU(0.3, True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        # ZZX: We remove the nonlinear activation on the last layer of our separator since we
        # found it may slow-down the convergence.
        self.output_nc = output_nc
        # model += [nn.Tanh()]  # v1
        # v2: remove Tanh
        # model += [nn.Sigmoid()]
        # model += [nn.ReLU()]

        self.model = nn.Sequential(*model)
        self.stride = stride

    def forward(self, input, reference):
        """Standard forward"""
        if isinstance(reference, torch.Tensor):
            reference = [reference]
        sub_bands = len(input)
        input = torch.cat(input, dim=1)
        C = self.in_out_ratio * input.shape[1] // sub_bands
        if self.stride != 1:
            reference = torch.cat(reference, dim=1)
            reference = self.ref_feature(reference)
            x = self.to_feature(input)
            x = torch.cat([x, reference], dim=1)
        else:
            x = torch.cat([input]+reference, dim=1)
        x = self.model(x)
        rep = self.output_nc // input.shape[1]
        if rep != 1:
            x = x + torch.cat([input]*rep, dim=1)
        else:
            x = x + input
        return [x[:,ix*C:(ix+1)*C] for ix in range(sub_bands)]

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim),
                        nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class Enhancer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Enhancer, self).__init__()

        self.relu = nn.LeakyReLU(0.2, inplace=True)

        # self.tanh = nn.Tanh()
        self.tanh = nn.Hardtanh(inplace=True)

        self.refine1 = nn.Conv2d(in_channels, 20, kernel_size=3, stride=1, padding=1) 
        self.refine2 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1)

        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  

        self.refine3 = nn.Conv2d(20 + 4, out_channels, kernel_size=3, stride=1, padding=1)
        # self.upsample = F.upsample_nearest
        self.upsample = F.upsample_bilinear

        self.batch1 = nn.InstanceNorm2d(100, affine=True)

    def forward(self, x):
        dehaze = self.relu((self.refine1(x)))
        dehaze = self.relu((self.refine2(dehaze)))
        shape_out = dehaze.data.size()
        
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(dehaze, 32) 

        x102 = F.avg_pool2d(dehaze, 16)

        x103 = F.avg_pool2d(dehaze, 8)

        x104 = F.avg_pool2d(dehaze, 4)

        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze), 1) 
        dehaze = self.tanh(self.refine3(dehaze))
        # dehaze = self.relu(self.refine3(dehaze))

        return dehaze