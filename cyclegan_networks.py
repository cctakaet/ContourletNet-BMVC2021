"""
This part of the code is built based on the project:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""


import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import contourlet_networks as contourletNet
from contourlet_torch import DepthToSpace, SpaceToDepth, ContourDec, ContourRec, dfbrec_layer, lprec_layer

###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly recay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, nlevels, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_64':
        net = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_512':
        net = UnetGenerator(input_nc, output_nc, 9, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'dwt_refine':  # no pix shuffle
        net = contourletNet.ContourGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, nlevs=nlevels, pix_shuffle=False, refine=True)
    elif netG == 'dwt_merge':
        net = contourletNet.ContourGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, nlevs=nlevels, pix_shuffle=True)
    elif netG == 'dwt_ori':
        net = contourletNet.ContourGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, nlevs=nlevels, pix_shuffle=False)
    elif netG == 'dwt_R':
        net = DwtPredRainGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, nlevs=nlevels)
    elif netG == 'dwtPS_R':
        net = DwtPredRainGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, nlevs=nlevels, pix_shuffle=True)
    elif netG == 'dwt_H':
        net = RefineHGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, nlevs=nlevels)
    elif netG == 'dwtPS_H':
        net = RefineHGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, nlevs=nlevels, pix_shuffle=True)
    elif netG == 'dwtPS':
        net = DwtGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, nlevs=nlevels, pix_shuffle=True)
    elif 'dwt' in netG:
        net = DwtGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, nlevs=nlevels)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

class DwtPredRainGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', nlevs=[4], pix_shuffle=False):
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
        super(DwtPredRainGenerator, self).__init__()
        param_dict = {
            'norm_layer': norm_layer,
            'use_dropout': use_dropout
        }
        self.dwt = DwtGenerator(input_nc, input_nc, ngf, **param_dict, n_blocks=4, nlevs=nlevs, pix_shuffle=pix_shuffle)
        self.resnet = ResnetGenerator(output_nc, input_nc, ngf, **param_dict, n_blocks=6)

    def forward(self, input):
        """Standard forward"""
        rain = self.dwt(input)
        input = torch.cat([input, input-rain], dim=1)
        rec = self.resnet(input)
        return torch.cat([rec, rain], dim=1)

class RefineHGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                    n_blocks=6, padding_type='reflect', nlevs=[4], pix_shuffle=False):
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
        super(RefineHGenerator, self).__init__()

        param_dict = {
            'norm_layer': norm_layer,
            'use_dropout': use_dropout
        }

        self.base = ResnetGenerator(input_nc, output_nc, ngf, **param_dict, n_blocks=6)

        self.pix_shuffle = pix_shuffle
        self.sub_models = 5
        self.decomposition = nn.ModuleList([ContourDec(nlev) for nlev in nlevs])
        if pix_shuffle:
            self.DepthToH = nn.ModuleList([DepthToSpace(h_factor=2**(nlev-2), w_factor=1) for nlev in nlevs])
            self.DepthToW = nn.ModuleList([DepthToSpace(h_factor=1, w_factor=2**(nlev-2)) for nlev in nlevs])

            self.model = nn.ModuleList([
                # BandGenerator(input_nc, output_nc, ngf, **param_dict, n_blocks=6),
                BandGenerator(output_nc, output_nc, ngf, **param_dict, n_blocks=6, high_pass=True),
                BandGenerator(output_nc, output_nc, ngf, **param_dict, n_blocks=6, high_pass=True),
                BandGenerator(output_nc, output_nc, ngf, **param_dict, n_blocks=6, high_pass=True),
                BandGenerator(output_nc, output_nc, ngf, **param_dict, n_blocks=6, high_pass=True)
                ])
            self.HtoDepth = nn.ModuleList([SpaceToDepth(h_factor=2**(nlev-2), w_factor=1) for nlev in nlevs])
            self.WtoDepth = nn.ModuleList([SpaceToDepth(h_factor=1, w_factor=2**(nlev-2)) for nlev in nlevs])
        else:
            self.model = nn.ModuleList([
                # BandGenerator(input_nc, output_nc, ngf, **param_dict, n_blocks=6),
                BandGenerator(output_nc<<(nlevs[0]-2), output_nc<<(nlevs[0]-2), ngf, **param_dict, n_blocks=6, high_pass=True),
                BandGenerator(output_nc<<(nlevs[0]-2), output_nc<<(nlevs[0]-2), ngf, **param_dict, n_blocks=6, high_pass=True),
                BandGenerator(output_nc<<(nlevs[0]-2), output_nc<<(nlevs[0]-2), ngf, **param_dict, n_blocks=6, high_pass=True),
                BandGenerator(output_nc<<(nlevs[0]-2), output_nc<<(nlevs[0]-2), ngf, **param_dict, n_blocks=6, high_pass=True)
                ])
        self.reconstruct = nn.ModuleList([ContourRec() for _ in range(len(nlevs))])
        # self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        # return self.model(input)
        # return [self.model[i](input[i]) for i in range(self.sub_models)]
        base_refine = self.base(input)
        xlo = torch.cat([input, base_refine[:,:3]], dim=1)
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
                
        for lv, refine in enumerate(refine_list):
            refine_list[lv][1:] = [model(refine[ix]) for ix, model in enumerate(self.model, 1)]
                
        for ix in range(len(self.decomposition)):
            toH = self.HtoDepth[ix] if self.pix_shuffle else lambda x: x
            toW = self.WtoDepth[ix] if self.pix_shuffle else lambda x: x
            combine = len(xhi_list[ix])>>2
            tmp1 = toH(refine_list[ix][1])
            C = tmp1.shape[1] // combine
            xhi_list[ix][:combine] = [tmp1[:,N*C:(N+1)*C] for N in range(combine)]
            
            tmp2 = toH(refine_list[ix][2])
            C = tmp2.shape[1] // combine
            xhi_list[ix][combine:2*combine] = [tmp2[:,N*C:(N+1)*C] for N in range(combine)]

            tmp3 = toW(refine_list[ix][3])
            C = tmp3.shape[1] // combine
            xhi_list[ix][2*combine:3*combine] = [tmp3[:,N*C:(N+1)*C] for N in range(combine)]
            
            tmp4 = toW(refine_list[ix][4])
            C = tmp4.shape[1] // combine
            xhi_list[ix][3*combine:4*combine] = [tmp4[:,N*C:(N+1)*C] for N in range(combine)]
            
        for lv in range(1, len(self.reconstruct)+1):
            # print(xlo_list[-lv].shape, xhi_list[-lv][0].shape)
            rec_img = self.reconstruct[-lv]([xlo_list[-lv], xhi_list[-lv]])
            if lv != len(self.reconstruct):
                xlo_list[-lv-1] = rec_img
            # print(rec_img.shape)
        rec_img += base_refine
        return rec_img

class DwtGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                    n_blocks=6, padding_type='reflect', nlevs=[4], pix_shuffle=False):
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
        super(DwtGenerator, self).__init__()

        self.pix_shuffle = pix_shuffle
        param_dict = {
            'norm_layer': norm_layer,
            'use_dropout': use_dropout
        }
        self.sub_models = 5
        self.decomposition = nn.ModuleList([ContourDec(nlev) for nlev in nlevs])
        high_block = int(6/len(self.decomposition))
        print(high_block, end=' ')
        if pix_shuffle:
            self.DepthToH = nn.ModuleList([DepthToSpace(h_factor=2**(nlev-2), w_factor=1) for nlev in nlevs])
            self.DepthToW = nn.ModuleList([DepthToSpace(h_factor=1, w_factor=2**(nlev-2)) for nlev in nlevs])

            high_model = []
            for i in range(len(self.decomposition)):
                high_model += [
                    BandGenerator(input_nc, output_nc, ngf, **param_dict, n_blocks=high_block, high_pass=True) for i in range(4)
                ]
            self.model = nn.ModuleList(
                [BandGenerator(input_nc, output_nc, ngf, **param_dict, n_blocks=6)]
                + high_model
                # BandGenerator(input_nc, output_nc, ngf, **param_dict, n_blocks=6, high_pass=True),
                # BandGenerator(input_nc, output_nc, ngf, **param_dict, n_blocks=6, high_pass=True),
                # BandGenerator(input_nc, output_nc, ngf, **param_dict, n_blocks=6, high_pass=True),
                # BandGenerator(input_nc, output_nc, ngf, **param_dict, n_blocks=6, high_pass=True)
                )
            self.HtoDepth = nn.ModuleList([SpaceToDepth(h_factor=2**(nlev-2), w_factor=1) for nlev in nlevs])
            self.WtoDepth = nn.ModuleList([SpaceToDepth(h_factor=1, w_factor=2**(nlev-2)) for nlev in nlevs])
        else:
            high_model = []
            for i in range(len(self.decomposition)):
                high_model += [
                    BandGenerator(input_nc<<(nlevs[0]-2), output_nc<<(nlevs[0]-2), ngf, **param_dict, n_blocks=high_block, high_pass=True, n_downsampling=0) for i in range(4)
                ]
            self.model = nn.ModuleList(
                [BandGenerator(input_nc, output_nc, ngf, **param_dict, n_blocks=6)]
                + high_model
                # BandGenerator(input_nc<<(nlevs[0]-2), output_nc<<(nlevs[0]-2), ngf, **param_dict, n_blocks=6, high_pass=True),
                # BandGenerator(input_nc<<(nlevs[0]-2), output_nc<<(nlevs[0]-2), ngf, **param_dict, n_blocks=6, high_pass=True),
                # BandGenerator(input_nc<<(nlevs[0]-2), output_nc<<(nlevs[0]-2), ngf, **param_dict, n_blocks=6, high_pass=True),
                # BandGenerator(input_nc<<(nlevs[0]-2), output_nc<<(nlevs[0]-2), ngf, **param_dict, n_blocks=6, high_pass=True)
                )
        # self.reconstruct = nn.ModuleList([ContourRec() for _ in range(len(nlevs))])
        self.reconstruct = ContourRec()
        # self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        # return self.model(input)
        # return [self.model[i](input[i]) for i in range(self.sub_models)]
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
                
        for lv, refine in enumerate(refine_list):
            if lv+1 == len(refine_list):
                refine_list[lv][0] = self.model[0](refine[0])
            #     refine_list[lv] = [model(refine[ix]) for ix, model in enumerate(self.model)]
            # else:
            refine_list[lv][1:] = [model(refine[ix]) for ix, model in enumerate(self.model[lv*4+1:(lv+1)*4+1], 1)]

            xlo_list[lv] = refine_list[lv][0]
                
        for ix in range(len(self.decomposition)):
            toH = self.HtoDepth[ix] if self.pix_shuffle else lambda x: x
            toW = self.WtoDepth[ix] if self.pix_shuffle else lambda x: x
            combine = len(xhi_list[ix])>>2
            tmp1 = toH(refine_list[ix][1])
            C = tmp1.shape[1] // combine
            xhi_list[ix][:combine] = [tmp1[:,N*C:(N+1)*C] for N in range(combine)]
            
            tmp2 = toH(refine_list[ix][2])
            C = tmp2.shape[1] // combine
            xhi_list[ix][combine:2*combine] = [tmp2[:,N*C:(N+1)*C] for N in range(combine)]

            tmp3 = toW(refine_list[ix][3])
            C = tmp3.shape[1] // combine
            xhi_list[ix][2*combine:3*combine] = [tmp3[:,N*C:(N+1)*C] for N in range(combine)]
            
            tmp4 = toW(refine_list[ix][4])
            C = tmp4.shape[1] // combine
            xhi_list[ix][3*combine:4*combine] = [tmp4[:,N*C:(N+1)*C] for N in range(combine)]
            
        bands_copy = [None] * len(self.decomposition) # copy from graph for loss  
        lap_copy = [None] * len(self.decomposition) # copy from graph for loss  
        # for lv in range(1, len(self.reconstruct)+1):
        for lv in range(-1, -(len(self.decomposition)+1), -1):
            # rec_img = self.reconstruct[-lv]([xlo_list[-lv], xhi_list[-lv]])
            bands_copy[lv] = [band[:,:3].clone() for band in xhi_list[lv]]
            xhi = dfbrec_layer(xhi_list[lv])
            lap_copy[lv] = [xlo_list[lv][:,:3].clone(), xhi[:,:3].clone()]
            if lv != -len(self.decomposition):
                # xlo_list[-lv-1] = rec_img
                # xlo_list[lv-1] = self.reconstruct([xlo_list[lv], xhi_list[lv]])
                xlo_list[lv-1] = lprec_layer(xlo_list[lv], xhi)
            else:
                # rec_img = self.reconstruct([xlo_list[lv], xhi_list[lv]])
                rec_img = lprec_layer(xlo_list[lv], xhi)

        return rec_img, bands_copy, lap_copy

class BandGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                    n_blocks=6, padding_type='reflect', high_pass=False, n_downsampling=2):
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
        super(BandGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            # use_bias = norm_layer == nn.InstanceNorm2d
            use_bias = True # zzx (if no normalization, we need bias)

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.LeakyReLU(0.5, True)]

        # n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.LeakyReLU(0.5, True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.LeakyReLU(0.5, True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        # ZZX: We remove the nonlinear activation on the last layer of our separator since we
        # found it may slow-down the convergence.
        self.high_pass = high_pass
        self.output_nc = output_nc
        # if high_pass:
        #     model += [nn.Tanh()]
        # model += [nn.Sigmoid()]
        # model += [nn.ReLU()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        if self.high_pass:
            rep = self.output_nc // input.shape[1]
            # if rep != 1:
            #     tmp = torch.cat([input]*rep, dim=1)
            # else:
            #     tmp = input
            # return tmp + self.model(input)
            if rep != 1:
                input = torch.cat([input]*rep, dim=1)
            tmp = self.model(input)
            return tmp + input
        return self.model(input)

#
class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
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
        super(ResnetGenerator, self).__init__()
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
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
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

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
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


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            # up = [uprelu, upconv, nn.Tanh()]
            # up = [uprelu, upconv, nn.Sigmoid()]
            # ZZX: We remove the nonlinear activation on the last layer of our separator since we
            # found it may slow - down the convergence.
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 2
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


    