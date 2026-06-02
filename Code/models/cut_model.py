import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import utils.util as util
from .mutual_info import MutualInformation
import einops
import torch.nn.functional as F
import cv2
from torch.autograd import Variable
class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.add_argument('--lambda_smooth', type=float, default=0.0, help='weight for smoothness of GAN output stack：GAN(G(X))')
        parser.add_argument('--lambda_spa_unsup_A', type=float, default=1, help='weight for unsupervised spatial loss with optical flow G(W(A)) = W(G(A))')
        parser.add_argument('--lambda_kl', type=float, default= 1, help='weight for kl divergency between G(X) and X to force structural consistancy')

        parser.add_argument('--unsup_idt_spa', type=bool, default=True, help='If true, apply unsupervised spatial loss with optical flow G(W(B)) = W(B). For identity check')
       
        parser.add_argument('--motion_level', type=float, default=8., help='weight for optical flow motion')
        parser.add_argument('--shift_level', type=float, default=10., help='weight for optical flow shift')
        parser.add_argument('--scale_level', type=float, default=0.0, help='weight for optical flow scaling')
        parser.add_argument('--noise_level', type=float, default=0.001, help='weight for gaussian noise in warp image')
     
        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        # self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'smoothness', 'NMI']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        self.MI = MutualInformation(num_bins=256, sigma=0.4, normalize=True).to(self.gpu_ids[0])

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def calculate_smooth_loss(self):
        """Calculate smoothness of GAN output between the consecutive slices  """
        generated_images = self.fake_B
        loss_smooth = 0.0
        for i in range(generated_images.shape[1] - 1):
            # Calculate the difference between consecutive images
            diff = generated_images[:,i + 1,::] - generated_images[:,i,::]
            # Calculate the mean squared difference
            loss_smooth += torch.mean(diff ** 2)
        # Normalize by the number of images minus one and batch size
        loss_smooth /= (generated_images.shape[1] - 1)*len(generated_images)
        return loss_smooth* self.opt.lambda_smooth

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() 
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        if self.opt.lambda_smooth > 0.0:
            self.loss_smooth = self.calculate_smooth_loss()
        else:
            self.loss_smooth = 0.0

        if self.opt.lambda_spa_unsup_A > 0.0:
            self.loss_unsup_spa = self.calculate_unsup_spa_loss()
        else:
            self.loss_unsup_spa = 0.0

        if self.opt.lambda_kl > 0.0:
            self.loss_kl = self.calculate_kl_divergence()
        else:
            self.loss_kl = 0
        
        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_smooth + self.loss_kl + self.loss_unsup_spa
        return self.loss_G

    def calculate_unsup_spa_loss(self):
        warped_real_A, fake_flow_A = self.GenerateFakeData(self.real_A, self.opt)
        real_A_warped = self.warp(self.real_A, fake_flow_A)
        # G(W(CTCS))
        warped_fake_B = self.netG(warped_real_A) 
        # W(G(CTCS))
        fake_B_warped = self.warp(self.fake_B, fake_flow_A)
        diff_A = torch.abs(warped_fake_B - fake_B_warped)

        # also for idt unsup spa
        if self.opt.unsup_idt_spa:
            warped_real_B = self.warp(self.real_B, fake_flow_A)
            # G(W(CCTA)) should equal to W(CCTA))
            # G(W(CCTA))
            warped_idt_B = self.netG(warped_real_B)
            diff_B = torch.abs(warped_real_B - warped_idt_B)
            diff = (torch.mean(diff_A) + torch.mean(diff_B)) *0.5
        else:
            diff = torch.mean(diff_A)
        return self.opt.lambda_spa_unsup_A* diff


    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def calculate_NMI_loss(self):
        # get VCTA image
        generated_images = self.fake_B
        input_image = self.real_A
        return self.MI(generated_images, input_image)

    def calculate_histograms(self, image, bins=256, min_value=0, max_value=255):
        # Flatten the image
        image = image.flatten()
        
        # Calculate histogram
        histogram = torch.histc(image, bins=bins, min=min_value, max=max_value)
        
        # Normalize histogram
        histogram = histogram / histogram.sum()
        
        return histogram
    
    def calculate_kl_divergence(self, bins=256):
        # Calculate normalized histograms

        img1 = self.fake_B.mean(dim=1, keepdim=True)
        img2 = self.real_A.mean(dim=1, keepdim=True)

        hist1 = self.calculate_histograms(img1, bins=bins).detach()
        hist2 = self.calculate_histograms(img2, bins=bins).detach()
        # Ensure that histograms are of the same shape and type
        assert hist1.shape == hist2.shape, "Histograms must have the same shape"
        assert hist1.dtype == hist2.dtype, "Histograms must have the same data type"
        # Add a small value to avoid division by zero
        epsilon = 1e-10
        hist1 = hist1 + epsilon
        hist2 = hist2 + epsilon
    
        # Calculate KL divergence
        kl_div = F.kl_div(hist1.log(), hist2, reduction='batchmean')
       
        return self.opt.lambda_kl*kl_div

    def GaussianNoise(self, ins, mean=0, stddev=0.03):
      # adapted from https://github.com/daooshee/ReReVST-Code/blob/master/train/loss_networks.py
      stddev = stddev + np.random.random() * stddev
      noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
      
      if ins.is_cuda:
          noise = noise.cuda()
      return ins + noise

    def GenerateFakeFlow(self, height, width, motion_level, shift_level, scale_level):
      # adapted from https://github.com/daooshee/ReReVST-Code/blob/master/train/loss_networks.py
      ''' height: img.shape[0]
          width:  img.shape[1] '''
      if scale_level > 0:
          flow = np.ones([height,width,2])
          scale_factor = random.uniform(-scale_level, scale_level)
          unit = np.arange(-3, 3, 0.01)
          cent_x = random.randint(0, len(unit) - width)
          cent_y = random.randint(0, len(unit) - height)
    
          x_range = unit[cent_x:cent_x+width]
          y_range = unit[cent_y:cent_y+height]
    
          xx = np.tile(x_range.reshape(1,-1), (height,1)) * scale_factor
          yy = np.tile(y_range.reshape(-1,1), (1,width)) * scale_factor
    
          flow[:,:,0] = xx
          flow[:,:,1] = yy
    
          return torch.from_numpy(flow.transpose((2, 0, 1))).float()
    
      if motion_level > 0:
          flow = np.random.normal(0, scale=motion_level, size = [height//100, width//100, 2])
          flow = cv2.resize(flow, (width, height))
          flow[:,:,0] += np.random.randint(-shift_level, shift_level)
          flow[:,:,1] += np.random.randint(-shift_level, shift_level)
          flow = cv2.blur(flow,(100,100))
      else:
          flow = np.ones([height,width,2])
          flow[:,:,0] = np.random.randint(-shift_level, shift_level)
          flow[:,:,1] = np.random.randint(-shift_level, shift_level)
    
      return torch.from_numpy(flow.transpose((2, 0, 1))).float()

    def GenerateFakeData(self, first_frame, hyperparameters):
      # adapted from https://github.com/daooshee/ReReVST-Code/blob/master/train/loss_networks.py
      B, C, H, W = first_frame.size()
      # print('first_frame size:')
      # print(first_frame.size())
    
      fake_flow = self.GenerateFakeFlow(H, W, hyperparameters.motion_level, hyperparameters.shift_level, hyperparameters.scale_level)
      if first_frame.is_cuda:
          fake_flow = fake_flow.cuda()
      fake_flow = fake_flow.expand(B, 2, H, W)
      second_frame = self.warp(first_frame, fake_flow)
      second_frame = self.GaussianNoise(second_frame, stddev=hyperparameters.noise_level)
    
      return second_frame, fake_flow

    def warp(self, x, flo, padding_mode='border'):
        B, C, H, W = x.size()
        # print('x size:')
        # print(x.size())
        # Mesh grid
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()
        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid - flo
        
        # Scale grid to [-1,1]
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0
        vgrid = vgrid.permute(0,2,3,1)
        output = F.grid_sample(x, vgrid, padding_mode=padding_mode, mode='bilinear')
        return output      
