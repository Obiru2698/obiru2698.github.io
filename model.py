import os
import time
import random

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from tensorboardX import SummaryWriter
import math
import Attention_feature_map as at
from scipy.stats import wasserstein_distance
import Myloss

def psnr2(img1, img2):
    if type(img1) == torch.Tensor:
        mse = torch.mean((img1 - img2) ** 2)
    else:
        mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255
    return 10 * math.log10((PIXEL_MAX ** 2) / mse)

def frequency_loss(im1, im2):
    im1_fft = torch.fft.fftn(im1)
    im1_fft_real = im1_fft.real
    im1_fft_imag = im1_fft.imag
    im2_fft = torch.fft.fftn(im2)
    im2_fft_real = im2_fft.real
    im2_fft_imag = im2_fft.imag
    loss = 0
    for i in range(im1.shape[0]):
        real_loss = wasserstein_distance(im1_fft_real[i].reshape(im1_fft_real[i].shape[0]*im1_fft_real[i].shape[1]*im1_fft_real[i].shape[2]).cpu().detach(),
                                         im2_fft_real[i].reshape(im2_fft_real[i].shape[0]*im2_fft_real[i].shape[1]*im2_fft_real[i].shape[2]).cpu().detach())
        imag_loss = wasserstein_distance(im1_fft_imag[i].reshape(im1_fft_imag[i].shape[0]*im1_fft_imag[i].shape[1]*im1_fft_imag[i].shape[2]).cpu().detach(),
                                         im2_fft_imag[i].reshape(im2_fft_imag[i].shape[0]*im2_fft_imag[i].shape[1]*im2_fft_imag[i].shape[2]).cpu().detach())
        total_loss = real_loss + imag_loss
        loss += total_loss
    return torch.tensor(loss / (im1.shape[2] * im2.shape[3]))

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h0 = F.relu(self.conv1_1(X), inplace=True)
        h1 = F.relu(self.conv1_2(h0), inplace=True)
        h2 = F.max_pool2d(h1, kernel_size=2, stride=2)

        h3 = F.relu(self.conv2_1(h2), inplace=True)
        h4 = F.relu(self.conv2_2(h3), inplace=True)
        h5 = F.max_pool2d(h4, kernel_size=2, stride=2)

        h6 = F.relu(self.conv3_1(h5), inplace=True)
        h7 = F.relu(self.conv3_2(h6), inplace=True)
        h8 = F.relu(self.conv3_3(h7), inplace=True)
        h9 = F.max_pool2d(h8, kernel_size=2, stride=2)
        h10 = F.relu(self.conv4_1(h9), inplace=True)
        h11 = F.relu(self.conv4_2(h10), inplace=True)
        conv4_3 = self.conv4_3(h11)
        result = F.relu(conv4_3, inplace=True)

        return result



def load_vgg16(model_dir):
    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    vgg = Vgg16()
    vgg.cuda()
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))

    return vgg


def compute_vgg_loss(enhanced_result, input_high):
    instance_norm = nn.InstanceNorm2d(512, affine=False)
    vgg = load_vgg16("./model")
    vgg.eval()
    for param in vgg.parameters():
        param.requires_grad = False
    img_fea = vgg(enhanced_result)
    target_fea = vgg(input_high)

    loss = torch.mean((instance_norm(img_fea) - instance_norm(target_fea)) ** 2)

    return loss


class DecomNet(nn.Module):
    def __init__(self, channel=16, kernel_size=3):
        super(DecomNet, self).__init__()

        self.decom0 = nn.Conv2d(4, channel, kernel_size, stride=1, padding=1, padding_mode='replicate')

        self.csb = nn.Sequential(nn.LeakyReLU(),
                                  at.CSB(n_feat=channel))    # loop_1

        self.decom1 = nn.Conv2d(channel, channel, kernel_size, stride=1, padding=1, padding_mode='replicate')   # loop_2       


        self.decom2 = nn.Conv2d(channel*2, channel, kernel_size, stride=1, padding=1, padding_mode='replicate')    # loop_3

        self.decom3 = nn.Sequential(nn.Conv2d(channel*2, channel, kernel_size, stride=1, padding=1, padding_mode='replicate'),
                                   nn.LeakyReLU(),
                                   at.CSB(n_feat=channel),    # loop_4
                                   
                                   nn.Conv2d(channel, channel, kernel_size, stride=1, padding=1, padding_mode='replicate'),
                                   nn.LeakyReLU(),

                                   nn.Conv2d(channel, 4, kernel_size=1, stride=1, padding=0),
                                   nn.LeakyReLU())

    def forward(self, input_im):
        input_max = torch.max(input_im, dim=1, keepdim=True)[0]
        input_img = torch.cat((input_max, input_im), dim=1)

        out0 = self.decom0(input_img)
        att0 = self.csb(out0)

        out1 = self.decom1(att0)
        att1 = self.csb(out1)

        out2 = self.decom2(torch.cat((out1,att1),dim=1))
        att2 = self.csb(out2)

        out = self.decom3(torch.cat((out0,att2), dim=1))

        R = torch.sigmoid(out[:, 0:3, :, :])
        L = torch.sigmoid(out[:, 3:4, :, :])

        return R, L


class DenoiseNet(nn.Module):
     def __init__(self, channel=16, kernel_size=3):
        super(DenoiseNet, self).__init__()

        self.csb = at.CSB(n_feat=channel)
        self.skff = at.SKFF(channel,height=2)

        #top
        self.top = nn.Sequential(nn.Conv2d(4, channel, kernel_size, padding=2, padding_mode='replicate', dilation=2),
                                 nn.LeakyReLU(),
                                 nn.Conv2d(channel, channel, kernel_size, padding=2, padding_mode='replicate', dilation=2),
                                 nn.LeakyReLU(),
                                 at.CSB(n_feat=channel))
        #conv func
        self.down_conv = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=2, stride=2, padding=0),  # 48*48
                                 nn.LeakyReLU())

        self.up_conv = nn.Sequential(nn.ConvTranspose2d(channel*2, channel, kernel_size=2, stride=2, padding=0,output_padding=0),
                                            nn.LeakyReLU())  # 24*24            

        #top
        self.fus2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(channel, 3, kernel_size=1, stride=1),
                                  nn.LeakyReLU())


     def forward(self, input_L, input_R):
        input_img = torch.cat((input_R, input_L), dim=1)

        top = self.top(input_img)
        #mid
        mid_conv = self.down_conv(top)
        mid_csb = self.csb(mid_conv)
        #bot
        bot_conv = self.down_conv(mid_csb)
        bot_csb = self.csb(bot_conv)
        #under
        under_conv = self.down_conv(bot_csb)
        under_csb = self.csb(under_conv)
        under_up = self.up_conv(torch.cat((under_csb,under_conv),dim=1))
        #bot
        fus0_skff = self.skff([under_up,bot_csb])
        fus0 = self.up_conv(torch.cat((fus0_skff,bot_conv),dim=1))
        #mid
        fus1_skff = self.skff([fus0,mid_csb])
        fus1 = self.up_conv(torch.cat((fus1_skff,mid_conv),dim=1))
        #top
        fus2_skff = self.skff([fus1,top])
        fus2 = self.fus2(fus2_skff)

        denoise_R = fus2

        return denoise_R


class RelightNet(nn.Module):
    def __init__(self, channel=16, kernel_size=3):
        super(RelightNet, self).__init__()

        self.Relu = nn.LeakyReLU()
        self.csb = at.CSB(n_feat=channel)
        self.skff = at.SKFF(channel,height=2)

        #top
        self.top = nn.Sequential(nn.Conv2d(4, channel, kernel_size, padding=2, padding_mode='replicate', dilation=2),
                                 nn.LeakyReLU(),
                                 nn.Conv2d(channel, channel, kernel_size, padding=2, padding_mode='replicate', dilation=2),
                                 nn.LeakyReLU(),
                                 at.CSB(n_feat=channel))
        #conv func
        self.down_conv = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=2, stride=2, padding=0),  # 48*48
                                 nn.LeakyReLU())

        self.up_conv = nn.Sequential(nn.ConvTranspose2d(channel*2, channel, kernel_size=2, stride=2, padding=0,output_padding=0),
                                            nn.LeakyReLU())  # 24*24    
       
        self.fus2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
                                  nn.LeakyReLU())

        self.Enhanceout1 = nn.Conv2d(channel * 4, channel, kernel_size, padding=1, padding_mode='replicate')
        self.Enhanceout2 = nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, input_L, denoise_R):
        input_img = torch.cat((input_L, denoise_R), dim=1)

        top = self.top(input_img)
        #mid
        mid_conv = self.down_conv(top)
        mid_csb = self.csb(mid_conv)
        #bot
        bot_conv = self.down_conv(mid_csb)
        bot_csb = self.csb(bot_conv)
        #under
        under_conv = self.down_conv(bot_csb)
        under_csb = self.csb(under_conv)
        under_up = self.up_conv(torch.cat((under_csb,under_conv),dim=1))
        #bot
        fus0_skff = self.skff([under_up,bot_csb])
        fus0 = self.up_conv(torch.cat((fus0_skff,bot_conv),dim=1))
        #mid
        fus1_skff = self.skff([fus0,mid_csb])
        fus1 = self.up_conv(torch.cat((fus1_skff,mid_conv),dim=1))
        #top
        fus2_skff = self.skff([fus1,top])
        fus2 = self.fus2(fus2_skff)

        up0_1 = F.interpolate(under_up, size=(input_img.size()[2], input_img.size()[3]))
        up1_1 = F.interpolate(fus0, size=(input_img.size()[2], input_img.size()[3]))
        up2_1 = F.interpolate(fus2, size=(input_img.size()[2], input_img.size()[3]))

        out33 = self.Relu(self.Enhanceout1(torch.cat((up0_1, up1_1, up2_1, fus2), dim=1)))
        out34 = self.Relu(self.Enhanceout2(out33))
        Enhanced_I = out34

        return Enhanced_I

writer = SummaryWriter('./runs')


class A4RNet(nn.Module):
    def __init__(self):
        super(A4RNet, self).__init__()

        self.DecomNet = DecomNet()
        self.DenoiseNet = DenoiseNet()
        self.RelightNet = RelightNet()
        self.vgg = load_vgg16("./model")

    def forward(self, input_low, input_high):

        input_low = np.array(input_low, dtype=np.float32)
        input_high = np.array(input_high, dtype=np.float32)

        input_low = Variable(torch.FloatTensor(torch.from_numpy(input_low))).cuda()
        input_high = Variable(torch.FloatTensor(torch.from_numpy(input_high))).cuda()

        # Forward DecomNet
        R_low, I_low = self.DecomNet(input_low)
        R_high, I_high = self.DecomNet(input_high)
        # Forward DenoiseNet
        denoise_R = self.DenoiseNet(I_low, R_low)
        # Forward RelightNet
        I_delta = self.RelightNet(I_low, denoise_R)

        # Other variables
        I_low_3 = torch.cat((I_low, I_low, I_low), dim=1)
        I_high_3 = torch.cat((I_high, I_high, I_high), dim=1)
        I_delta_3 = torch.cat((I_delta, I_delta, I_delta), dim=1)

        L_spa = Myloss.L_spa()

        # DecomNet_loss
        self.vgg_loss = compute_vgg_loss(R_low * I_low_3,  input_low).cuda() + compute_vgg_loss(R_high * I_high_3, input_high).cuda()
        #SPA
        self.recon_loss_low = torch.mean(L_spa(R_low * I_low_3,input_low)).cuda()
        self.recon_loss_high = torch.mean(L_spa(R_high * I_high_3,input_high)).cuda()
        self.recon_loss_mutal_low = torch.mean(L_spa(R_high * I_low_3,input_low)).cuda()
        self.recon_loss_mutal_high = torch.mean(L_spa(R_low * I_high_3,input_high)).cuda()
        #L1
        self.recon_loss_low1 = F.l1_loss(R_low * I_low_3, input_low).cuda()
        self.recon_loss_high1 = F.l1_loss(R_high * I_high_3, input_high).cuda()
        self.recon_loss_mutal_low1 = F.l1_loss(R_high * I_low_3, input_low).cuda()
        self.recon_loss_mutal_high1 = F.l1_loss(R_low * I_high_3, input_high).cuda()
        
        # self.R_loss_spa = torch.mean(L_spa(R_high, R_low)).cuda()
        # self.L_loss_spa = torch.mean(L_spa(I_high_3, I_low_3)).cuda()

        self.loss_spa = self.recon_loss_low + \
                        self.recon_loss_high + \
                        0.1 * self.recon_loss_mutal_low + \
                        0.1 * self.recon_loss_mutal_high
        
        self.loss_RI = self.recon_loss_low1 + \
                        self.recon_loss_high1 + \
                        0.1 * self.recon_loss_mutal_low1 + \
                        0.1 * self.recon_loss_mutal_high1

        self.loss_Decom = self.loss_spa + self.loss_RI + 0.1 * self.vgg_loss
        
        # DenoiseNet_loss
        self.denoise_loss = torch.mean(L_spa(denoise_R, R_high)).cuda()
        self.denoise_loss1 = F.l1_loss(denoise_R, R_high).cuda()
        self.denoise_vgg = compute_vgg_loss(denoise_R, R_high).cuda()
        self.loss_Denoise = self.denoise_loss + \
                            self.denoise_loss1 + \
                            0.1 * self.denoise_vgg

        # RelightNet_loss
        self.Relight_loss = torch.mean(L_spa(denoise_R * I_delta_3, input_high)).cuda()
        self.Relight_loss1 = F.l1_loss(denoise_R * I_delta_3, input_high).cuda()
        self.Relight_vgg = compute_vgg_loss(denoise_R * I_delta_3, input_high).cuda()
        self.fre_loss = frequency_loss(denoise_R * I_delta_3, input_high).cuda()

        self.loss_Relight = self.Relight_loss + 0.1 * self.Relight_vgg + 0.01 * self.fre_loss + self.Relight_loss1

        self.output_R_low = R_low.detach().cpu()
        self.output_I_low = I_low_3.detach().cpu()
        self.output_I_delta = I_delta_3.detach().cpu()
        self.output_R_denoise = denoise_R.detach().cpu()
        self.output_S = denoise_R.detach().cpu() * I_delta_3.detach().cpu()

    def evaluate(self, epoch_num, eval_low_data_names, eval_high_data_names, vis_dir, train_phase):
        print("Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))
        psnr = 0
        with torch.no_grad():# Otherwise the intermediate gradient would take up huge amount of CUDA memory
            for idx in range(len(eval_low_data_names)):
                eval_low_img = Image.open(eval_low_data_names[idx])
                eval_low_img = np.array(eval_low_img, dtype="float32") / 255.0
                eval_low_img = np.transpose(eval_low_img, [2, 0, 1])
                input_low_eval = np.expand_dims(eval_low_img, axis=0)
                eval_high_img = Image.open(eval_high_data_names[idx])
                eval_high_img = np.array(eval_high_img, dtype="float32")

                if train_phase == "Decom":
                    self.forward(input_low_eval, input_low_eval)
                    result_1 = self.output_R_low
                    result_2 = self.output_I_low
                    input = np.squeeze(input_low_eval)
                    result_1 = np.squeeze(result_1)
                    result_2 = np.squeeze(result_2)
                    cat_image = np.concatenate([result_1, result_2], axis=2)
                if train_phase == 'Denoise':
                    self.forward(input_low_eval, input_low_eval)
                    result_1 = self.output_R_denoise
                    input = np.squeeze(input_low_eval)
                    result_1 = result_1.numpy().squeeze(0)
                    cat_image = result_1
                if train_phase == "Relight":
                    self.forward(input_low_eval, input_low_eval)
                    result_4 = self.output_S
                    input = np.squeeze(input_low_eval)
                    result_4 = result_4.numpy().squeeze(0)
                    cat_image = result_4

                cat_image = np.transpose(cat_image, (1, 2, 0))
            # print(cat_image.shape)
            #     im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
            #     im_test = np.array(im, dtype='float32')
            #     psnr += psnr2(im_test, eval_high_img)
            # print('psnr=', psnr / len(eval_low_data_names))
            # writer.add_scalar('runs/psnr', psnr / len(eval_low_data_names), epoch_num)

    def save(self, iter_num, ckpt_dir):
        save_dir = ckpt_dir + '/' + self.train_phase + '/'
        save_name = save_dir + '/' + str(iter_num) + '.tar'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.train_phase == 'Decom':
            torch.save(self.DecomNet.state_dict(), save_name)
        if self.train_phase == 'Denoise':
            torch.save(self.DenoiseNet.state_dict(), save_name)
        if self.train_phase == 'Relight':
            torch.save(self.RelightNet.state_dict(), save_name)

    def load(self, ckpt_dir):
        load_dir = ckpt_dir + '/' + self.train_phase + '/'
        if os.path.exists(load_dir):
            load_ckpts = os.listdir(load_dir)
            load_ckpts.sort()
            load_ckpts = sorted(load_ckpts, key=len)
            if len(load_ckpts) > 0:
                load_ckpt = load_ckpts[-1]
                global_step = int(load_ckpt[:-4])
                ckpt_dict = torch.load(load_dir + load_ckpt)
                if self.train_phase == 'Decom':
                    self.DecomNet.load_state_dict(ckpt_dict)
                if self.train_phase == 'Denoise':
                    self.DenoiseNet.load_state_dict(ckpt_dict)
                if self.train_phase == 'Relight':
                    self.RelightNet.load_state_dict(ckpt_dict)
                return True, global_step
            else:
                return False, 0
        else:
            return False, 0


    def train(self,
              train_low_data_names,
              train_high_data_names,
              eval_low_data_names,
              eval_high_data_names,
              batch_size,
              patch_size, epoch,
              lr,
              vis_dir,
              ckpt_dir,
              eval_every_epoch,
              train_phase):
        assert len(train_low_data_names) == len(train_high_data_names)
        numBatch = len(train_low_data_names) // int(batch_size)

        # Create the optimizers
        self.train_op_Decom = optim.Adam(self.DecomNet.parameters(),
                                         lr=lr[0], betas=(0.9, 0.999), weight_decay=0.0001)
        self.train_op_Denoise = optim.Adam(self.DenoiseNet.parameters(),
                                           lr=lr[0], betas=(0.9, 0.999), weight_decay=0.0001)
        self.train_op_Relight = optim.Adam(self.RelightNet.parameters(),
                                           lr=lr[0], betas=(0.9, 0.999), weight_decay=0.0001)

        # Initialize a network if its checkpoint is available
        self.train_phase = train_phase
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("No pretrained model to restore!")

        print("Start training for phase %s, with start epoch %d start iter %d : " %
             (self.train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id = 0
        for epoch in range(start_epoch, epoch):
            self.lr = lr[epoch]
            # Adjust learning rate
            for param_group in self.train_op_Decom.param_groups:    #decom
                param_group['lr'] = self.lr
            for param_group in self.train_op_Denoise.param_groups:  #denoise
                param_group['lr'] = self.lr
            for param_group in self.train_op_Relight.param_groups:  #relight
                param_group['lr'] = self.lr
            for batch_id in range(start_step, numBatch):
                # Generate training data for a batch
                batch_input_low = np.zeros((batch_size, 3, patch_size, patch_size,), dtype="float32")
                batch_input_high = np.zeros((batch_size, 3, patch_size, patch_size,), dtype="float32")
                for patch_id in range(batch_size):
                    # Load images
                    train_low_img = Image.open(train_low_data_names[image_id])
                    train_low_img = np.array(train_low_img, dtype='float32')/255.0
                    train_high_img = Image.open(train_high_data_names[image_id])
                    train_high_img = np.array(train_high_img, dtype='float32')/255.0
                    # Take random crops
                    h, w, _ = train_low_img.shape
                    x = random.randint(0, h - patch_size)   #0 ~ h-patch_size 사이의 랜덤 값 발생
                    y = random.randint(0, w - patch_size)
                    train_low_img = train_low_img[x: x + patch_size, y: y + patch_size, :]
                    train_high_img = train_high_img[x: x + patch_size, y: y + patch_size, :]
                    # Data augmentation
                    if random.random() < 0.5:
                        train_low_img = np.flipud(train_low_img)
                        train_high_img = np.flipud(train_high_img)
                    if random.random() < 0.5:
                        train_low_img = np.fliplr(train_low_img)
                        train_high_img = np.fliplr(train_high_img)
                    rot_type = random.randint(1, 4)
                    if random.random() < 0.5:
                        train_low_img = np.rot90(train_low_img, rot_type)
                        train_high_img = np.rot90(train_high_img, rot_type)
                    # Permute the images to tensor format
                    train_low_img = np.transpose(train_low_img, (2, 0, 1))
                    train_high_img = np.transpose(train_high_img, (2, 0, 1))
                    # Prepare the batch
                    batch_input_low[patch_id, :, :, :] = train_low_img
                    batch_input_high[patch_id, :, :, :] = train_high_img
                    self.input_low = batch_input_low
                    self.input_high = batch_input_high

                    image_id = (image_id + 1) % len(train_low_data_names)
                    if image_id == 0:
                        tmp = list(zip(train_low_data_names, train_high_data_names))
                        random.shuffle(list(tmp))
                        train_low_data_names, train_high_data_names = zip(*tmp)

                # Feed-Forward to the network and obtain loss
                self.forward(self.input_low,  self.input_high)
                if self.train_phase == "Decom":
                    self.train_op_Decom.zero_grad()
                    self.loss_Decom.backward()
                    self.train_op_Decom.step()
                    global_step += 1
                    loss = self.loss_Decom.item()
                if self.train_phase == 'Denoise':
                    self.train_op_Denoise.zero_grad()
                    self.loss_Denoise.backward()
                    self.train_op_Denoise.step()
                    global_step += 1
                    loss = self.loss_Denoise.item()
                elif self.train_phase == "Relight":
                    self.train_op_Relight.zero_grad()
                    self.loss_Relight.backward()
                    self.train_op_Relight.step()
                    global_step += 1
                    loss = self.loss_Relight.item()

                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
                      % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1
                writer.add_scalar('runs/loss', loss, global_step)
                img = torch.rand(3, 3, 96, 96).numpy()
                if global_step % 10 == 0:
                    img[:1, :, :, :] = batch_input_low[:1, :, :, :]
                    img[1:2, :, :, :] = self.output_S[:1, :, :, :]
                    img[2:3, :, :, :] = batch_input_high[:1, :, :, :]
                    writer.add_images('results', img)

            # Evaluate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                self.save(iter_num, ckpt_dir)
                self.evaluate(epoch + 1, eval_low_data_names, eval_high_data_names,
                              vis_dir=vis_dir, train_phase=train_phase)

        print("Finished training for phase %s." % train_phase)


    def predict(self,
                test_low_data_names,
                res_dir,
                ckpt_dir):

        # Load the network with a pre-trained checkpoint
        self.train_phase = 'Decom'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
            print(self.train_phase, "  : Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception

        self.train_phase = 'Denoise'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
            print(self.train_phase, ": Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception

        self.train_phase = 'Relight'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
            print(self.train_phase, ": Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception

        # Set this switch to True to also save the reflectance and shading maps
        save_R_L = False


        for idx in range(len(test_low_data_names)):
            test_img_path = test_low_data_names[idx]
            test_img_name = test_img_path.split('/')[-1]
            print('Processing ', test_img_name)
            test_low_img = Image.open(test_img_path)

            #test_low_img = test_low_img.resize((600, 400))


            test_low_img = np.array(test_low_img, dtype="float32")/255.0
            test_low_img = np.transpose(test_low_img, (2, 0, 1))
            input_low_test = np.expand_dims(test_low_img, axis=0)

            self.forward(input_low_test, input_low_test)

            # data1 = self.output_R_low
            # data2 = self.output_I_low

            result_1 = self.output_R_denoise
            result_2 = self.output_I_low
            result_3 = self.output_I_delta
            result_4 = self.output_S
            input = np.squeeze(input_low_test)
            result_1 = np.squeeze(result_1)
            result_2 = np.squeeze(result_2)
            result_3 = np.squeeze(result_3)
            result_4 = np.squeeze(result_4)

            # data1 = np.squeeze(data1)
            # data2 = np.squeeze(data2)
            if save_R_L:
                cat_image = np.concatenate([input, result_1, result_2, result_3, result_4], axis=2)
            else:
                cat_image = result_4.numpy()
                # data1 = data1.numpy()
                # data2 = data2.numpy()
                # result_1 = result_1.numpy()
                # result_2 = result_2.numpy()
                # result_3 = result_3.numpy()


            # data1 = np.transpose(data1, (1, 2, 0))
            # data2 = np.transpose(data2, (1, 2, 0))
            # image_1 = np.transpose(result_1, (1, 2, 0))
            # image_2 = np.transpose(result_2, (1, 2, 0))
            # image_3 = np.transpose(result_3, (1, 2, 0))
            cat_image = np.transpose(cat_image, (1, 2, 0))
            # print(cat_image.shape)
            # d1 = Image.fromarray(np.clip(data1 * 255.0, 0, 255.0).astype('uint8'))
            # d2 = Image.fromarray(np.clip(data2 * 255.0, 0, 255.0).astype('uint8'))

            # im1 = Image.fromarray(np.clip(image_1 * 255.0, 0, 255.0).astype('uint8'))
            # im2 = Image.fromarray(np.clip(image_2 * 255.0, 0, 255.0).astype('uint8'))
            # im3 = Image.fromarray(np.clip(image_3 * 255.0, 0, 255.0).astype('uint8'))

            im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))

            filepath = res_dir + '/' + test_img_name
            # d1.save(filepath[:-4] + 'I.jpg')
            # d2.save(filepath[:-4] + 'R.jpg')
            # im1.save(filepath[:-4] + 'R_denoise.jpg')
            # im2.save(filepath[:-4] + 'I_low.jpg')
            # im3.save(filepath[:-4] + 'I_delta.jpg')

            im.save(filepath[:-4] + '.jpg')
