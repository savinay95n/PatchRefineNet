# -----------------------------------------------------------------------------------------------------------------------
# network.py: This is the code for PRN Auxiliary Refinement model

# Usage: python src/kvasir/refine-models/network.py

# Note: To independently run, always run this file from PatchRefineNetwork (root) directory.
# -----------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchviz import make_dot
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class PSPModule(nn.Module):
    def __init__(self, features, out_features = 256, sizes = (1,2,4,8)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size = 1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h,w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

class ResidualBlock(nn.Module):
    def __init__(self,ch_out,t=2):
        super(ResidualBlock,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            ResidualBlock(ch_out,t=t),
            ResidualBlock(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class U_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)

        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)

        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        # decoding + concat path
        d4 = self.Up4(x4)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2U_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1,t=2):
        super(R2U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=32,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=32,ch_out=64,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=128,ch_out=256,t=t)
                

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=128, ch_out=64,t=t)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=64, ch_out=32,t=t)
        
        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        # decoding + concat path
        d4 = self.Up4(x4)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)

        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = Attention_block(F_g=32,F_l=32,F_int=16)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        # decoding + concat path
        d4 = self.Up4(x4)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2AttU_Net_PSP(nn.Module):
    def __init__(self,img_ch=1,output_ch=1,t=2,psp_infeatures = 256, psp_outfeatures = 256, sizes = (1,2,4,8)):
        super(R2AttU_Net_PSP,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=32,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=32,ch_out=64,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=128,ch_out=256,t=t)  

        self.psp = PSPModule(psp_infeatures, psp_outfeatures, sizes)      

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_RRCNN4 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_RRCNN3 = RRCNN_block(ch_in=128, ch_out=64,t=t)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = Attention_block(F_g=32,F_l=32,F_int=16)
        self.Up_RRCNN2 = RRCNN_block(ch_in=64, ch_out=32,t=t)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x4 = self.psp(x4)

        # decoding + concat path
        d4 = self.Up4(x4)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class ThreshNetEncoder(nn.Module):
    def __init__(self,img_ch=1,t=2,psp_infeatures = 256, psp_outfeatures = 256, sizes = (1,2,4,8)):
        super(ThreshNetEncoder,self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=32,t=t)
        self.RRCNN2 = RRCNN_block(ch_in=32,ch_out=64,t=t)
        self.RRCNN3 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        self.RRCNN4 = RRCNN_block(ch_in=128,ch_out=256,t=t)  
        self.psp = PSPModule(psp_infeatures, psp_outfeatures, sizes)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)
        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)
        x4 = self.psp(x4)
        return x1, x2, x3, x4

class ThreshNetDecoder(nn.Module):
    def __init__(self,in_ch,out_ch,t=2):
        super(ThreshNetDecoder,self).__init__()

        self.Up = up_conv(ch_in=in_ch,ch_out=out_ch)
        self.Up_Conv = RRCNN_block(ch_in=in_ch, ch_out=out_ch, t=t)
        self.Attn = Attention_block(F_g=in_ch//2, F_l=out_ch, F_int=out_ch//2)
    
    def forward(self, input1, input2):
        d = self.Up(input2)
        x = self.Attn(g=d,x=input1)
        d = torch.cat((x,d),dim=1)
        d = self.Up_Conv(d)
        return d

class ThreshNetP0(nn.Module):
    # Model for Image-Specific-Thresholding
    def __init__(self,img_ch=1,output_ch=1,t=2):
        super(ThreshNetP0,self).__init__()
        # global Decoder
        self.encoding = ThreshNetEncoder()
        self.decode3_global = ThreshNetDecoder(256, 128)
        self.decode2_global = ThreshNetDecoder(128, 64)
        self.decode1_global = ThreshNetDecoder(64, 32)

        self.out_global = nn.Sequential(
            nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0),
            nn.Sigmoid()
        )

    def forward(self,x):
        # encoding path
        x0, x1, x2, center = self.encoding(x)

        # global path
        decode3_global = self.decode3_global(x2, center)
        decode2_global = self.decode2_global(x1, decode3_global)
        decode1_global = self.decode1_global(x0, decode2_global)
        output_global = self.out_global(decode1_global)

        return output_global

class ThreshNetP4(nn.Module):
    def __init__(self,img_ch=1,output_ch=1,t=2):
        super(ThreshNetP4,self).__init__()
        # global Decoder
        self.encoding = ThreshNetEncoder()
        self.decode3_global = ThreshNetDecoder(256, 128)
        self.decode2_global = ThreshNetDecoder(128, 64)
        self.decode1_global = ThreshNetDecoder(64, 32)
        # local decoder
        self.decode3_local = ThreshNetDecoder(256, 128)
        self.decode2_local = ThreshNetDecoder(128, 64)
        self.decode1_local = ThreshNetDecoder(64, 32)

        self.conv_last_global = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_last_local = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        )
            
    def forward(self,x):
        # encoding path
        x0, x1, x2, center = self.encoding(x)

        # global path
        decode3_global = self.decode3_global(x2, center)
        decode2_global = self.decode2_global(x1, decode3_global)
        decode1_global = self.decode1_global(x0, decode2_global)
        output_global = self.conv_last_global(decode1_global)

        # First row of patches
        center_p11 = center[:,:,:16,:16]
        center_p12 = center[:,:,:16,16:32]
        center_p13 = center[:,:,:16,32:48]
        center_p14 = center[:,:,:16,48:]

        x2_p11 = x2[:,:,:32,:32]
        x2_p12 = x2[:,:,:32,32:64]
        x2_p13 = x2[:,:,:32,64:96]
        x2_p14 = x2[:,:,:32,96:]

        x1_p11 = x1[:,:,:64,:64]
        x1_p12 = x1[:,:,:64,64:128]
        x1_p13 = x1[:,:,:64,128:192]
        x1_p14 = x1[:,:,:64,192:]

        x0_p11 = x0[:,:,:128,:128]
        x0_p12 = x0[:,:,:128,128:256]
        x0_p13 = x0[:,:,:128,256:384]
        x0_p14 = x0[:,:,:128,384:]

        # Second row of patches
        center_p21 = center[:,:,16:32,:16]
        center_p22 = center[:,:,16:32,16:32]
        center_p23 = center[:,:,16:32,32:48]
        center_p24 = center[:,:,16:32,48:]

        x2_p21 = x2[:,:,32:64,:32]
        x2_p22 = x2[:,:,32:64,32:64]
        x2_p23 = x2[:,:,32:64,64:96]
        x2_p24 = x2[:,:,32:64,96:]

        x1_p21 = x1[:,:,64:128,:64]
        x1_p22 = x1[:,:,64:128,64:128]
        x1_p23 = x1[:,:,64:128,128:192]
        x1_p24 = x1[:,:,64:128,192:]

        x0_p21 = x0[:,:,128:256,:128]
        x0_p22 = x0[:,:,128:256,128:256]
        x0_p23 = x0[:,:,128:256,256:384]
        x0_p24 = x0[:,:,128:256,384:]

        # Third row of patches
        center_p31 = center[:,:,32:48,:16]
        center_p32 = center[:,:,32:48,16:32]
        center_p33 = center[:,:,32:48,32:48]
        center_p34 = center[:,:,32:48,48:]

        x2_p31 = x2[:,:,64:96,:32]
        x2_p32 = x2[:,:,64:96,32:64]
        x2_p33 = x2[:,:,64:96,64:96]
        x2_p34 = x2[:,:,64:96,96:]

        x1_p31 = x1[:,:,128:192,:64]
        x1_p32 = x1[:,:,128:192,64:128]
        x1_p33 = x1[:,:,128:192,128:192]
        x1_p34 = x1[:,:,128:192,192:]

        x0_p31 = x0[:,:,256:384,:128]
        x0_p32 = x0[:,:,256:384,128:256]
        x0_p33 = x0[:,:,256:384,256:384]
        x0_p34 = x0[:,:,256:384,384:]

        # Fourth row of patches
        center_p41 = center[:,:,48:,:16]
        center_p42 = center[:,:,48:,16:32]
        center_p43 = center[:,:,48:,32:48]
        center_p44 = center[:,:,48:,48:]

        x2_p41 = x2[:,:,96:,:32]
        x2_p42 = x2[:,:,96:,32:64]
        x2_p43 = x2[:,:,96:,64:96]
        x2_p44 = x2[:,:,96:,96:]

        x1_p41 = x1[:,:,192:,:64]
        x1_p42 = x1[:,:,192:,64:128]
        x1_p43 = x1[:,:,192:,128:192]
        x1_p44 = x1[:,:,192:,192:]

        x0_p41 = x0[:,:,384:,:128]
        x0_p42 = x0[:,:,384:,128:256]
        x0_p43 = x0[:,:,384:,256:384]
        x0_p44 = x0[:,:,384:,384:]

        # First row of patches decode
        decode3_p11 = self.decode3_local(x2_p11, center_p11)
        decode2_p11 = self.decode2_local(x1_p11, decode3_p11)
        decode1_p11 = self.decode1_local(x0_p11, decode2_p11)
        output_p11 = self.conv_last_local(decode1_p11)
        
        decode3_p12 = self.decode3_local(x2_p12, center_p12)
        decode2_p12 = self.decode2_local(x1_p12, decode3_p12)
        decode1_p12 = self.decode1_local(x0_p12, decode2_p12)
        output_p12 = self.conv_last_local(decode1_p12)

        decode3_p13 = self.decode3_local(x2_p13, center_p13)
        decode2_p13 = self.decode2_local(x1_p13, decode3_p13)
        decode1_p13 = self.decode1_local(x0_p13, decode2_p13)
        output_p13 = self.conv_last_local(decode1_p13)

        decode3_p14 = self.decode3_local(x2_p14, center_p14)
        decode2_p14 = self.decode2_local(x1_p14, decode3_p14)
        decode1_p14 = self.decode1_local(x0_p14, decode2_p14)
        output_p14 = self.conv_last_local(decode1_p14)

        # Second row of patches decode
        decode3_p21 = self.decode3_local(x2_p21, center_p21)
        decode2_p21 = self.decode2_local(x1_p21, decode3_p21)
        decode1_p21 = self.decode1_local(x0_p21, decode2_p21)
        output_p21 = self.conv_last_local(decode1_p21)
        
        decode3_p22 = self.decode3_local(x2_p22, center_p22)
        decode2_p22 = self.decode2_local(x1_p22, decode3_p22)
        decode1_p22 = self.decode1_local(x0_p22, decode2_p22)
        output_p22 = self.conv_last_local(decode1_p22)

        decode3_p23 = self.decode3_local(x2_p23, center_p23)
        decode2_p23 = self.decode2_local(x1_p23, decode3_p23)
        decode1_p23 = self.decode1_local(x0_p23, decode2_p23)
        output_p23 = self.conv_last_local(decode1_p23)

        decode3_p24 = self.decode3_local(x2_p24, center_p24)
        decode2_p24 = self.decode2_local(x1_p24, decode3_p24)
        decode1_p24 = self.decode1_local(x0_p24, decode2_p24)
        output_p24 = self.conv_last_local(decode1_p24)

        # Third row of patches decode
        decode3_p31 = self.decode3_local(x2_p31, center_p31)
        decode2_p31 = self.decode2_local(x1_p31, decode3_p31)
        decode1_p31 = self.decode1_local(x0_p31, decode2_p31)
        output_p31 = self.conv_last_local(decode1_p31)
        
        decode3_p32 = self.decode3_local(x2_p32, center_p32)
        decode2_p32 = self.decode2_local(x1_p32, decode3_p32)
        decode1_p32 = self.decode1_local(x0_p32, decode2_p32)
        output_p32 = self.conv_last_local(decode1_p32)

        decode3_p33 = self.decode3_local(x2_p33, center_p33)
        decode2_p33 = self.decode2_local(x1_p33, decode3_p33)
        decode1_p33 = self.decode1_local(x0_p33, decode2_p33)
        output_p33 = self.conv_last_local(decode1_p33)

        decode3_p34 = self.decode3_local(x2_p34, center_p34)
        decode2_p34 = self.decode2_local(x1_p34, decode3_p34)
        decode1_p34 = self.decode1_local(x0_p34, decode2_p34)
        output_p34 = self.conv_last_local(decode1_p34)

        # Fourth row of patches decode
        decode3_p41 = self.decode3_local(x2_p41, center_p41)
        decode2_p41 = self.decode2_local(x1_p41, decode3_p41)
        decode1_p41 = self.decode1_local(x0_p41, decode2_p41)
        output_p41 = self.conv_last_local(decode1_p41)
        
        decode3_p42 = self.decode3_local(x2_p42, center_p42)
        decode2_p42 = self.decode2_local(x1_p42, decode3_p42)
        decode1_p42 = self.decode1_local(x0_p42, decode2_p42)
        output_p42 = self.conv_last_local(decode1_p42)

        decode3_p43 = self.decode3_local(x2_p43, center_p43)
        decode2_p43 = self.decode2_local(x1_p43, decode3_p43)
        decode1_p43 = self.decode1_local(x0_p43, decode2_p43)
        output_p43 = self.conv_last_local(decode1_p43)

        decode3_p44 = self.decode3_local(x2_p44, center_p44)
        decode2_p44 = self.decode2_local(x1_p44, decode3_p44)
        decode1_p44 = self.decode1_local(x0_p44, decode2_p44)
        output_p44 = self.conv_last_local(decode1_p44)

        row1 = torch.cat((output_p11, output_p12, output_p13, output_p14), axis=3)
        row2 = torch.cat((output_p21, output_p22, output_p23, output_p24), axis=3)
        row3 = torch.cat((output_p31, output_p32, output_p33, output_p34), axis=3)
        row4 = torch.cat((output_p41, output_p42, output_p43, output_p44), axis=3)
        output_local = torch.cat((row1, row2, row3, row4), axis=2)

        return output_global, output_local

class ThreshNetP64(nn.Module):
    def __init__(self,img_ch=1,output_ch=1,t=2):
        super(ThreshNetP64,self).__init__()
        # global Decoder
        self.encoding = ThreshNetEncoder()
        self.decode3_global = ThreshNetDecoder(256, 128)
        self.decode2_global = ThreshNetDecoder(128, 64)
        self.decode1_global = ThreshNetDecoder(64, 32)
        # local decoder
        self.decode3_local = ThreshNetDecoder(256, 128)
        self.decode2_local = ThreshNetDecoder(128, 64)
        self.decode1_local = ThreshNetDecoder(64, 32)

        self.out_global = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        self.out_local = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        )
            
    def forward(self,x):
        # encoding path
        x0, x1, x2, center = self.encoding(x)

        # global path
        decode3_global = self.decode3_global(x2, center)
        decode2_global = self.decode2_global(x1, decode3_global)
        decode1_global = self.decode1_global(x0, decode2_global)
        output_global = self.out_global(decode1_global)

        # First row of patches
        center_p11 = center[:,:,:8,:8]
        center_p12 = center[:,:,:8,8:16]
        center_p13 = center[:,:,:8,16:24]
        center_p14 = center[:,:,:8,24:32]
        center_p15 = center[:,:,:8,32:40]
        center_p16 = center[:,:,:8,40:48] 
        center_p17 = center[:,:,:8,48:56]
        center_p18 = center[:,:,:8,56:]

        x2_p11 = x2[:,:,:16,:16]
        x2_p12 = x2[:,:,:16,16:32]
        x2_p13 = x2[:,:,:16,32:48]
        x2_p14 = x2[:,:,:16,48:64]
        x2_p15 = x2[:,:,:16,64:80]
        x2_p16 = x2[:,:,:16,80:96]
        x2_p17 = x2[:,:,:16,96:112]
        x2_p18 = x2[:,:,:16,112:]

        x1_p11 = x1[:,:,:32,:32]
        x1_p12 = x1[:,:,:32,32:64]
        x1_p13 = x1[:,:,:32,64:96]
        x1_p14 = x1[:,:,:32,96:128]
        x1_p15 = x1[:,:,:32,128:160]
        x1_p16 = x1[:,:,:32,160:192]
        x1_p17 = x1[:,:,:32,192:224]
        x1_p18 = x1[:,:,:32,224:]

        x0_p11 = x0[:,:,:64,:64]
        x0_p12 = x0[:,:,:64,64:128]
        x0_p13 = x0[:,:,:64,128:192]
        x0_p14 = x0[:,:,:64,192:256]
        x0_p15 = x0[:,:,:64,256:320]
        x0_p16 = x0[:,:,:64,320:384]
        x0_p17 = x0[:,:,:64,384:448]
        x0_p18 = x0[:,:,:64,448:]

        # Second row of patches
        center_p21 = center[:,:,8:16,:8]
        center_p22 = center[:,:,8:16,8:16]
        center_p23 = center[:,:,8:16,16:24]
        center_p24 = center[:,:,8:16,24:32]
        center_p25 = center[:,:,8:16,32:40]
        center_p26 = center[:,:,8:16,40:48] 
        center_p27 = center[:,:,8:16,48:56]
        center_p28 = center[:,:,8:16,56:]

        x2_p21 = x2[:,:,16:32,:16]
        x2_p22 = x2[:,:,16:32,16:32]
        x2_p23 = x2[:,:,16:32,32:48]
        x2_p24 = x2[:,:,16:32,48:64]
        x2_p25 = x2[:,:,16:32,64:80]
        x2_p26 = x2[:,:,16:32,80:96]
        x2_p27 = x2[:,:,16:32,96:112]
        x2_p28 = x2[:,:,16:32,112:]

        x1_p21 = x1[:,:,32:64,:32]
        x1_p22 = x1[:,:,32:64,32:64]
        x1_p23 = x1[:,:,32:64,64:96]
        x1_p24 = x1[:,:,32:64,96:128]
        x1_p25 = x1[:,:,32:64,128:160]
        x1_p26 = x1[:,:,32:64,160:192]
        x1_p27 = x1[:,:,32:64,192:224]
        x1_p28 = x1[:,:,32:64,224:]

        x0_p21 = x0[:,:,64:128,:64]
        x0_p22 = x0[:,:,64:128,64:128]
        x0_p23 = x0[:,:,64:128,128:192]
        x0_p24 = x0[:,:,64:128,192:256]
        x0_p25 = x0[:,:,64:128,256:320]
        x0_p26 = x0[:,:,64:128,320:384]
        x0_p27 = x0[:,:,64:128,384:448]
        x0_p28 = x0[:,:,64:128,448:]

        # Third row of patches
        center_p31 = center[:,:,16:24,:8]
        center_p32 = center[:,:,16:24,8:16]
        center_p33 = center[:,:,16:24,16:24]
        center_p34 = center[:,:,16:24,24:32]
        center_p35 = center[:,:,16:24,32:40]
        center_p36 = center[:,:,16:24,40:48] 
        center_p37 = center[:,:,16:24,48:56]
        center_p38 = center[:,:,16:24,56:]

        x2_p31 = x2[:,:,32:48,:16]
        x2_p32 = x2[:,:,32:48,16:32]
        x2_p33 = x2[:,:,32:48,32:48]
        x2_p34 = x2[:,:,32:48,48:64]
        x2_p35 = x2[:,:,32:48,64:80]
        x2_p36 = x2[:,:,32:48,80:96]
        x2_p37 = x2[:,:,32:48,96:112]
        x2_p38 = x2[:,:,32:48,112:]

        x1_p31 = x1[:,:,64:96,:32]
        x1_p32 = x1[:,:,64:96,32:64]
        x1_p33 = x1[:,:,64:96,64:96]
        x1_p34 = x1[:,:,64:96,96:128]
        x1_p35 = x1[:,:,64:96,128:160]
        x1_p36 = x1[:,:,64:96,160:192]
        x1_p37 = x1[:,:,64:96,192:224]
        x1_p38 = x1[:,:,64:96,224:]

        x0_p31 = x0[:,:,128:192,:64]
        x0_p32 = x0[:,:,128:192,64:128]
        x0_p33 = x0[:,:,128:192,128:192]
        x0_p34 = x0[:,:,128:192,192:256]
        x0_p35 = x0[:,:,128:192,256:320]
        x0_p36 = x0[:,:,128:192,320:384]
        x0_p37 = x0[:,:,128:192,384:448]
        x0_p38 = x0[:,:,128:192,448:]

        # Fourth row of patches
        center_p41 = center[:,:,24:32,:8]
        center_p42 = center[:,:,24:32,8:16]
        center_p43 = center[:,:,24:32,16:24]
        center_p44 = center[:,:,24:32,24:32]
        center_p45 = center[:,:,24:32,32:40]
        center_p46 = center[:,:,24:32,40:48] 
        center_p47 = center[:,:,24:32,48:56]
        center_p48 = center[:,:,24:32,56:]

        x2_p41 = x2[:,:,48:64,:16]
        x2_p42 = x2[:,:,48:64,16:32]
        x2_p43 = x2[:,:,48:64,32:48]
        x2_p44 = x2[:,:,48:64,48:64]
        x2_p45 = x2[:,:,48:64,64:80]
        x2_p46 = x2[:,:,48:64,80:96]
        x2_p47 = x2[:,:,48:64,96:112]
        x2_p48 = x2[:,:,48:64,112:]

        x1_p41 = x1[:,:,96:128,:32]
        x1_p42 = x1[:,:,96:128,32:64]
        x1_p43 = x1[:,:,96:128,64:96]
        x1_p44 = x1[:,:,96:128,96:128]
        x1_p45 = x1[:,:,96:128,128:160]
        x1_p46 = x1[:,:,96:128,160:192]
        x1_p47 = x1[:,:,96:128,192:224]
        x1_p48 = x1[:,:,96:128,224:]

        x0_p41 = x0[:,:,192:256,:64]
        x0_p42 = x0[:,:,192:256,64:128]
        x0_p43 = x0[:,:,192:256,128:192]
        x0_p44 = x0[:,:,192:256,192:256]
        x0_p45 = x0[:,:,192:256,256:320]
        x0_p46 = x0[:,:,192:256,320:384]
        x0_p47 = x0[:,:,192:256,384:448]
        x0_p48 = x0[:,:,192:256,448:]

        # Fifth row of patches
        center_p51 = center[:,:,32:40,:8]
        center_p52 = center[:,:,32:40,8:16]
        center_p53 = center[:,:,32:40,16:24]
        center_p54 = center[:,:,32:40,24:32]
        center_p55 = center[:,:,32:40,32:40]
        center_p56 = center[:,:,32:40,40:48] 
        center_p57 = center[:,:,32:40,48:56]
        center_p58 = center[:,:,32:40,56:]

        x2_p51 = x2[:,:,64:80,:16]
        x2_p52 = x2[:,:,64:80,16:32]
        x2_p53 = x2[:,:,64:80,32:48]
        x2_p54 = x2[:,:,64:80,48:64]
        x2_p55 = x2[:,:,64:80,64:80]
        x2_p56 = x2[:,:,64:80,80:96]
        x2_p57 = x2[:,:,64:80,96:112]
        x2_p58 = x2[:,:,64:80,112:]

        x1_p51 = x1[:,:,128:160,:32]
        x1_p52 = x1[:,:,128:160,32:64]
        x1_p53 = x1[:,:,128:160,64:96]
        x1_p54 = x1[:,:,128:160,96:128]
        x1_p55 = x1[:,:,128:160,128:160]
        x1_p56 = x1[:,:,128:160,160:192]
        x1_p57 = x1[:,:,128:160,192:224]
        x1_p58 = x1[:,:,128:160,224:]

        x0_p51 = x0[:,:,256:320,:64]
        x0_p52 = x0[:,:,256:320,64:128]
        x0_p53 = x0[:,:,256:320,128:192]
        x0_p54 = x0[:,:,256:320,192:256]
        x0_p55 = x0[:,:,256:320,256:320]
        x0_p56 = x0[:,:,256:320,320:384]
        x0_p57 = x0[:,:,256:320,384:448]
        x0_p58 = x0[:,:,256:320,448:]

        # Sixth row of patches
        center_p61 = center[:,:,40:48,:8]
        center_p62 = center[:,:,40:48,8:16]
        center_p63 = center[:,:,40:48,16:24]
        center_p64 = center[:,:,40:48,24:32]
        center_p65 = center[:,:,40:48,32:40]
        center_p66 = center[:,:,40:48,40:48] 
        center_p67 = center[:,:,40:48,48:56]
        center_p68 = center[:,:,40:48,56:]

        x2_p61 = x2[:,:,80:96,:16]
        x2_p62 = x2[:,:,80:96,16:32]
        x2_p63 = x2[:,:,80:96,32:48]
        x2_p64 = x2[:,:,80:96,48:64]
        x2_p65 = x2[:,:,80:96,64:80]
        x2_p66 = x2[:,:,80:96,80:96]
        x2_p67 = x2[:,:,80:96,96:112]
        x2_p68 = x2[:,:,80:96,112:]

        x1_p61 = x1[:,:,160:192,:32]
        x1_p62 = x1[:,:,160:192,32:64]
        x1_p63 = x1[:,:,160:192,64:96]
        x1_p64 = x1[:,:,160:192,96:128]
        x1_p65 = x1[:,:,160:192,128:160]
        x1_p66 = x1[:,:,160:192,160:192]
        x1_p67 = x1[:,:,160:192,192:224]
        x1_p68 = x1[:,:,160:192,224:]

        x0_p61 = x0[:,:,320:384,:64]
        x0_p62 = x0[:,:,320:384,64:128]
        x0_p63 = x0[:,:,320:384,128:192]
        x0_p64 = x0[:,:,320:384,192:256]
        x0_p65 = x0[:,:,320:384,256:320]
        x0_p66 = x0[:,:,320:384,320:384]
        x0_p67 = x0[:,:,320:384,384:448]
        x0_p68 = x0[:,:,320:384,448:]

        # Seventh row of patches
        center_p71 = center[:,:,48:56,:8]
        center_p72 = center[:,:,48:56,8:16]
        center_p73 = center[:,:,48:56,16:24]
        center_p74 = center[:,:,48:56,24:32]
        center_p75 = center[:,:,48:56,32:40]
        center_p76 = center[:,:,48:56,40:48] 
        center_p77 = center[:,:,48:56,48:56]
        center_p78 = center[:,:,48:56,56:]

        x2_p71 = x2[:,:,96:112,:16]
        x2_p72 = x2[:,:,96:112,16:32]
        x2_p73 = x2[:,:,96:112,32:48]
        x2_p74 = x2[:,:,96:112,48:64]
        x2_p75 = x2[:,:,96:112,64:80]
        x2_p76 = x2[:,:,96:112,80:96]
        x2_p77 = x2[:,:,96:112,96:112]
        x2_p78 = x2[:,:,96:112,112:]

        x1_p71 = x1[:,:,64:96,:32]
        x1_p72 = x1[:,:,64:96,32:64]
        x1_p73 = x1[:,:,64:96,64:96]
        x1_p74 = x1[:,:,64:96,96:128]
        x1_p75 = x1[:,:,64:96,128:160]
        x1_p76 = x1[:,:,64:96,160:192]
        x1_p77 = x1[:,:,64:96,192:224]
        x1_p78 = x1[:,:,64:96,224:]

        x0_p71 = x0[:,:,384:448,:64]
        x0_p72 = x0[:,:,384:448,64:128]
        x0_p73 = x0[:,:,384:448,128:192]
        x0_p74 = x0[:,:,384:448,192:256]
        x0_p75 = x0[:,:,384:448,256:320]
        x0_p76 = x0[:,:,384:448,320:384]
        x0_p77 = x0[:,:,384:448,384:448]
        x0_p78 = x0[:,:,384:448,448:]

        # Eigth row of patches
        center_p81 = center[:,:,56:,:8]
        center_p82 = center[:,:,56:,8:16]
        center_p83 = center[:,:,56:,16:24]
        center_p84 = center[:,:,56:,24:32]
        center_p85 = center[:,:,56:,32:40]
        center_p86 = center[:,:,56:,40:48] 
        center_p87 = center[:,:,56:,48:56]
        center_p88 = center[:,:,56:,56:]

        x2_p81 = x2[:,:,112:,:16]
        x2_p82 = x2[:,:,112:,16:32]
        x2_p83 = x2[:,:,112:,32:48]
        x2_p84 = x2[:,:,112:,48:64]
        x2_p85 = x2[:,:,112:,64:80]
        x2_p86 = x2[:,:,112:,80:96]
        x2_p87 = x2[:,:,112:,96:112]
        x2_p88 = x2[:,:,112:,112:]

        x1_p81 = x1[:,:,224:,:32]
        x1_p82 = x1[:,:,224:,32:64]
        x1_p83 = x1[:,:,224:,64:96]
        x1_p84 = x1[:,:,224:,96:128]
        x1_p85 = x1[:,:,224:,128:160]
        x1_p86 = x1[:,:,224:,160:192]
        x1_p87 = x1[:,:,224:,192:224]
        x1_p88 = x1[:,:,224:,224:]

        x0_p81 = x0[:,:,448:,:64]
        x0_p82 = x0[:,:,448:,64:128]
        x0_p83 = x0[:,:,448:,128:192]
        x0_p84 = x0[:,:,448:,192:256]
        x0_p85 = x0[:,:,448:,256:320]
        x0_p86 = x0[:,:,448:,320:384]
        x0_p87 = x0[:,:,448:,384:448]
        x0_p88 = x0[:,:,448:,448:]

        # First row of patches decode
        decode3_p11 = self.decode3_local(x2_p11, center_p11)
        decode2_p11 = self.decode2_local(x1_p11, decode3_p11)
        decode1_p11 = self.decode1_local(x0_p11, decode2_p11)
        output_p11 = self.out_local(decode1_p11)
        
        decode3_p12 = self.decode3_local(x2_p12, center_p12)
        decode2_p12 = self.decode2_local(x1_p12, decode3_p12)
        decode1_p12 = self.decode1_local(x0_p12, decode2_p12)
        output_p12 = self.out_local(decode1_p12)

        decode3_p13 = self.decode3_local(x2_p13, center_p13)
        decode2_p13 = self.decode2_local(x1_p13, decode3_p13)
        decode1_p13 = self.decode1_local(x0_p13, decode2_p13)
        output_p13 = self.out_local(decode1_p13)

        decode3_p14 = self.decode3_local(x2_p14, center_p14)
        decode2_p14 = self.decode2_local(x1_p14, decode3_p14)
        decode1_p14 = self.decode1_local(x0_p14, decode2_p14)
        output_p14 = self.out_local(decode1_p14)

        decode3_p15 = self.decode3_local(x2_p15, center_p15)
        decode2_p15 = self.decode2_local(x1_p15, decode3_p15)
        decode1_p15 = self.decode1_local(x0_p15, decode2_p15)
        output_p15 = self.out_local(decode1_p15)
        
        decode3_p16 = self.decode3_local(x2_p16, center_p16)
        decode2_p16 = self.decode2_local(x1_p16, decode3_p16)
        decode1_p16 = self.decode1_local(x0_p16, decode2_p16)
        output_p16 = self.out_local(decode1_p16)

        decode3_p17 = self.decode3_local(x2_p17, center_p17)
        decode2_p17 = self.decode2_local(x1_p17, decode3_p17)
        decode1_p17 = self.decode1_local(x0_p17, decode2_p17)
        output_p17 = self.out_local(decode1_p17)

        decode3_p18 = self.decode3_local(x2_p18, center_p18)
        decode2_p18 = self.decode2_local(x1_p18, decode3_p18)
        decode1_p18 = self.decode1_local(x0_p18, decode2_p18)
        output_p18 = self.out_local(decode1_p18)

        # Second row of patches decode
        decode3_p21 = self.decode3_local(x2_p21, center_p21)
        decode2_p21 = self.decode2_local(x1_p21, decode3_p21)
        decode1_p21 = self.decode1_local(x0_p21, decode2_p21)
        output_p21 = self.out_local(decode1_p21)
        
        decode3_p22 = self.decode3_local(x2_p22, center_p22)
        decode2_p22 = self.decode2_local(x1_p22, decode3_p22)
        decode1_p22 = self.decode1_local(x0_p22, decode2_p22)
        output_p22 = self.out_local(decode1_p22)

        decode3_p23 = self.decode3_local(x2_p23, center_p23)
        decode2_p23 = self.decode2_local(x1_p23, decode3_p23)
        decode1_p23 = self.decode1_local(x0_p23, decode2_p23)
        output_p23 = self.out_local(decode1_p23)

        decode3_p24 = self.decode3_local(x2_p24, center_p24)
        decode2_p24 = self.decode2_local(x1_p24, decode3_p24)
        decode1_p24 = self.decode1_local(x0_p24, decode2_p24)
        output_p24 = self.out_local(decode1_p24)

        decode3_p25 = self.decode3_local(x2_p25, center_p25)
        decode2_p25 = self.decode2_local(x1_p25, decode3_p25)
        decode1_p25 = self.decode1_local(x0_p25, decode2_p25)
        output_p25 = self.out_local(decode1_p25)
        
        decode3_p26 = self.decode3_local(x2_p26, center_p26)
        decode2_p26 = self.decode2_local(x1_p26, decode3_p26)
        decode1_p26 = self.decode1_local(x0_p26, decode2_p26)
        output_p26 = self.out_local(decode1_p26)

        decode3_p27 = self.decode3_local(x2_p27, center_p27)
        decode2_p27 = self.decode2_local(x1_p27, decode3_p27)
        decode1_p27 = self.decode1_local(x0_p27, decode2_p27)
        output_p27 = self.out_local(decode1_p27)

        decode3_p28 = self.decode3_local(x2_p28, center_p28)
        decode2_p28 = self.decode2_local(x1_p28, decode3_p28)
        decode1_p28 = self.decode1_local(x0_p28, decode2_p28)
        output_p28 = self.out_local(decode1_p28)

        # Third row of patches decode
        decode3_p31 = self.decode3_local(x2_p31, center_p31)
        decode2_p31 = self.decode2_local(x1_p31, decode3_p31)
        decode1_p31 = self.decode1_local(x0_p31, decode2_p31)
        output_p31 = self.out_local(decode1_p31)
        
        decode3_p32 = self.decode3_local(x2_p32, center_p32)
        decode2_p32 = self.decode2_local(x1_p32, decode3_p32)
        decode1_p32 = self.decode1_local(x0_p32, decode2_p32)
        output_p32 = self.out_local(decode1_p32)

        decode3_p33 = self.decode3_local(x2_p33, center_p33)
        decode2_p33 = self.decode2_local(x1_p33, decode3_p33)
        decode1_p33 = self.decode1_local(x0_p33, decode2_p33)
        output_p33 = self.out_local(decode1_p33)

        decode3_p34 = self.decode3_local(x2_p34, center_p34)
        decode2_p34 = self.decode2_local(x1_p34, decode3_p34)
        decode1_p34 = self.decode1_local(x0_p34, decode2_p34)
        output_p34 = self.out_local(decode1_p34)

        decode3_p35 = self.decode3_local(x2_p35, center_p35)
        decode2_p35 = self.decode2_local(x1_p35, decode3_p35)
        decode1_p35 = self.decode1_local(x0_p35, decode2_p35)
        output_p35 = self.out_local(decode1_p35)
        
        decode3_p36 = self.decode3_local(x2_p36, center_p36)
        decode2_p36 = self.decode2_local(x1_p36, decode3_p36)
        decode1_p36 = self.decode1_local(x0_p36, decode2_p36)
        output_p36 = self.out_local(decode1_p36)

        decode3_p37 = self.decode3_local(x2_p37, center_p37)
        decode2_p37 = self.decode2_local(x1_p37, decode3_p37)
        decode1_p37 = self.decode1_local(x0_p37, decode2_p37)
        output_p37 = self.out_local(decode1_p37)

        decode3_p38 = self.decode3_local(x2_p38, center_p38)
        decode2_p38 = self.decode2_local(x1_p38, decode3_p38)
        decode1_p38 = self.decode1_local(x0_p38, decode2_p38)
        output_p38 = self.out_local(decode1_p38)

        # Fourth row of patches decode
        decode3_p41 = self.decode3_local(x2_p41, center_p41)
        decode2_p41 = self.decode2_local(x1_p41, decode3_p41)
        decode1_p41 = self.decode1_local(x0_p41, decode2_p41)
        output_p41 = self.out_local(decode1_p41)
        
        decode3_p42 = self.decode3_local(x2_p42, center_p42)
        decode2_p42 = self.decode2_local(x1_p42, decode3_p42)
        decode1_p42 = self.decode1_local(x0_p42, decode2_p42)
        output_p42 = self.out_local(decode1_p42)

        decode3_p43 = self.decode3_local(x2_p43, center_p43)
        decode2_p43 = self.decode2_local(x1_p43, decode3_p43)
        decode1_p43 = self.decode1_local(x0_p43, decode2_p43)
        output_p43 = self.out_local(decode1_p43)

        decode3_p44 = self.decode3_local(x2_p44, center_p44)
        decode2_p44 = self.decode2_local(x1_p44, decode3_p44)
        decode1_p44 = self.decode1_local(x0_p44, decode2_p44)
        output_p44 = self.out_local(decode1_p44)

        decode3_p45 = self.decode3_local(x2_p45, center_p45)
        decode2_p45 = self.decode2_local(x1_p45, decode3_p45)
        decode1_p45 = self.decode1_local(x0_p45, decode2_p45)
        output_p45 = self.out_local(decode1_p45)
        
        decode3_p46 = self.decode3_local(x2_p46, center_p46)
        decode2_p46 = self.decode2_local(x1_p46, decode3_p46)
        decode1_p46 = self.decode1_local(x0_p46, decode2_p46)
        output_p46 = self.out_local(decode1_p46)

        decode3_p47 = self.decode3_local(x2_p47, center_p47)
        decode2_p47 = self.decode2_local(x1_p47, decode3_p47)
        decode1_p47 = self.decode1_local(x0_p47, decode2_p47)
        output_p47 = self.out_local(decode1_p47)

        decode3_p48 = self.decode3_local(x2_p48, center_p48)
        decode2_p48 = self.decode2_local(x1_p48, decode3_p48)
        decode1_p48 = self.decode1_local(x0_p48, decode2_p48)
        output_p48 = self.out_local(decode1_p48)

        # Fifth row of patches decode
        decode3_p51 = self.decode3_local(x2_p51, center_p51)
        decode2_p51 = self.decode2_local(x1_p51, decode3_p51)
        decode1_p51 = self.decode1_local(x0_p51, decode2_p51)
        output_p51 = self.out_local(decode1_p51)
        
        decode3_p52 = self.decode3_local(x2_p52, center_p52)
        decode2_p52 = self.decode2_local(x1_p52, decode3_p52)
        decode1_p52 = self.decode1_local(x0_p52, decode2_p52)
        output_p52 = self.out_local(decode1_p52)

        decode3_p53 = self.decode3_local(x2_p53, center_p53)
        decode2_p53 = self.decode2_local(x1_p53, decode3_p53)
        decode1_p53 = self.decode1_local(x0_p53, decode2_p53)
        output_p53 = self.out_local(decode1_p53)

        decode3_p54 = self.decode3_local(x2_p54, center_p54)
        decode2_p54 = self.decode2_local(x1_p54, decode3_p54)
        decode1_p54 = self.decode1_local(x0_p54, decode2_p54)
        output_p54 = self.out_local(decode1_p54)

        decode3_p55 = self.decode3_local(x2_p55, center_p55)
        decode2_p55 = self.decode2_local(x1_p55, decode3_p55)
        decode1_p55 = self.decode1_local(x0_p55, decode2_p55)
        output_p55 = self.out_local(decode1_p55)
        
        decode3_p56 = self.decode3_local(x2_p56, center_p56)
        decode2_p56 = self.decode2_local(x1_p56, decode3_p56)
        decode1_p56 = self.decode1_local(x0_p56, decode2_p56)
        output_p56 = self.out_local(decode1_p56)

        decode3_p57 = self.decode3_local(x2_p57, center_p57)
        decode2_p57 = self.decode2_local(x1_p57, decode3_p57)
        decode1_p57 = self.decode1_local(x0_p57, decode2_p57)
        output_p57 = self.out_local(decode1_p57)

        decode3_p58 = self.decode3_local(x2_p58, center_p58)
        decode2_p58 = self.decode2_local(x1_p58, decode3_p58)
        decode1_p58 = self.decode1_local(x0_p58, decode2_p58)
        output_p58 = self.out_local(decode1_p58)

        # Sixth row of patches decode
        decode3_p61 = self.decode3_local(x2_p61, center_p61)
        decode2_p61 = self.decode2_local(x1_p61, decode3_p61)
        decode1_p61 = self.decode1_local(x0_p61, decode2_p61)
        output_p61 = self.out_local(decode1_p61)
        
        decode3_p62 = self.decode3_local(x2_p62, center_p62)
        decode2_p62 = self.decode2_local(x1_p62, decode3_p62)
        decode1_p62 = self.decode1_local(x0_p62, decode2_p62)
        output_p62 = self.out_local(decode1_p62)

        decode3_p63 = self.decode3_local(x2_p63, center_p63)
        decode2_p63 = self.decode2_local(x1_p63, decode3_p63)
        decode1_p63 = self.decode1_local(x0_p63, decode2_p63)
        output_p63 = self.out_local(decode1_p63)

        decode3_p64 = self.decode3_local(x2_p64, center_p64)
        decode2_p64 = self.decode2_local(x1_p64, decode3_p64)
        decode1_p64 = self.decode1_local(x0_p64, decode2_p64)
        output_p64 = self.out_local(decode1_p64)

        decode3_p65 = self.decode3_local(x2_p65, center_p65)
        decode2_p65 = self.decode2_local(x1_p65, decode3_p65)
        decode1_p65 = self.decode1_local(x0_p65, decode2_p65)
        output_p65 = self.out_local(decode1_p65)
        
        decode3_p66 = self.decode3_local(x2_p66, center_p66)
        decode2_p66 = self.decode2_local(x1_p66, decode3_p66)
        decode1_p66 = self.decode1_local(x0_p66, decode2_p66)
        output_p66 = self.out_local(decode1_p66)

        decode3_p67 = self.decode3_local(x2_p67, center_p67)
        decode2_p67 = self.decode2_local(x1_p67, decode3_p67)
        decode1_p67 = self.decode1_local(x0_p67, decode2_p67)
        output_p67 = self.out_local(decode1_p67)

        decode3_p68 = self.decode3_local(x2_p68, center_p68)
        decode2_p68 = self.decode2_local(x1_p68, decode3_p68)
        decode1_p68 = self.decode1_local(x0_p68, decode2_p68)
        output_p68 = self.out_local(decode1_p68)

        # Seventh row of patches decode
        decode3_p71 = self.decode3_local(x2_p71, center_p71)
        decode2_p71 = self.decode2_local(x1_p71, decode3_p71)
        decode1_p71 = self.decode1_local(x0_p71, decode2_p71)
        output_p71 = self.out_local(decode1_p71)
        
        decode3_p72 = self.decode3_local(x2_p72, center_p72)
        decode2_p72 = self.decode2_local(x1_p72, decode3_p72)
        decode1_p72 = self.decode1_local(x0_p72, decode2_p72)
        output_p72 = self.out_local(decode1_p72)

        decode3_p73 = self.decode3_local(x2_p73, center_p73)
        decode2_p73 = self.decode2_local(x1_p73, decode3_p73)
        decode1_p73 = self.decode1_local(x0_p73, decode2_p73)
        output_p73 = self.out_local(decode1_p73)

        decode3_p74 = self.decode3_local(x2_p74, center_p74)
        decode2_p74 = self.decode2_local(x1_p74, decode3_p74)
        decode1_p74 = self.decode1_local(x0_p74, decode2_p74)
        output_p74 = self.out_local(decode1_p74)

        decode3_p75 = self.decode3_local(x2_p75, center_p75)
        decode2_p75 = self.decode2_local(x1_p75, decode3_p75)
        decode1_p75 = self.decode1_local(x0_p75, decode2_p75)
        output_p75 = self.out_local(decode1_p75)
        
        decode3_p76 = self.decode3_local(x2_p76, center_p76)
        decode2_p76 = self.decode2_local(x1_p76, decode3_p76)
        decode1_p76 = self.decode1_local(x0_p76, decode2_p76)
        output_p76 = self.out_local(decode1_p76)

        decode3_p77 = self.decode3_local(x2_p77, center_p77)
        decode2_p77 = self.decode2_local(x1_p77, decode3_p77)
        decode1_p77 = self.decode1_local(x0_p77, decode2_p77)
        output_p77 = self.out_local(decode1_p77)

        decode3_p78 = self.decode3_local(x2_p78, center_p78)
        decode2_p78 = self.decode2_local(x1_p78, decode3_p78)
        decode1_p78 = self.decode1_local(x0_p78, decode2_p78)
        output_p78 = self.out_local(decode1_p78)

        # Eigth row of patches decode
        decode3_p81 = self.decode3_local(x2_p81, center_p81)
        decode2_p81 = self.decode2_local(x1_p81, decode3_p81)
        decode1_p81 = self.decode1_local(x0_p81, decode2_p81)
        output_p81 = self.out_local(decode1_p81)
        
        decode3_p82 = self.decode3_local(x2_p82, center_p82)
        decode2_p82 = self.decode2_local(x1_p82, decode3_p82)
        decode1_p82 = self.decode1_local(x0_p82, decode2_p82)
        output_p82 = self.out_local(decode1_p82)

        decode3_p83 = self.decode3_local(x2_p83, center_p83)
        decode2_p83 = self.decode2_local(x1_p83, decode3_p83)
        decode1_p83 = self.decode1_local(x0_p83, decode2_p83)
        output_p83 = self.out_local(decode1_p83)

        decode3_p84 = self.decode3_local(x2_p84, center_p84)
        decode2_p84 = self.decode2_local(x1_p84, decode3_p84)
        decode1_p84 = self.decode1_local(x0_p84, decode2_p84)
        output_p84 = self.out_local(decode1_p84)

        decode3_p85 = self.decode3_local(x2_p85, center_p85)
        decode2_p85 = self.decode2_local(x1_p85, decode3_p85)
        decode1_p85 = self.decode1_local(x0_p85, decode2_p85)
        output_p85 = self.out_local(decode1_p85)
        
        decode3_p86 = self.decode3_local(x2_p86, center_p86)
        decode2_p86 = self.decode2_local(x1_p86, decode3_p86)
        decode1_p86 = self.decode1_local(x0_p86, decode2_p86)
        output_p86 = self.out_local(decode1_p86)

        decode3_p87 = self.decode3_local(x2_p87, center_p87)
        decode2_p87 = self.decode2_local(x1_p87, decode3_p87)
        decode1_p87 = self.decode1_local(x0_p87, decode2_p87)
        output_p87 = self.out_local(decode1_p87)

        decode3_p88 = self.decode3_local(x2_p88, center_p88)
        decode2_p88 = self.decode2_local(x1_p88, decode3_p88)
        decode1_p88 = self.decode1_local(x0_p88, decode2_p88)
        output_p88 = self.out_local(decode1_p88)

        row1 = torch.cat((output_p11, output_p12, output_p13, output_p14, output_p15, output_p16, output_p17, output_p18), axis=3)
        row2 = torch.cat((output_p21, output_p22, output_p23, output_p24, output_p25, output_p26, output_p27, output_p28), axis=3)
        row3 = torch.cat((output_p31, output_p32, output_p33, output_p34, output_p35, output_p36, output_p37, output_p38), axis=3)
        row4 = torch.cat((output_p41, output_p42, output_p43, output_p44, output_p45, output_p46, output_p47, output_p48), axis=3)

        row5 = torch.cat((output_p51, output_p52, output_p53, output_p54, output_p55, output_p56, output_p57, output_p58), axis=3)
        row6 = torch.cat((output_p61, output_p62, output_p63, output_p64, output_p65, output_p66, output_p67, output_p68), axis=3)
        row7 = torch.cat((output_p71, output_p72, output_p73, output_p74, output_p75, output_p76, output_p77, output_p78), axis=3)
        row8 = torch.cat((output_p81, output_p82, output_p83, output_p84, output_p85, output_p86, output_p87, output_p88), axis=3)
        
        output_local = torch.cat((row1, row2, row3, row4, row5, row6, row7, row8), axis=2)
        
        return output_global, output_local

if __name__ == "__main__":
    model = ThreshNetP64().to(device)
    summary(model)
    # test yout model
    image = torch.rand((1, 1, 512, 512)).to(device)
    out_global, out_local = model(image)
    print('Global Output Shape = ', out_global.shape)
    print('Local Output Shape = ', out_local.shape)

