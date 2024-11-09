import torch
from torch import nn
import math
from attontion import Spatial_Module, Spectral_Module,Spatial_X_Module,Spatial_Y_Module

class SFC_back(nn.Module):
    # Spectral feature conversion (SFC)
    def __init__(self, c1, c2, act):  # ch_in, ch_out, kernel, stride, padding, groups
        super(SFC_back, self).__init__()
        # self.conv = Conv(c1 // 4, c2, k, s, p, g, act)
        self.pad_d=int(math.ceil(c1/4)*4)
        self.act=act
        

        self.conv1=nn.Sequential(nn.Conv3d(in_channels=self.pad_d//4, out_channels=c2,kernel_size=(3,3,1), stride=(1, 1, 1),padding=(1,1,0)),
                                nn.BatchNorm3d(c2,  eps=0.00001, momentum=0.1, affine=True), 
                                self.act) 
        self.conv2=nn.Sequential(nn.Conv3d(in_channels=24, out_channels=24,kernel_size=(3,3,1), stride=(1, 1, 1),padding=(1,1,0)),
                                nn.BatchNorm3d(24,  eps=0.00001, momentum=0.1, affine=True), 
                                self.act) 

    def forward(self, x):
        b,_,w,h,d=x.shape                          
        pad_d=self.pad_d
        x=x.permute(0,1,4,2,3)                       

        if pad_d>d:
            temp=torch.zeros((b,_,pad_d,w,h)).cuda()
            temp[:,:,:d,...]=x
            temp[:,:,d:,...]=1e-8
            x=temp
            del temp
        
        xx=torch.zeros((b,_,pad_d//4,2*w,2*h)).cuda()   

        xx[..., ::2, ::2]   = x[:,:,:pad_d//4,...]
        xx[..., 1::2, ::2]  = x[:,:,pad_d//4:pad_d//2,...]
        xx[..., ::2, 1::2]  = x[:,:,pad_d//2:3*pad_d//4,...]
        xx[..., 1::2, 1::2] = x[:,:,3*pad_d//4:pad_d,...]

        xx=xx.permute(0,2,3,4,1)                   
        xx=self.conv1(xx)                          
        xx=xx.permute(0,4,2,3,1)                  
        x=self.conv2(xx)
        return x


class MBSFC_l(nn.Module):
    def __init__(self, band, classes):
        super(MBSFC_l, self).__init__()
        self.name = 'Focus_l'
        self.band=band

        out_depth=int( math.ceil( math.ceil(band/4) ) )

        # act=nn.Hardswish()
        # act=nn.LeakyReLU()
        # act=nn.SiLU()
        act = nn.Mish()
        # act = nn.ReLU()

        self.conv_feature=nn.Sequential(
                                        nn.Conv3d(in_channels=1, out_channels=24,kernel_size=(1, 1, 7), stride=(1, 1, 1),padding=(0,0,3)),
                                        nn.BatchNorm3d(24,  eps=0.00001, momentum=0.1, affine=True), 
                                        act,
                                        ) 

        self.stem= SFC_back(self.band,out_depth,act=act)

        kernel_3d=out_depth

        # spectral Branch
        self.conv11 =nn.Sequential(nn.Conv3d(in_channels=24, out_channels=12, padding=(0, 0, 3),kernel_size=(1, 1, 7), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,  eps=0.00001, momentum=0.1, affine=True), 
                                    act)

        self.conv12 =nn.Sequential(nn.Conv3d(in_channels=36, out_channels=12, padding=(0, 0, 3),kernel_size=(1, 1, 7), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,  eps=0.00001, momentum=0.1, affine=True), 
                                    act)
        
        self.conv13 =nn.Sequential(nn.Conv3d(in_channels=48, out_channels=12, padding=(0, 0, 3),kernel_size=(1, 1, 7), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,  eps=0.00001, momentum=0.1, affine=True), 
                                    act)

        


        self.conv14 =nn.Sequential(nn.Conv3d(in_channels=60, out_channels=60, padding=(0, 0, 0),kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(60,  eps=0.00001, momentum=0.1, affine=True), 
                                    # act,
                                    # nn.Dropout(p=0.5)
                                    )

        # Spatial Branch x
        self.conv21 =nn.Sequential(nn.Conv3d(in_channels=24, out_channels=12, padding=(2, 0, 0),kernel_size=(5, 1, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,  eps=0.00001, momentum=0.1, affine=True), 
                                    act)

        self.conv22 =nn.Sequential(nn.Conv3d(in_channels=36, out_channels=12, padding=(2, 0, 0),kernel_size=(5, 1, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,  eps=0.00001, momentum=0.1, affine=True), 
                                    act)
        
        self.conv23 =nn.Sequential(nn.Conv3d(in_channels=48, out_channels=12, padding=(2, 0, 0),kernel_size=(5, 1, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,  eps=0.00001, momentum=0.1, affine=True), 
                                    act)

        self.conv24 =nn.Sequential(nn.Conv3d(in_channels=60, out_channels=60, padding=(0, 0, 0),kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(60,  eps=0.00001, momentum=0.1, affine=True), 
                                    # mish()
                                    )

        # Spatial Branch y
        self.conv31 =nn.Sequential(nn.Conv3d(in_channels=24, out_channels=12, padding=(0, 2, 0),kernel_size=(1, 5, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,  eps=0.00001, momentum=0.1, affine=True), 
                                    act)

        self.conv32 =nn.Sequential(nn.Conv3d(in_channels=36, out_channels=12, padding=(0, 2, 0),kernel_size=(1, 5, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,  eps=0.00001, momentum=0.1, affine=True), 
                                    act)
        
        self.conv33 =nn.Sequential(nn.Conv3d(in_channels=48, out_channels=12, padding=(0, 2, 0),kernel_size=(1, 5, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,  eps=0.00001, momentum=0.1, affine=True), 
                                    act)

        self.conv34 =nn.Sequential(nn.Conv3d(in_channels=60, out_channels=60, padding=(0, 0, 0),kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(60,  eps=0.00001, momentum=0.1, affine=True), 
                                    # mish()
                                    )

        self.batch_norm_spectral = nn.Sequential(
                                    nn.BatchNorm3d(60,  eps=0.00001, momentum=0.1, affine=True),
                                    act,
                                    nn.Dropout(p=0.5)
                                    
                                    )
        
        self.batch_norm_spatial_x = nn.Sequential(
                                    nn.BatchNorm3d(60,  eps=0.00001, momentum=0.1, affine=True),
                                    act,
                                    nn.Dropout(p=0.5)
                                    )
        
        self.batch_norm_spatial_y = nn.Sequential(
                                    nn.BatchNorm3d(60,  eps=0.00001, momentum=0.1, affine=True),
                                    act,
                                    nn.Dropout(p=0.5)
                                    )


        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
                                # nn.Dropout(p=0.5),
                                nn.Linear(180, classes) # ,
                                # nn.Softmax()
                                 )

        self.attention_spectral = Spectral_Module(60)
        self.attention_spatial_x = Spatial_X_Module(60)
        self.attention_spatial_y = Spatial_Y_Module(60)

    def forward(self, x):
        x=self.conv_feature(x)    
        x=self.stem(x)            

        # spectral
        x11 = self.conv11(x)
        x12 = torch.cat((x, x11), dim=1)
        x12 = self.conv12(x12)
        x13 = torch.cat((x, x11, x12), dim=1)
        x13 = self.conv13(x13)
        x14 = torch.cat((x, x11, x12, x13), dim=1)
        x14 = self.conv14(x14)

        # 光谱注意力通道
        x1 = self.attention_spectral(x14)
        x1 = torch.mul(x1, x14)

        # spatial x
        x21 = self.conv21(x)
        x22 = torch.cat((x, x21), dim=1)
        x22 = self.conv22(x22)
        x23 = torch.cat((x, x21, x22), dim=1)
        x23 = self.conv23(x23)
        x24 = torch.cat((x, x21, x22, x23), dim=1)
        x24 = self.conv24(x24)

        # 空间x注意力机制 
        x2 = self.attention_spatial_x(x24)
        x2 = torch.mul(x2, x24)

        # spatial y
        x31 = self.conv31(x)
        x32 = torch.cat((x, x31), dim=1)
        x32 = self.conv32(x32)
        x33 = torch.cat((x, x31, x32), dim=1)
        x33 = self.conv33(x33)
        x34 = torch.cat((x, x31, x32, x33), dim=1)
        x34 = self.conv34(x34)

        # 空间y注意力机制 
        x3 = self.attention_spatial_y(x34)
        x3 = torch.mul(x3, x34)

        # model1
        x1 = self.batch_norm_spectral(x1)
        x1 = self.global_pooling(x1)
        x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        
        x2 = self.batch_norm_spatial_x(x2)
        x2= self.global_pooling(x2)
        x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)

        x3 = self.batch_norm_spatial_y(x3)
        x3= self.global_pooling(x3)
        x3 = x3.squeeze(-1).squeeze(-1).squeeze(-1)

        x_pre = torch.cat((x1, x2, x3), dim=1)

        output = self.full_connection(x_pre)

        return output


class MBSFC_m(nn.Module):
    def __init__(self, band, classes):
        super(MBSFC_m, self).__init__()
        self.name = 'Focus_m'
        self.band=band

        out_depth=int( math.ceil( math.ceil(band/4)/2 ) )

        # act=nn.Hardswish()
        # act=nn.LeakyReLU()
        # act=nn.SiLU()
        act = nn.Mish()
        # act = nn.ReLU()

        self.conv_feature=nn.Sequential(
                                        nn.Conv3d(in_channels=1, out_channels=24,kernel_size=(1, 1, 7), stride=(1, 1, 1),padding=(0,0,3)),
                                        nn.BatchNorm3d(24,  eps=0.00001, momentum=0.1, affine=True), 
                                        act,
                                        ) 

        self.stem= SFC_back(self.band,out_depth,act=act)

        kernel_3d=out_depth

        # spectral Branch
        self.conv11 =nn.Sequential(nn.Conv3d(in_channels=24, out_channels=12, padding=(0, 0, 3),kernel_size=(1, 1, 7), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,  eps=0.00001, momentum=0.1, affine=True), 
                                    act)

        self.conv12 =nn.Sequential(nn.Conv3d(in_channels=36, out_channels=12, padding=(0, 0, 3),kernel_size=(1, 1, 7), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,  eps=0.00001, momentum=0.1, affine=True), 
                                    act)
        
        self.conv13 =nn.Sequential(nn.Conv3d(in_channels=48, out_channels=12, padding=(0, 0, 3),kernel_size=(1, 1, 7), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,  eps=0.00001, momentum=0.1, affine=True), 
                                    act)

        


        self.conv14 =nn.Sequential(nn.Conv3d(in_channels=60, out_channels=60, padding=(0, 0, 0),kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(60,  eps=0.00001, momentum=0.1, affine=True), 
                                    )


        # Spatial Branch
        self.conv21 = nn.Sequential(nn.Conv3d(in_channels=24, out_channels=12,padding=(0, 0, 3),kernel_size=(1, 1, 7), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12, eps=0.00001, momentum=0.1, affine=True),
                                    act
                                )
        
        self.conv22 = nn.Sequential(nn.Conv3d(in_channels=36, out_channels=12, padding=(2, 2, 0),kernel_size=(5, 5, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12, eps=0.00001, momentum=0.1, affine=True),
                                    act
                                )
        
        self.conv23 = nn.Sequential(nn.Conv3d(in_channels=48, out_channels=12, padding=(2, 2, 0),kernel_size=(5, 5, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12, eps=0.00001, momentum=0.1, affine=True),
                                    act
                                )

        self.conv24 = nn.Sequential(nn.Conv3d(in_channels=60, out_channels=60, padding=(0, 0, 0),kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(60, eps=0.00001, momentum=0.1, affine=True),

        )
        

        self.batch_norm_spectral = nn.Sequential(
                                    nn.BatchNorm3d(60,  eps=0.00001, momentum=0.1, affine=True),
                                    act,
                                    nn.Dropout(p=0.5)
                                    )
        
        self.batch_norm_spatial = nn.Sequential(
                                    nn.BatchNorm3d(60,  eps=0.00001, momentum=0.1, affine=True),
                                    act,
                                    nn.Dropout(p=0.5)
                                    )

        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
                                #nn.Dropout(p=0.5),
                                nn.Linear(120, classes) # ,
                                # nn.Softmax()
        )

        self.attention_spectral = Spectral_Module(60)
        self.attention_spatial = Spatial_Module(60)

    def forward(self, x):
        x=self.conv_feature(x)    
        x=self.stem(x)           

        # spectral
        x11 = self.conv11(x)
        x12 = torch.cat((x, x11), dim=1)
        x12 = self.conv12(x12)
        x13 = torch.cat((x, x11, x12), dim=1)
        x13 = self.conv13(x13)
        x14 = torch.cat((x, x11, x12, x13), dim=1)
        x14 = self.conv14(x14)

        # 光谱注意力通道
        x1 = self.attention_spectral(x14)
        x1 = torch.mul(x1, x14)

        # spatial
        x21 = self.conv21(x)
        x22 = torch.cat((x, x21), dim=1)
        x22 = self.conv22(x22)
        x23 = torch.cat((x, x21, x22), dim=1)
        x23 = self.conv23(x23)
        x24 = torch.cat((x, x21, x22, x23), dim=1)
        x24 = self.conv24(x24)

        # 空间x注意力机制 
        x2 = self.attention_spatial(x24)
        x2 = torch.mul(x2, x24)

        # model1
        x1 = self.batch_norm_spectral(x1)
        x1 = self.global_pooling(x1)
        x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        
        x2 = self.batch_norm_spatial(x2)
        x2= self.global_pooling(x2)
        x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)

        x_pre = torch.cat((x1, x2), dim=1)

        output = self.full_connection(x_pre)

        return output

class MBSFC_s(nn.Module):
    def __init__(self, band, classes):
        super(MBSFC_s, self).__init__()
        self.name = 'Focus_s'
        self.band=band

        out_depth=int( math.ceil( math.ceil(band/4)/4 ) )

        # act=nn.Hardswish()
        # act=nn.LeakyReLU()
        # act=nn.SiLU()
        act = nn.Mish()
        # act = nn.ReLU()

        self.conv_feature=nn.Sequential(
                                        nn.Conv3d(in_channels=1, out_channels=24,kernel_size=(1, 1, 7), stride=(1, 1, 1),padding=(0,0,3)),
                                        nn.BatchNorm3d(24,  eps=0.00001, momentum=0.1, affine=True), 
                                        act,
                                        ) 

        self.stem= SFC_back(self.band,out_depth,act=act)

        kernel_3d=out_depth

        # spectral Branch
        self.conv11 =nn.Sequential(nn.Conv3d(in_channels=24, out_channels=12, padding=(0, 0, 3),kernel_size=(1, 1, 7), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,  eps=0.00001, momentum=0.1, affine=True), 
                                    act)

        self.conv12 =nn.Sequential(nn.Conv3d(in_channels=36, out_channels=12, padding=(0, 0, 3),kernel_size=(1, 1, 7), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,  eps=0.00001, momentum=0.1, affine=True), 
                                    act)
        
        self.conv13 =nn.Sequential(nn.Conv3d(in_channels=48, out_channels=12, padding=(0, 0, 3),kernel_size=(1, 1, 7), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,  eps=0.00001, momentum=0.1, affine=True), 
                                    act)

        


        self.conv14 =nn.Sequential(nn.Conv3d(in_channels=60, out_channels=60, padding=(0, 0, 0),kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(60,  eps=0.00001, momentum=0.1, affine=True), 
                                    # act,
                                    # nn.Dropout(p=0.5)
                                    )

        

        self.batch_norm_spectral = nn.Sequential(
                                    nn.BatchNorm3d(60,  eps=0.00001, momentum=0.1, affine=True),
                                    act,
                                    nn.Dropout(p=0.5)
                                    
                                    )

        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
                                # nn.Dropout(p=0.5),
                                nn.Linear(60, classes) # ,
                                # nn.Softmax()
                                 )

        self.attention_spectral = Spectral_Module(60)

    def forward(self, x):
        x=self.conv_feature(x)   
        x=self.stem(x)           

        # spectral
        x11 = self.conv11(x)
        x12 = torch.cat((x, x11), dim=1)
        x12 = self.conv12(x12)
        x13 = torch.cat((x, x11, x12), dim=1)
        x13 = self.conv13(x13)
        x14 = torch.cat((x, x11, x12, x13), dim=1)
        x14 = self.conv14(x14)

        # 光谱注意力通道
        x1 = self.attention_spectral(x14)
        x1 = torch.mul(x1, x14)

        # model1
        x1 = self.batch_norm_spectral(x1)
        x1 = self.global_pooling(x1)
        x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        
        output = self.full_connection(x1)

        return output



