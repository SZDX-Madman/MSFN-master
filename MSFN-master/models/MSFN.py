import torch
import torch.nn as nn

from triplet_attention import TripletAttention as TA
from blocks import upsample,downsample

class ResBlock(nn.Module):
    def __init__(self,Channels, kSize=3,):
        super(ResBlock, self).__init__()
        Ch = Channels
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv0 = nn.Conv2d(Ch, Ch, kSize, padding=(kSize-1)//2, stride=1,padding_mode='reflect')
        self.conv1_1=nn.Conv2d(Ch,Ch//2,kSize,padding=2*(kSize-1)//2,stride=1,dilation=2,padding_mode='reflect')
        self.conv1_2 = nn.Conv2d(Ch, Ch//2, kSize, padding=3 * (kSize - 1) // 2, stride=1, dilation=3,
                                 padding_mode='reflect')
        self.conv2_1 = nn.Conv2d(Ch*2, Ch, 1, padding=0, stride=1,padding_mode='reflect')
        self.conv2 = nn.Conv2d(Ch, Ch, kSize, padding=(kSize-1)//2, stride=1,padding_mode='reflect')
        self.conv3 = nn.Conv2d(Ch, Ch, kSize, padding=(kSize-1)//2, stride=1,padding_mode='reflect')
    def forward(self, x):
        t=self.relu(self.conv0(x))
        t2=self.relu(self.conv1_1(t))
        t3=self.relu(self.conv1_2(t))
        t=(self.conv2_1(torch.cat([t,t2,t3],1)))
        t=self.relu(self.conv2(t))
        t=self.relu(self.conv3(t))
        return x + t

class DenseBlock(nn.Module):
    def __init__(self, num=4,ch=128):
        super(DenseBlock, self).__init__()
        self.conv0 = ResBlock(ch)
        self.conv1 = ResBlock(ch)
        self.conv2 = ResBlock(ch)
        self.num=num
        if(num==4):
            self.conv3 = ResBlock(ch)
        self.LFF=nn.Sequential(
            TA(ch*num),
            nn.Conv2d(ch *num, ch, 1, padding=0, stride=1,)
        )


    def forward(self, x):
        res = []
        ox = x
        x=self.conv0(x)
        res.append(x)
        x = self.conv1(x)
        res.append(x)
        x = self.conv2(x)
        res.append(x)
        if(self.num==4):
            x = self.conv3(x)
            res.append(x)

        return self.LFF(torch.cat(res, 1)) + ox

class SKFF(nn.Module):
    def __init__(self, in_channels, height=2, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V


#MSFB
class MSFB(nn.Module):
    def __init__(self,block_num=3,in_ch=128,out_ch=128):
        super(MSFB, self).__init__()
        self.block_num=block_num
        self.scale1=downsample(in_ch,in_ch)
        self.scale2=upsample(in_ch,out_ch)
        self.conv1=nn.ModuleList()
        self.conv1.append(DenseBlock(4,in_ch))

        self.conv2=nn.ModuleList()
        self.conv2.append(DenseBlock(4,in_ch))
        for i in range(block_num-1):
            self.conv1.append(ScaleBlock1('up',in_ch,out_ch))
            self.conv2.append(ScaleBlock1('down',in_ch,out_ch))
        self.attention=nn.Sequential(TA(2 * in_ch),
                                     nn.Conv2d(2*in_ch,out_ch,1,1,0))

    def forward(self,x):
        out=[]
        out.append(self.conv1[0](x))
        out.append(self.conv2[0](self.scale1(x)))
        for i in range(1,self.block_num):
            d1=self.conv1[i](out[0],out[1])
            d2=self.conv2[i](out[1],out[0])
            out=[d1,d2]
        out[1]=self.scale2(out[1])
        out=self.attention(torch.cat(out,dim=1))
        return x+out
class ScaleBlock1(nn.Module):
    def __init__(self,type,in_ch=128,out_ch=128):
        super(ScaleBlock1, self).__init__()
        if(type=='up'):
            self.scale=upsample(in_ch,out_ch)
            self.conv = nn.Sequential(SKFF(in_ch),
                                      DenseBlock(4, in_ch))
        else:
            self.scale=downsample(in_ch,out_ch)

            self.conv=nn.Sequential(SKFF(in_ch),
                                    DenseBlock(4,in_ch))
    def forward(self,x1,x2):
        x2=self.scale(x2)
        return self.conv([x1,x2])
class ScaleBlock2(nn.Module):
    def __init__(self,type,in_ch=128,out_ch=128):
        super(ScaleBlock2, self).__init__()
        if(type=='up'):
            self.scale=nn.Sequential(upsample(in_ch,out_ch),
                                     # upsample(in_ch,out_ch)
                                     )
        else:
            self.scale=nn.Sequential(downsample(in_ch,out_ch),
                                     # downsample(in_ch,out_ch)
                                     )

        self.conv=nn.Sequential(SKFF(in_ch),
                                MSFB(3,in_ch,out_ch))
    def forward(self,x1,x2):
        x2=self.scale(x2)
        return self.conv([x1,x2])


class MSFN(nn.Module):
    def __init__(self,block_num=3,in_c=3,out_c=3,mid_c=128):
        super(MSFN, self).__init__()
        self.encoding=nn.Sequential(nn.Conv2d(in_c,mid_c,kernel_size=3,padding=1,stride=1,padding_mode='reflect'),
                                    nn.LeakyReLU(0.2,inplace=True),
                                    nn.Conv2d(mid_c,mid_c,kernel_size=3,padding=1,stride=2,padding_mode='reflect'),
                                    nn.LeakyReLU(0.2,inplace=True))
        self.down=nn.Sequential(downsample(mid_c,mid_c),
                                # downsample(mid_c,mid_c)
                                )

        self.upblocks=nn.ModuleList()
        self.upblocks.append(DenseBlock(4,mid_c))
        self.downblocks=nn.ModuleList()
        self.downblocks.append(DenseBlock(4,mid_c))
        self.block_num=block_num
        for i in range(block_num-1):

            self.upblocks.append(ScaleBlock2('up',mid_c,mid_c))
            self.downblocks.append(ScaleBlock2('down',mid_c,mid_c))
        self.up=nn.Sequential(upsample(mid_c,mid_c),
                              # upsample(mid_c,mid_c)
                                )
        self.att=nn.Sequential(TA(2 * mid_c),
                                     nn.Conv2d(2*mid_c,mid_c,1,1,0))
        self.decoding =nn.Sequential(
            # nn.ConvTranspose2d(Grate,Grate//2,3,stride=2,padding=1,output_padding=1),
            #nn.ConvTranspose2d(Grate//2, 3, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(mid_c,mid_c,3,padding=1,stride=2,output_padding=1),
            # nn.Conv2d(mid_c,mid_c//2,3,padding=1,stride=1,padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_c, mid_c//4, 3, padding=1, stride=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_c//4,out_c,3,padding=1,stride=1,padding_mode='reflect')
        )

    def forward(self,x):
        out=[]
        fea=self.encoding(x)

        out.append(self.upblocks[0](fea))
        out.append(self.downblocks[0](self.down(fea)))

        for i in range(1,self.block_num):
            d1=self.upblocks[i](out[0],out[1])
            d2=self.downblocks[i](out[1],out[0])
            out=[d1,d2]
        out[1]=self.up(out[1])
        out=self.att(torch.cat(out,dim=1))
        out=out+fea
        out=self.decoding(out)
        return out+x
class Wrapper(nn.Module):
    def __init__(self,):
        super().__init__()
        self.MSFN =MSFN()

    def forward(self, input):
        x = (input["input"])
        x = self.MSFN(x)
        # x = x + input["input"]
        return {"image": x}


def get():
    return Wrapper()