import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo

b_condition=True
text_dim = 1024
z_dim = 100
embedding_dim = 128
r_num = 2
gf_dim = 64
df_dim = 64

class INCEPTION_V3(nn.Module):
    def __init__(self):
        super(INCEPTION_V3, self).__init__()
        self.model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        # print(next(model.parameters()).data)
        state_dict = \
            model_zoo.load_url(url, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        for param in self.model.parameters():
            param.requires_grad = False
        print('Load pretrained inception_v3 model from', url)
        # print(next(self.model.parameters()).data)
        # print(self.model)

    def forward(self, input):
        # [-1.0, 1.0] --> [0, 1.0]
        x = input * 0.5 + 0.5
        # mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        # --> mean = 0, std = 1
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225
        #
        # --> fixed-size input: batch x 3 x 299 x 299
        # x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
        # 299 x 299 x 3
        x = self.model(x)
        x = nn.Softmax(dim=-1)(x)
        return x

# Generator
class G_NET(nn.Module):
    def __init__(
        self, 
        gf_dim=64,
        z_dim=100,
        r_num=2,
        levels=3,
        text_dim=1024,
        embedding_dim=128,
        b_condition=True,
        ca=True):
        
        super(G_NET, self).__init__()
        self.gf_dim = gf_dim
        self.z_dim = z_dim
        self.r_num = r_num
        self.levels = levels
        self.gf_dim = gf_dim
        self.text_dim = text_dim
        self.embedding_dim = embedding_dim
        self.b_condition = b_condition
        self.ca = ca
        self.define_module()

    def define_module(self):
        if self.ca and self.b_condition:
            self.ca_net = CA_NET(text_dim=self.text_dim, embedding_dim=self.embedding_dim)

        if self.levels > 0:
            self.h_net1 = INIT_STAGE_G(ngf=self.gf_dim * 16, b_condition=self.b_condition)
            self.img_net1 = GET_IMAGE_G(self.gf_dim)
        if self.levels > 1:
            self.h_net2 = NEXT_STAGE_G(self.gf_dim)
            self.img_net2 = GET_IMAGE_G(self.gf_dim // 2)
        if self.levels > 2:
            self.h_net3 = NEXT_STAGE_G(self.gf_dim // 2)
            self.img_net3 = GET_IMAGE_G(self.gf_dim // 4)
        if self.levels > 3: # Recommended structure (mainly limited by GPU memory), and not test yet
            self.h_net4 = NEXT_STAGE_G(self.gf_dim // 4, num_residual=1)
            self.img_net4 = GET_IMAGE_G(self.gf_dim // 8)
        if self.levels > 4:
            self.h_net4 = NEXT_STAGE_G(self.gf_dim // 8, num_residual=1)
            self.img_net4 = GET_IMAGE_G(self.gf_dim // 16)

    def forward(self, z_code, text_embedding=None):
        if self.b_condition and text_embedding is not None:
            c_code, mu, logvar = self.ca_net(text_embedding)
        else:
            c_code, mu, logvar = z_code, None, None
        fake_imgs = []
        if self.levels > 0:
            h_code1 = self.h_net1(z_code, c_code)
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)
        if self.levels > 1:
            h_code2 = self.h_net2(h_code1, c_code)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
        if self.levels > 2:
            h_code3 = self.h_net3(h_code2, c_code)
            fake_img3 = self.img_net3(h_code3)
            fake_imgs.append(fake_img3)
        if self.levels > 3:
            h_code4 = self.h_net4(h_code3, c_code)
            fake_img4 = self.img_net4(h_code4)
            fake_imgs.append(fake_img4)

        return fake_imgs, mu, logvar


class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self, text_dim=1024, embedding_dim=128):
        super(CA_NET, self).__init__()
        self.t_dim = text_dim
        self.ef_dim = embedding_dim
        self.fc = nn.Linear(self.t_dim, self.ef_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.ef_dim]
        logvar = x[:, self.ef_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(mu.device)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        if self.training:
            c_code = self.reparametrize(mu, logvar)
            return c_code, mu, logvar
        else:
           return mu, mu, logvar


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, "channels can't divide by 2!"
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, b_condition):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.b_condition = b_condition
        if self.b_condition:
            self.in_dim = z_dim + embedding_dim
        else:
            self.in_dim = z_dim
        self.define_module()

    def define_module(self):
        in_dim = self.in_dim
        ngf = self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())


        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z_code, c_code=None):
        if self.b_condition and c_code is not None:
            in_code = torch.cat((c_code, z_code), 1)
        else:
            in_code = z_code
        # state size 16ngf x 4 x 4
        out_code = self.fc(in_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size 8ngf x 8 x 8
        out_code = self.upsample1(out_code)
        # state size 4ngf x 16 x 16
        out_code = self.upsample2(out_code)
        # state size 2ngf x 32 x 32
        out_code = self.upsample3(out_code)
        # state size ngf x 64 x 64
        out_code = self.upsample4(out_code)

        return out_code


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, num_residual=r_num):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        if b_condition:
            self.ef_dim = embedding_dim
        else:
            self.ef_dim = z_dim
        self.num_residual = num_residual
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        efg = self.ef_dim

        self.jointConv = Block3x3_relu(ngf + efg, ngf)
        self.residual = self._make_layer(ResBlock, ngf)
        self.upsample = upBlock(ngf, ngf // 2)

    def forward(self, h_code, c_code):
        s_size = h_code.size(2)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, s_size, s_size)
        # state size (ngf+egf) x in_size x in_size
        h_c_code = torch.cat((c_code, h_code), 1)
        # state size ngf x in_size x in_size
        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)

        return out_code


class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


class ScaleUp(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super(ScaleUp, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode)
        return x


def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        ScaleUp(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block

# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block

class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


# ************************************************
# Discriminator
# ************************************************
# For 64 x 64 images
class D_NET64(nn.Module):
    def __init__(self):
        super(D_NET64, self).__init__()
        self.df_dim = df_dim
        self.ef_dim = embedding_dim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        if b_condition:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)

        if b_condition and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((c_code, x_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        if b_condition:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]


# For 128 x 128 images
class D_NET128(nn.Module):
    def __init__(self):
        super(D_NET128, self).__init__()
        self.df_dim = df_dim
        self.ef_dim = embedding_dim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        if b_condition:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
            self.uncond_logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s32_1(x_code)

        if b_condition and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((c_code, x_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        if b_condition:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]


# For 256 x 256 images
class D_NET256(nn.Module):
    def __init__(self):
        super(D_NET256, self).__init__()
        self.df_dim = df_dim
        self.ef_dim = embedding_dim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s64_1 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s64_2 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        if b_condition:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s64(x_code)
        x_code = self.img_code_s64_1(x_code)
        x_code = self.img_code_s64_2(x_code)

        if b_condition and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((c_code, x_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        if b_condition:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]


def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 16
def encode_image_by_16times(ndf):
    encode_img = nn.Sequential(
        # --> state size. ndf x in_size/2 x in_size/2
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 2ndf x x in_size/4 x in_size/4
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 4ndf x in_size/8 x in_size/8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 8ndf x in_size/16 x in_size/16
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img

if __name__ == '__main__':
    class Args: pass
    args = Args()
    args.device='cuda'
    args.z_dim=100
    args.input_dim=1024
    args.levels=3

    bs = 2
    device=args.device
    G = G_NET(gf_dim=64, z_dim=args.z_dim, r_num=2, levels=args.levels, b_condition=True, ca=True).to(device)
    G = nn.DataParallel(G)
    Ds = []
    imgs = []
    Ds.append(D_NET64())
    imgs.append(torch.randn(bs, 3, 64, 64).to(device))
    if args.levels >= 2:
        Ds.append(D_NET128())
        imgs.append(torch.randn(bs, 3, 128, 128).to(device))
    if args.levels >= 3:
        Ds.append(D_NET256())
        imgs.append(torch.randn(bs, 3, 256, 256).to(device))
    for i in range(len(Ds)):
        Ds[i] = nn.DataParallel(Ds[i].to(device))
        
    noise = torch.randn(bs, args.z_dim).to(device)
    txt_feat = torch.randn(bs, args.input_dim).to(device)
    fake_imgs, mu, logvar = G(noise, txt_feat)
    print(len(fake_imgs))
    print(fake_imgs[2].shape)
    print(mu.shape, logvar.shape)
    
    for level in range(3):
        D = Ds[level]
        img = imgs[level]
        real_logits = D(img, mu.detach())
        print(real_logits[0].shape, real_logits[1].shape)