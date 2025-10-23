import torch.nn as nn
import torch


class Clamper(nn.Module):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min = min_val
        self.max = max_val
    def forward(self, x):
        return torch.clamp(x, self.min, self.max)

class UNet_pre(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder_block1 = EncoderBlock(in_channels=in_channels, out_channels=64)
        self.encoder_block2 = EncoderBlock(in_channels=64, out_channels=128)
        self.encoder_block3 = EncoderBlock(in_channels=128, out_channels=256)
        self.encoder_block4 = EncoderBlock(in_channels=256, out_channels=512)

        self.bridge = nn.Sequential(
            CNNBlock(in_channels=512, out_channels=1024),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, output_padding=(1,0))
        )

        self.decoder_block1 = DecoderBlock(in_channels=1024, out_channels1=512, out_channels2=256)
        self.decoder_block2 = DecoderBlock(in_channels=512, out_channels1=256, out_channels2=128)
        self.decoder_block3 = DecoderBlock(in_channels=256, out_channels1=128, out_channels2=64, output_padding=(0,1))

        self.last1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.last2 = nn.Conv2d(in_channels=64, out_channels=in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X_old1, X = self.encoder_block1(X)
        X_old2, X = self.encoder_block2(X)
        X_old3, X = self.encoder_block3(X)
        X_old4, X = self.encoder_block4(X)


        X = self.bridge(X)
        X = torch.cat([X_old4, X], dim=1)  # I exprect the X to have the shape ( (batch) x C x H x W)
        X = self.decoder_block1(X)
        X = torch.cat([X_old3, X], dim=1)
        X = self.decoder_block2(X)
        X = torch.cat([X_old2, X], dim=1)
        X = self.decoder_block3(X)
        X = torch.cat([X_old1, X], dim=1)
        X = self.last1(X)
        X = self.last2(X)


        return self.sigmoid(X)

class UNet(nn.Module):  # Input: [(Batch) x 1 X 362 X 362] -> [(Batch) x 1 x 362 x 362]
    def __init__(self, in_channels):
        super().__init__()
        self.encoder_block1 = EncoderBlock(in_channels=in_channels, out_channels=64)
        self.encoder_block2 = EncoderBlock(in_channels=64, out_channels=128)
        self.encoder_block3 = EncoderBlock(in_channels=128, out_channels=256)
        self.encoder_block4 = EncoderBlock(in_channels=256, out_channels=512)

        self.bridge = nn.Sequential(
            CNNBlock(in_channels=512, out_channels=1024),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, output_padding=(1,1))
        )

        self.decoder_block1 = DecoderBlock(in_channels=1024, out_channels1=512, out_channels2=256)
        self.decoder_block2 = DecoderBlock(in_channels=512, out_channels1=256, out_channels2=128, output_padding=(1,1))
        self.decoder_block3 = DecoderBlock(in_channels=256, out_channels1=128, out_channels2=64)

        self.last1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.last2 = nn.Conv2d(in_channels=64, out_channels=in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X_old1, X = self.encoder_block1(X)
        X_old2, X = self.encoder_block2(X)
        X_old3, X = self.encoder_block3(X)
        X_old4, X = self.encoder_block4(X)


        X = self.bridge(X)
        X = torch.cat([X_old4, X], dim=1)  # I exprect the X to have the shape ( (batch) x C x H x W)
        X = self.decoder_block1(X)
        X = torch.cat([X_old3, X], dim=1)
        X = self.decoder_block2(X)
        X = torch.cat([X_old2, X], dim=1)
        X = self.decoder_block3(X)
        X = torch.cat([X_old1, X], dim=1)
        X = self.last1(X)
        X = self.last2(X)

        return self.sigmoid(X)

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

    def forward(self, X):
        return self.layer(X)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = CNNBlock(in_channels=in_channels, out_channels=out_channels)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, X):
        X_old = self.conv(X)  # for skip connections
        P = self.pool(X_old)
        return X_old, P


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, output_padding=0):
        super().__init__()
        self.conv = CNNBlock(in_channels=in_channels, out_channels=out_channels1)
        self.upsample = nn.ConvTranspose2d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=2,
                                           stride=2, output_padding=output_padding)

    def forward(self, X):
        X = self.conv(X)
        return self.upsample(X)


####################

class TrainableFourierSeries(nn.Module):
    def __init__(self, freqs, init_filter, L=50):
        super().__init__()

        # Step23: Store Fourier Series Coefficients
        a, b, a_0 = self.cos_sin_coeffs(init_filter, L)

        # Step 3: Normalize the frequencies
        freqs_range = freqs.max() - freqs.min()
        normalized_freqs = (freqs - freqs.min()) / freqs_range

        # Step 4: Create cos and sin matrices
        i = torch.arange(1, L + 1, dtype=normalized_freqs.dtype)
        cos_terms = torch.cos(2 * torch.pi * normalized_freqs.unsqueeze(1) * i)
        sin_terms = torch.sin(2 * torch.pi * normalized_freqs.unsqueeze(1) * i)

        self.register_buffer('cos_sin_stuff', torch.cat([cos_terms, sin_terms], dim=1))
        self.coeffs = torch.nn.Parameter(torch.cat([a, b]), requires_grad=True)
        self.const = torch.nn.Parameter(a_0, requires_grad=True)

    def forward(self, X):
        filter = self.const + torch.matmul(self.cos_sin_stuff, self.coeffs)
        filter = torch.fft.fftshift(filter)
        return filter.unsqueeze(0).unsqueeze(0).unsqueeze(0) # (1,1,1,513)

    def cos_sin_coeffs(self, f, L):
        N = f.shape[0]

        # Step 1: Perform the FFT
        fft_result = torch.fft.fft(f)

        # Step 2: Extract the coefficients
        a0 = fft_result[0].real / N

        # Initialize lists for a_i and b_i
        a_i_list = []
        b_i_list = []

        for i in range(1, L + 1):
            # a_i is derived from the real part
            ai = (2 / N) * fft_result[i].real
            a_i_list.append(ai.item())

            # b_i is derived from the imaginary part
            bi = (-2 / N) * fft_result[i].imag
            b_i_list.append(bi.item())

        a_i_list = torch.Tensor(a_i_list)
        b_i_list = torch.Tensor(b_i_list)

        return a_i_list, b_i_list, a0


#############
class LearnableWindow(nn.Module):
    def __init__(self, init_tensor=torch.ones(513)):
        super().__init__()
        self.weights = nn.Parameter(init_tensor)  # or random init
    def forward(self, x):
        return self.weights.unsqueeze(0).unsqueeze(0).unsqueeze(0)
##############

class LearnableWindowII(nn.Module):
    def __init__(self, init_tensor=torch.ones((1000,513))):
        super(LearnableWindowII, self).__init__()
        self.weights = nn.Parameter(init_tensor)
    def forward(self, x):
        return self.weights.unsqueeze(0).unsqueeze(0) # (1,1,1000,513)


class PseudoResnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = CNNBlock(in_channels=1, out_channels=3)
        self.conv2 = CNNBlock(in_channels=3, out_channels=9)
        self.conv3 = CNNBlock(in_channels=9, out_channels=27)
        self.conv4 = CNNBlock(in_channels=27, out_channels=81)
        self.conv5 = CNNBlock(in_channels=81, out_channels=81)
        self.conv6 = CNNBlock(in_channels=81, out_channels=27)
        self.conv6 = CNNBlock(in_channels=27, out_channels=9)
        self.conv7 = CNNBlock(in_channels=9, out_channels=3)
        self.conv8 = CNNBlock(in_channels=3, out_channels=1)

    def forward(self, X):
        X1 = self.conv1(X)
        X2 = self.conv2(X1)
        X3 = self.conv3(X2)
        X4 = self.conv4(X3)

        X5 = self.conv5(X4)

        X6 = self.conv6(X5 + X4)
        X7 = self.conv7(X6 + X3)
        X8 = self.conv8(X7 + X2)


        return X8 + X1
    

class UNet_no_activation(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder_block1 = EncoderBlock(in_channels=in_channels, out_channels=64)
        self.encoder_block2 = EncoderBlock(in_channels=64, out_channels=128)
        self.encoder_block3 = EncoderBlock(in_channels=128, out_channels=256)
        self.encoder_block4 = EncoderBlock(in_channels=256, out_channels=512)

        self.bridge = nn.Sequential(
            CNNBlock(in_channels=512, out_channels=1024),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, output_padding=(1,1))
        )

        self.decoder_block1 = DecoderBlock(in_channels=1024, out_channels1=512, out_channels2=256)
        self.decoder_block2 = DecoderBlock(in_channels=512, out_channels1=256, out_channels2=128, output_padding=(1,1))
        self.decoder_block3 = DecoderBlock(in_channels=256, out_channels1=128, out_channels2=64)

        self.last1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.last2 = nn.Conv2d(in_channels=64, out_channels=in_channels, kernel_size=1)

    def forward(self, X):
        X_old1, X = self.encoder_block1(X)
        X_old2, X = self.encoder_block2(X)
        X_old3, X = self.encoder_block3(X)
        X_old4, X = self.encoder_block4(X)


        X = self.bridge(X)
        X = torch.cat([X_old4, X], dim=1)  # I exprect the X to have the shape ( (batch) x C x H x W)
        X = self.decoder_block1(X)
        X = torch.cat([X_old3, X], dim=1)
        X = self.decoder_block2(X)
        X = torch.cat([X_old2, X], dim=1)
        X = self.decoder_block3(X)
        X = torch.cat([X_old1, X], dim=1)
        X = self.last1(X)
        X = self.last2(X)

        return X
    
class UNet_pre_no_activation(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder_block1 = EncoderBlock(in_channels=in_channels, out_channels=64)
        self.encoder_block2 = EncoderBlock(in_channels=64, out_channels=128)
        self.encoder_block3 = EncoderBlock(in_channels=128, out_channels=256)
        self.encoder_block4 = EncoderBlock(in_channels=256, out_channels=512)

        self.bridge = nn.Sequential(
            CNNBlock(in_channels=512, out_channels=1024),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, output_padding=(1,0))
        )

        self.decoder_block1 = DecoderBlock(in_channels=1024, out_channels1=512, out_channels2=256)
        self.decoder_block2 = DecoderBlock(in_channels=512, out_channels1=256, out_channels2=128)
        self.decoder_block3 = DecoderBlock(in_channels=256, out_channels1=128, out_channels2=64, output_padding=(0,1))

        self.last1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.last2 = nn.Conv2d(in_channels=64, out_channels=in_channels, kernel_size=1)

    def forward(self, X):
        X_old1, X = self.encoder_block1(X)
        X_old2, X = self.encoder_block2(X)
        X_old3, X = self.encoder_block3(X)
        X_old4, X = self.encoder_block4(X)


        X = self.bridge(X)
        X = torch.cat([X_old4, X], dim=1)  # I exprect the X to have the shape ( (batch) x C x H x W)
        X = self.decoder_block1(X)
        X = torch.cat([X_old3, X], dim=1)
        X = self.decoder_block2(X)
        X = torch.cat([X_old2, X], dim=1)
        X = self.decoder_block3(X)
        X = torch.cat([X_old1, X], dim=1)
        X = self.last1(X)
        X = self.last2(X)


        return X