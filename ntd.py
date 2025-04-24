import os
import torch
import time
import numpy as np
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from torch import nn
from scipy.io import loadmat
from scipy import signal

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_flag = 0
if gpu_flag == 1:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

start_time = time.time()

def Gaussian_downsampleT(x, psf, s):
    if isinstance(psf, np.ndarray):
        psf = torch.tensor(psf, dtype=x.dtype, device=x.device)
    psf = psf.unsqueeze(0).unsqueeze(0)
    psf = psf.expand(x.size(1), 1, psf.size(2), psf.size(3))
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
    elif x.ndim == 3:
        x = x.unsqueeze(1)  # [C, H, W] -> [1, C, H, W]
    elif x.ndim == 4:
        pass
    x2 = F.conv2d(x, psf, padding='same', groups=x.size(1))
    y = x2[:, :, ::s, ::s]

    return y

def Gaussian_downsampleX(x, psf, s):
    psf = torch.tensor(psf, dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
    elif x.ndim == 3:
        x = x.unsqueeze(1)  # [N, H, W] -> [N, 1, H, W]
    x2 = F.conv2d(x, psf, padding='same')
    y = x2[:, :, 0::s, :]

    return y.squeeze(1)  # [N, 1, H, W] -> [N, H, W]

def Gaussian_downsample(x, psf, s):
    x_np = x.detach().cpu().numpy()
    psf_np = psf if isinstance(psf, np.ndarray) else psf.detach().cpu().numpy()
    y = np.zeros((x_np.shape[0], int(x_np.shape[1] / s), int(x_np.shape[2])))

    if x_np.ndim == 2:
        x_np = np.expand_dims(x_np, axis=0)

    for i in range(x_np.shape[0]):
        x1 = x_np[i, :, :]
        x2 = signal.convolve2d(x1, psf_np, boundary='symm', mode='same')
        y[i, :, :] = x2[0::s, 0::]

    return torch.tensor(y, dtype=torch.complex64, device=x.device)

def down_sample(u, s0=0, scale=4):
    u = torch.fft.ifft(torch.fft.fft(u, dim=-2), dim=-2)
    u = u[:, s0::scale, :]
    return u

def psnr(img1, img2, dynamic_range=1.0):
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    mse = np.mean((img1_ - img2_) ** 2)
    if mse <= 1e-10:
        return np.inf
    return 20 * np.log10(dynamic_range / np.sqrt(mse))

def mpsnr(img1, img2, dynamic_range=1.0):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    psnr_values = []
    for i in range(img1.shape[2]):  # Assuming the third dimension represents bands
        psnr_values.append(psnr(img1[:, :, i], img2[:, :, i], dynamic_range))
    
    return np.mean(psnr_values)

def mrmse(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    rmse_values = []
    for i in range(img1.shape[2]):  # Assuming the third dimension represents bands
        rmse_values.append(rmse(img1[:, :, i], img2[:, :, i]))
    
    return np.mean(rmse_values)

def ssim_band(img1, img2, data_range=1.0):
    return ssim(img1, img2, data_range=data_range)

def mssim(img1, img2, data_range=1.0):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    ssim_values = []
    for i in range(img1.shape[2]):
        ssim_values.append(ssim_band(img1[:, :, i], img2[:, :, i], data_range))
    
    return np.mean(ssim_values)

def ergas(img1, img2, ratio=4):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    h, w, b = img1.shape
    mean_ref = np.mean(img1, axis=(0, 1))
    mse = np.mean((img1 - img2) ** 2, axis=(0, 1))
    ergas_value = 100 * ratio * np.sqrt(np.mean(mse / (mean_ref ** 2)))
    
    return ergas_value

def nergas(img_fake, img_real, scale=4):
    if not img_fake.shape == img_real.shape:
        raise ValueError('Input images must have the same dimensions.')
    img_fake_ = img_fake.astype(np.float64)
    img_real_ = img_real.astype(np.float64)
    if img_fake_.ndim == 2:
        mean_real = img_real_.mean()
        mse = np.mean((img_fake_ - img_real_)**2)
        return 100 / scale * np.sqrt(mse / (mean_real**2 + np.finfo(np.float64).eps))
    elif img_fake_.ndim == 3:
        means_real = img_real_.reshape(-1, img_real_.shape[2]).mean(axis=0)
        mses = ((img_fake_ - img_real_)**2).reshape(-1, img_fake_.shape[2]).mean(axis=0)
        return 100 / scale * np.sqrt((mses / (means_real**2 + np.finfo(np.float64).eps)).mean())
    else:
        raise ValueError('Wrong input image dimensions.')

def rmse(gt, result):
    gt = gt * 255
    result = result * 255
    mse = np.mean((gt - result) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def sam(array1, array2):
    norm1 = np.linalg.norm(array1, axis=-1, keepdims=True)
    norm2 = np.linalg.norm(array2, axis=-1, keepdims=True)

    array1_norm = array1 / (norm1 + 1e-10)
    array2_norm = array2 / (norm2 + 1e-10)

    dot_product = np.sum(array1_norm * array2_norm, axis=-1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    average_sam = np.mean(angle_deg)

    return average_sam

train_data = loadmat("./imgb9.mat")
hr_msi = torch.tensor(train_data['MSI'].transpose([2, 1, 0]), dtype=torch.float32).unsqueeze(0).to(device)
lr_hsi = torch.tensor(train_data['HSI'].transpose([2, 1, 0]), dtype=torch.float32).unsqueeze(0).to(device)
ground_truth = torch.tensor(train_data['S'], dtype=torch.float32)
D_c = torch.tensor(train_data['F'], dtype=torch.float32).to(device)
kernel = torch.tensor(train_data['psf'], dtype=torch.float32).to(device)

class Model(nn.Module):
    def __init__(self, channel, height, width, rank):
        super(Model, self).__init__()
        self.ps = [1, 1, 1, 1]
        self.channel = channel
        self.height = height
        self.width = width
        self.rank = rank
        conv1_1, conv1_2, conv1_3 = self.conv_generation()
        self.conv1_1 = conv1_1
        self.conv1_2 = conv1_2
        self.conv1_3 = conv1_3
        self.pool = nn.AdaptiveAvgPool2d(self.ps[0])

    def forward(self, v):
        channel_forward = self.pool(v)
        height_forward = self.pool(v.permute(0, 3, 1, 2).contiguous())
        width_forward = self.pool(v.permute(0, 2, 3, 1).contiguous())
        c_out, h_out, w_out = [], [], []

        for i in range(0, self.rank):
            c_out.append(self.conv1_1[i](channel_forward))
            h_out.append(self.conv1_2[i](height_forward))
            w_out.append(self.conv1_3[i](width_forward))

        channel_forward = torch.cat(c_out, -1).squeeze(-2)
        height_forward = torch.cat(h_out, -1).squeeze(-2)
        width_forward = torch.cat(w_out, -1).squeeze(-2)
        return channel_forward, height_forward, width_forward  # C x rank, H x rank, W x rank

    def conv_generation(self):
        n = 1
        conv1 = []
        for _ in range(0, self.rank):
            conv1.append(nn.Sequential(
                nn.Conv2d(self.channel, self.channel // n, kernel_size=1, bias=False),
                nn.Sigmoid()))
        conv1 = nn.ModuleList(conv1)

        conv2 = []
        for _ in range(0, self.rank):
            conv2.append(nn.Sequential(
                nn.Conv2d(self.height, self.height // n, kernel_size=1, bias=False),
                nn.Sigmoid()))
        conv2 = nn.ModuleList(conv2)

        conv3 = []
        for _ in range(0, self.rank):
            conv3.append(nn.Sequential(
                nn.Conv2d(self.width, self.width // n, kernel_size=1, bias=False),
                nn.Sigmoid()))
        conv3 = nn.ModuleList(conv3)
        return conv1, conv2, conv3


net_hr_msi = Model(3, 256, 256, rank=8).to(device)
net_lr_hsi = Model(31, 64, 64, rank=8).to(device)
option = torch.optim.Adam(list(net_hr_msi.parameters()) + list(net_lr_hsi.parameters()), lr=0.0001)

for epoch in range(100):
    c, H, W = net_hr_msi(hr_msi)
    C, h, w = net_lr_hsi(lr_hsi)
    #import pdb; pdb.set_trace()
    hr_msi_hat = torch.einsum(
        '...cr, ...hr, ...wr-> ...chw', torch.einsum('...cr, ic-> ...ir', C, D_c), H, W)
    lr_hsi_hat = torch.einsum(
        '...cr, ...hr, ...wr-> ...chw', C, Gaussian_downsampleX(H, kernel, 4), Gaussian_downsampleX(W, kernel, 4))
    channel_res = torch.einsum(
            '...cr, ...rk-> ...ck', D_c, C)
    fusion_t = torch.einsum(
     '...cr, ...hr, ...wr-> ...chw', C, H, W)
    
    res_fusion = torch.einsum('cr, ...rhw->...chw', D_c, fusion_t)
    alpha = 1.0
    beta = 1.0
    loss_fusion = alpha * torch.norm(Gaussian_downsampleT(fusion_t, kernel, 4) - lr_hsi, p=1) + beta * torch.norm(res_fusion - hr_msi, p=1)
    loss_input = alpha * torch.norm(hr_msi_hat.squeeze() - hr_msi, p=1) + beta * torch.norm(lr_hsi_hat.squeeze() - lr_hsi, p=1)
    loss_components = torch.norm(channel_res - c, p=1) + torch.norm(Gaussian_downsampleX(H, kernel, 4) - h, p=1) + torch.norm(Gaussian_downsampleX(W, kernel, 4) - w, p=1)
    loss = loss_fusion + loss_input + loss_components
    option.zero_grad()
    loss.backward()
    option.step()
    fusion = torch.einsum(
      '...cr, ...hr, ...wr-> ...chw', C, H, W).data.cpu().squeeze().detach().numpy().transpose([2, 1, 0])
    x = np.array(ground_truth)
    y = np.array(fusion)

    if (epoch % 10) == 0:
        current_mpsnr = mpsnr(x, y, dynamic_range=1.0)
        current_mrmse = mrmse(x, y)
        current_sam = sam(x, y)
        current_mssim = mssim(x, y, data_range=1.0)
        current_nergas = nergas(x, y)
        print(f'Epoch: {epoch:.0f}, Loss: {loss.item():.4f}, PSNR: {current_mpsnr:.4f}, RMSE: {current_mrmse:.4f}, SAM: {current_sam:.4f}, SSIM: {current_mssim:.4f}, ERGAS: {current_nergas:.4f}')

end_time = time.time()
elapsed_time = end_time - start_time
print('\n-----------Results Report-----------\n')
print('Dataset: imgb9, Rank: 512\n')
print(f"Starting Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
print(f"Ending Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
print(f"Execution Time: {elapsed_time:.4f}s\n")
print(f'HR-MSI Size: {hr_msi.data.cpu().squeeze().detach().numpy().transpose([2, 1, 0]).shape}')
print(f'LR-HSI Size: {lr_hsi.data.cpu().squeeze().detach().numpy().transpose([2, 1, 0]).shape}')
print(f'HR-HSI Size: {fusion.shape}')