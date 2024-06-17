import os
import math
import torch
import time
import numpy as np
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from torch import nn
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_flag = 1
if gpu_flag == 1:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

start_time = time.time()
# def Gaussian_downsampleX(x,psf,s):
#     x_np = x.detach().cpu().numpy()
#     y=np.zeros((x_np.shape[0], int(x_np.shape[1]/s), int(x_np.shape[2])))
#     if x.ndim==2:
#         x=np.expand_dims(x,axis=0)
#     for i in range(x_np.shape[0]):
#         x1=x[i,:,:]
#         x2=signal.convolve2d(x1,psf, boundary='symm',mode='same')
#         y[i,:,:]=x2[0::s,0::]
#     return torch.tensor(y, dtype=torch.complex64, device=x.device)

# def Gaussian_downsampleT(x, psf, s):
#     psf = torch.tensor(psf, dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)

#     if x.ndim == 2:
#         x = x.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
#     elif x.ndim == 3:
#         x = x.unsqueeze(1)  # [N, H, W] -> [N, 1, H, W]

#     #padding = (psf.shape[2] // 2, psf.shape[3] // 2)  
#     x2 = F.conv2d(x, psf, padding='same')

#     y = x2[:, :, 0::s, 0::s]

#     return y.squeeze(1)  # [N, 1, H, W] -> [N, H, W]
    
def Gaussian_downsampleT(x, psf, s):
    # 如果输入是 numpy 数组，将其转换为 PyTorch tensor
    if isinstance(psf, np.ndarray):
        psf = torch.tensor(psf, dtype=x.dtype, device=x.device)

    # 确保 psf 具有正确的形状 [out_channels, in_channels/groups, kernel_height, kernel_width]
    psf = psf.unsqueeze(0).unsqueeze(0)  # 形状变为 [1, 1, kernel_height, kernel_width]
    psf = psf.expand(x.size(1), 1, psf.size(2), psf.size(3))  # 形状变为 [in_channels, 1, kernel_height, kernel_width]

    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
    elif x.ndim == 3:
        x = x.unsqueeze(1)  # [C, H, W] -> [1, C, H, W]
    elif x.ndim == 4:
        pass  # [N, C, H, W] 直接使用

    # Apply Gaussian blur
    x2 = F.conv2d(x, psf, padding='same', groups=x.size(1))

    # Downsample first two dimensions
    y = x2[:, :, ::s, ::s]

    return y

def Gaussian_downsampleX(x, psf, s):
    psf = torch.tensor(psf, dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)

    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
    elif x.ndim == 3:
        x = x.unsqueeze(1)  # [N, H, W] -> [N, 1, H, W]

    #padding = (psf.shape[2] // 2, psf.shape[3] // 2)  
    x2 = F.conv2d(x, psf, padding='same')

    y = x2[:, :, 0::s, :]

    return y.squeeze(1)  # [N, 1, H, W] -> [N, H, W]

def Gaussian_downsample(x, psf, s):
    # 将输入的PyTorch张量转换为NumPy数组，并确保不追踪梯度
    x_np = x.detach().cpu().numpy()
    psf_np = psf if isinstance(psf, np.ndarray) else psf.detach().cpu().numpy()
    # 初始化输出数组
    y = np.zeros((x_np.shape[0], int(x_np.shape[1] / s), int(x_np.shape[2])))

    # 如果输入是2D的，扩展为3D
    if x_np.ndim == 2:
        x_np = np.expand_dims(x_np, axis=0)

    # 对每个通道进行卷积和下采样
    for i in range(x_np.shape[0]):
        x1 = x_np[i, :, :]
        x2 = signal.convolve2d(x1, psf_np, boundary='symm', mode='same')
        y[i, :, :] = x2[0::s, 0::]

    # 将结果转换回PyTorch张量，并返回
    return torch.tensor(y, dtype=torch.complex64, device=x.device)



def down_sample(u, s0=0, scale=4):
    u = torch.fft.ifft(torch.fft.fft(u, dim=-2), dim=-2)
    u = u[:, s0::scale, :]
    return u


# def psnr(gt, result):
#     mse = np.mean((gt - result) ** 2)
#     if mse == 0:
#         return 100
#     pixel_max = 1.0
#     return 20 * math.log10(pixel_max / math.sqrt(mse))

def psnr(img1, img2, dynamic_range=1.0):
    """PSNR metric, assuming images are in the range [0, 1]"""
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    mse = np.mean((img1_ - img2_) ** 2)
    if mse <= 1e-10:
        return np.inf
    return 20 * np.log10(dynamic_range / np.sqrt(mse))

def mpsnr(img1, img2, dynamic_range=1.0):
    """Mean Peak Signal-to-Noise Ratio for hyperspectral images"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    psnr_values = []
    for i in range(img1.shape[2]):  # Assuming the third dimension represents bands
        psnr_values.append(psnr(img1[:, :, i], img2[:, :, i], dynamic_range))
    
    return np.mean(psnr_values)

# def rmse(img1, img2):
#     """Root Mean Squared Error"""
#     if not img1.shape == img2.shape:
#         raise ValueError('Input images must have the same dimensions.')
#     return np.sqrt(np.mean((img1 - img2) ** 2))

def mrmse(img1, img2):
    """Mean Root Mean Squared Error for hyperspectral images"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    rmse_values = []
    for i in range(img1.shape[2]):  # Assuming the third dimension represents bands
        rmse_values.append(rmse(img1[:, :, i], img2[:, :, i]))
    
    return np.mean(rmse_values)

def ssim_band(img1, img2, data_range=1.0):
    """SSIM for a single band"""
    return ssim(img1, img2, data_range=data_range)

def mssim(img1, img2, data_range=1.0):
    """Mean Structural Similarity Index for hyperspectral images"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    ssim_values = []
    for i in range(img1.shape[2]):
        ssim_values.append(ssim_band(img1[:, :, i], img2[:, :, i], data_range))
    
    return np.mean(ssim_values)

def ergas(img1, img2, ratio=4):
    """ERGAS metric for hyperspectral images"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    h, w, b = img1.shape
    mean_ref = np.mean(img1, axis=(0, 1))
    mse = np.mean((img1 - img2) ** 2, axis=(0, 1))
    ergas_value = 100 * ratio * np.sqrt(np.mean(mse / (mean_ref ** 2)))
    
    return ergas_value

# def sam(img1, img2):
#     """Spectral Angle Mapper for hyperspectral images"""
#     if not img1.shape == img2.shape:
#         raise ValueError('Input images must have the same dimensions.')
    
#     h, w, b = img1.shape
#     img1_reshaped = img1.reshape(-1, b)
#     img2_reshaped = img2.reshape(-1, b)
    
#     dot_product = np.sum(img1_reshaped * img2_reshaped, axis=1)
#     norm1 = np.linalg.norm(img1_reshaped, axis=1)
#     norm2 = np.linalg.norm(img2_reshaped, axis=1)
#     cos_theta = dot_product / (norm1 * norm2)
#     sam_angles = np.arccos(np.clip(cos_theta, -1, 1))
    
#     return np.mean(sam_angles)

def rmse(gt, result):
    gt = gt * 255
    result = result * 255
    mse = np.mean((gt - result) ** 2)
    rmse = np.sqrt(mse)
    # return np.sqrt(((predictions - targets) ** -1).mean())
    return rmse



def sam(array1, array2):
    # 计算每个光谱向量的范数
    norm1 = np.linalg.norm(array1, axis=-1, keepdims=True)
    norm2 = np.linalg.norm(array2, axis=-1, keepdims=True)

    # 正规化数组，避免除以零
    array1_norm = array1 / (norm1 + 1e-10)
    array2_norm = array2 / (norm2 + 1e-10)

    # 计算正规化向量之间的点积
    dot_product = np.sum(array1_norm * array2_norm, axis=-1)

    # 确保余弦值在有效范围内
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # 计算角度（弧度制）
    angle_rad = np.arccos(dot_product)

    # 将角度从弧度转换为度
    angle_deg = np.degrees(angle_rad)

    # 计算所有像素的 SAM 值的平均值
    average_sam = np.mean(angle_deg)

    return average_sam

# 初始化最佳值
best_mpsnr = -np.inf
best_mrmse = np.inf
best_sam = np.inf
best_mssim = -np.inf
best_ergas = np.inf

# 记录最佳epoch
best_mpsnr_epoch = 0
best_mrmse_epoch = 0
best_sam_epoch = 0
best_mssim_epoch = 0
best_ergas_epoch = 0

train_data = loadmat("./Cave_4.mat")
hr_msi = torch.tensor(train_data['MSI'].transpose([2, 1, 0]), dtype=torch.float32).unsqueeze(0).to(device)
lr_hsi = torch.tensor(train_data['HSI'].transpose([2, 1, 0]), dtype=torch.float32).unsqueeze(0).to(device)
ground_truth = torch.tensor(train_data['S'], dtype=torch.float32)
D_c = torch.tensor(train_data['F'], dtype=torch.float32).to(device)
kernel = torch.tensor(train_data['psf'], dtype=torch.float32).to(device)
srf = [
     [0.005, 0.007, 0.012, 0.015, 0.023, 0.025, 0.030, 0.026, 0.024, 0.019,
        0.010, 0.004, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.002, 0.003, 0.005, 0.007,
        0.012, 0.013, 0.015, 0.016, 0.017, 0.02, 0.013, 0.011, 0.009, 0.005,
        0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.003],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
        0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.003, 0.010, 0.012, 0.013, 0.022,
        0.020, 0.020, 0.018, 0.017, 0.016, 0.016, 0.014, 0.014, 0.013]]

#D_c = torch.tensor(srf, dtype=torch.float32).to(device)

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
        #self.lam = nn.Parameter(torch.empty(self.rank), requires_grad=True)
        #self.lam.data = torch.ones(self.rank)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.kaiming_normal_(m.weight.data, mode='fan_out')

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


net_hr_msi = Model(3, 256, 256, rank=8192).to(device)
net_lr_hsi = Model(31, 64, 64, rank=8192).to(device)
option = torch.optim.Adam(list(net_hr_msi.parameters()) + list(net_lr_hsi.parameters()), lr=0.0001)
writer = SummaryWriter(log_dir='./runs/harvard')

for epoch in range(50001):
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

    L_fusionHSI = torch.norm(Gaussian_downsampleT(fusion_t, kernel, 4) - lr_hsi)
    L_fusionMSI = torch.norm(res_fusion - hr_msi)
    L_inputMSI = torch.norm(hr_msi_hat.squeeze() - hr_msi)
    L_inputHSI = torch.norm(lr_hsi_hat.squeeze() - lr_hsi)
    L_componentS = torch.norm(channel_res - c)
    L_componentH =torch.norm(Gaussian_downsampleX(H, kernel, 4) - h)
    L_componentW = torch.norm(Gaussian_downsampleX(W, kernel, 4) - w)

    #loss_fusion = torch.norm(Gaussian_downsampleT(fusion_t, kernel, 4) - lr_hsi) + torch.norm(res_fusion - hr_msi)
    #loss_input = torch.norm(hr_msi_hat.squeeze() - hr_msi) + torch.norm(lr_hsi_hat.squeeze() - lr_hsi)
    #loss_components = torch.norm(channel_res - c) + torch.norm(Gaussian_downsampleX(H, kernel, 4) - h) + torch.norm(Gaussian_downsampleX(W, kernel, 4) - w)

    loss = L_fusionHSI + L_inputHSI + L_inputMSI + 0.35 * (L_componentS + L_componentW + L_componentH)

    option.zero_grad()
    loss.backward()
    option.step()
    fusion = torch.einsum(
      '...cr, ...hr, ...wr-> ...chw', C, H, W).data.cpu().squeeze().detach().numpy().transpose([2, 1, 0])
    x = np.array(ground_truth)
    y = np.array(fusion)
    # if (epoch%50) == 0:
    #     print(f'Epoch: {epoch:.0f}, Loss: {loss.item():.3f}, MPSNR: {mpsnr(x, y, dynamic_range=1.0):.3f}, RMSE: {mrmse(x, y):.3f}, SAM: {sam(x, y):.3f}, SSIM: {mssim(x, y, data_range=1.0):.3f}, ERGAS: {ergas(x, y):.3f}')
    if (epoch % 10) == 0:
        current_mpsnr = mpsnr(x, y, dynamic_range=1.0)
        current_mrmse = mrmse(x, y)
        current_sam = sam(x, y)
        current_mssim = mssim(x, y, data_range=1.0)
        current_ergas = ergas(x, y)

        print(f'Epoch: {epoch:.0f}, Loss: {loss.item():.4f}, PSNR: {current_mpsnr:.4f}, RMSE: {current_mrmse:.4f}, SAM: {current_sam:.4f}, SSIM: {current_mssim:.4f}, ERGAS: {current_ergas:.4f}')
        print(f'Epoch: {epoch:.0f}, L_fusionMSI: {L_fusionMSI:.4f}, L_inputMSI: {L_inputMSI:.4f}, L_fusionHSI: {L_fusionHSI:.4f}, L_inputHSI: {L_inputHSI:.4f}, L_componentS: {L_componentS:.4f}, L_componentH: {L_componentH:.4f}, L_componentW: {L_componentW:.4f}\n')

        if current_mpsnr > best_mpsnr:
            best_mpsnr = current_mpsnr
            best_mpsnr_epoch = epoch

        if current_mrmse < best_mrmse:
            best_mrmse = current_mrmse
            best_mrmse_epoch = epoch

        if current_sam < best_sam:
            best_sam = current_sam
            best_sam_epoch = epoch

        if current_mssim > best_mssim:
            best_mssim = current_mssim
            best_mssim_epoch = epoch

        if current_ergas < best_ergas:
            best_ergas = current_ergas
            best_ergas_epoch = epoch

    writer.add_scalar('LOSS', loss.item(), epoch)
    writer.add_scalar('PSNR', psnr(x, y), epoch)
    writer.add_scalar('RMSE', rmse(x, y), epoch)

writer.close()
end_time = time.time()
elapsed_time = end_time - start_time
print('\n-----------Results Report-----------\n')
print('Dataset: Harvard, Rank: 8000, Loss Setting: 10:3.5\n')
print(f"Starting Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
print(f"Ending Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
print(f"Execution Time: {elapsed_time:.4f}s\n")
print(f'Best PSNR: {best_mpsnr:.4f} at epoch {best_mpsnr_epoch}')
print(f'Best RMSE: {best_mrmse:.4f} at epoch {best_mrmse_epoch}')
print(f'Best SAM: {best_sam:.4f} at epoch {best_sam_epoch}')
print(f'Best SSIM: {best_mssim:.4f} at epoch {best_mssim_epoch}')
print(f'Best ERGAS: {best_ergas:.4f} at epoch {best_ergas_epoch}\n')
#print(f'HR-MSI Size: {hr_msi.data.cpu().squeeze().detach().numpy().transpose([2, 1, 0]).shape}')
#print(f'LR-HSI Size: {lr_hsi.data.cpu().squeeze().detach().numpy().transpose([2, 1, 0]).shape}')
#print(f'HR-HSI Size: {fusion.shape}')
plt.title('Obtained HR-HSI')
plt.imshow(fusion[:, :, [1, 16, 30]])
plt.savefig('./c_r8000_10:3.5.pdf', format='pdf', dpi=300)
#plt.show()
