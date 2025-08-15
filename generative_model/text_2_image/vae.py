import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 超参数设置
batch_size = 128
latent_dim = 20
epochs = 10
lr = 1e-3

# 数据加载与预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 将像素值归一化到[-1, 1]
])

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# VAE模型定义
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 2 * latent_dim)  # 输出均值和对数方差
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Tanh()  # 输出范围[-1, 1]，匹配输入归一化
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 编码
        x = x.view(-1, 784)
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)  # 分割为均值和对数方差

        # 重参数化采样
        z = self.reparameterize(mu, logvar)

        # 解码
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


# 初始化模型和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)


# 损失函数：重构损失 + KL散度
def loss_function(recon_x, x, mu, logvar):
    # 重构损失（MSE或BCE，这里使用MSE）
    recon_loss = nn.MSELoss(reduction='sum')(recon_x, x.view(-1, 784))

    # KL散度项（公式：-0.5 * sum(1 + logvar - mu^2 - exp(logvar))）
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss


# 训练循环
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_loader.dataset):.4f}")

# 生成新样本
model.eval()
with torch.no_grad():
    # 从标准正态分布采样潜在变量
    z = torch.randn(16, latent_dim).to(device)
    generated = model.decoder(z).cpu()
    generated = generated.view(-1, 28, 28).numpy()

# 可视化生成的图像
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(generated[i], cmap='gray')
    ax.axis('off')
plt.show()
