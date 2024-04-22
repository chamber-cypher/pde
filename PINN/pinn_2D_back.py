import torch
import matplotlib.pyplot as plt
from poprogress import simple_progress

h = 100
N = 1000
N1 = 100
N2 = 10

checkpoint_path = 'model_3.pth'  
epochs = 10000


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(22)


loss = torch.nn.MSELoss()

def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), 
                                   create_graph=True,
                                   only_inputs=True,)[0]
    else:
        return gradients(gradients(u,x), x, order = order - 1)


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1,32),
            torch .nn .Tanh(),
            torch.nn.Linear(32,32),
            torch.nn.Tanh(),
            torch.nn.Linear(32,32),
            torch.nn.Tanh(),
            torch.nn.Linear(32,1)
        )
        self.k = torch.nn.Parameter(torch.tensor(0.1))
        self.f = torch.nn.Parameter(torch.tensor(0.1))
    def forward(self,x):
        return self.net(x)

def l_interior(u):
    n=10000
    x = torch.rand(n, 1)
    x = x.to('cuda:0')
    cond = 0 * torch.exp(x)
    x = x.requires_grad_(True)
    uxy = u(x)
    return loss(gradients(uxy, x, 2)+ u.k *gradients(uxy, x, 1)+ u.f * uxy, cond)
def l_data(u):
    n=100
    x = torch.rand(n,1)
    x = x.to('cuda:0')
    cond = 9 * torch.exp(-2 * x) - 7 * torch.exp(-3 * x)
    noise_ratio = torch.rand(1).to('cuda:0') * 0.05 + 0.05  
    noise = torch.rand_like(cond).to('cuda:0') * 2 * noise_ratio - noise_ratio
    cond = cond + noise 
    x = x.requires_grad_(True)
    uxy = u(x)
    return loss(uxy, cond)
def l_left(u):
    n = 100
    y = torch.rand(n,1)
    x = torch.zeros_like(y)
    x = x.to('cuda:0')
    cond = 2 * torch.ones_like(x)
    x = x.requires_grad_(True)
    uxy = u(x)
    return loss(uxy,cond)
def l_x_left(u):
    n = 100
    y = torch.rand(n,1)
    x = torch.zeros_like(y)
    x = x.to('cuda:0')
    cond = 3 * torch.ones_like(x)
    x = x.requires_grad_(True)
    uxy = u(x)
    return loss(gradients(uxy, x, 1),cond)

u= MLP().cuda()
optimizer = torch.optim.Adam(u.parameters(), lr=0.001)

fig = plt.figure(figsize=(5, 5))
ax1 = fig.add_subplot(111)  

if torch.jit.is_tracing():  
    checkpoint_path = 'model_checkpoint_traced.pth'  # 如果使用TorchScript，可能需要不同的文件扩展名  
  
if torch.cuda.is_available():  
    device = torch.device('cuda')  
    u = u.to(device)  
else:  
    device = torch.device('cpu')  
  
start_epoch = 0  
try:  
    checkpoint = torch.load(checkpoint_path, map_location=device)  
    u.load_state_dict(checkpoint['model_state_dict'])  
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
    start_epoch = checkpoint['epoch']  
    print(f"Loaded checkpoint at epoch {start_epoch}")  
except FileNotFoundError:  
    print("Checkpoint not found. Starting from epoch 0.")  

for epoch in simple_progress(range(epochs)):
    optimizer.zero_grad()
    #l = l_interior(u)  + l_down(u) + l_left(u) + l_right(u)  + l_up(u)
    l = l_data(u)  + l_interior(u) + l_left(u) + l_x_left(u)
    #l = l_interior(u)
    l.backward()
    optimizer.step()
u = u.cpu()

torch.save({  
    'epoch': epoch + start_epoch,  # 保存最后一个epoch的索引  
    'model_state_dict': u.state_dict(),  
    'optimizer_state_dict': optimizer.state_dict(),  
}, checkpoint_path)  
print(f"Model checkpoint saved at {checkpoint_path}")


xx = torch.linspace(0, 1, 100).reshape(100, 1)
u_pred = u(xx)
u_real = 9 * torch.exp(-2 * xx) - 7 * torch.exp(-3 * xx)
u_error = torch.abs(u_pred-u_real)
mse = torch.mean(u_error.pow(2))  
rmse = torch.sqrt(mse)  
print('loss')
print(rmse.detach().numpy())
print('K:')
print(u.k.detach().numpy())
print('F:')
print(u.f.detach().numpy())
# 创建一个大图

n=100
x = torch.rand(n,1)
cond = 9 * torch.exp(-2 * x) - 7 * torch.exp(-3 * x)
noise_ratio = torch.rand(1) * 0.05 + 0.05  # 随机数乘以0.05再加0.05，得到0.05到0.1之间的值  
noise = torch.rand_like(cond) * 2 * noise_ratio - noise_ratio  # 缩放噪声到-noise_ratio到noise_ratio之间  
cond = cond + noise 

ax1.scatter(x, cond, label='Noisy Condition', color='purple', alpha=0.5)  
ax1.plot(xx.detach().numpy(), u_pred.detach().numpy(), label='PINN Solve', color='red')  
ax1.plot(xx.detach().numpy(), u_real.detach().numpy(), label='Real Solve', color='green')  
ax1.plot(xx.detach().numpy(), u_error.detach().numpy(), label='Absolute Error', color='blue')  

# 添加图例
ax1.legend()

# 设置标题和坐标轴标签
ax1.set_title('Combined Plots')
ax1.set_xlabel('X')
ax1.set_ylabel('Z')

# 显示图形
plt.show()
