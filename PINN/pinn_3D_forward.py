# solve the pde of u_xx + u_yy = (x^2+y^2)exp(-x*y) in the domain [0,1]x[0,1] with u = exp(-x*y) on the boundary

import torch
import matplotlib.pyplot as plt
from poprogress import simple_progress

num_hidden_units = 100
num_points_total = 1000
num_points_inner = 100
num_points_side = 10

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)

LOSS = torch.nn.MSELoss()

def autogradient(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), 
                                   create_graph=True,
                                   only_inputs=True,)[0]
    else:
        return autogradient(autogradient(u, x), x, order = order - 1)


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2,32),
            torch .nn .Tanh(),
            torch.nn.Linear(32,32),
            torch.nn.Tanh(),
            torch.nn.Linear(32,32),
            torch.nn.Tanh(),
            torch.nn.Linear(32,1)
        )
    def forward(self,x):
        return self.net(x)

def loss_interior(u):
    n = num_points_total
    x = torch.rand(n, 1)
    y = torch.rand(n,1)
    x = x.to('cuda:0')
    y = y.to('cuda:0')
    cond = (y ** 2 +  x ** 2) * torch.exp(-x * y)
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    uxy = u(torch.cat([x,y],dim=1))
    return LOSS(autogradient(uxy, x, 2) + autogradient(uxy, y, 2), cond)

def loss_down_yy(u):
    n = num_points_side
    x = torch.rand(n,1)
    y = torch.zeros_like(x)
    x = x.to('cuda:0')
    y = y.to('cuda:0')
    cond = x ** 2
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    uxy = u(torch.cat([x, y], dim=1))
    return LOSS(autogradient(uxy,y,2), cond)

def loss_up_yy(u):
    n = num_points_side
    x = torch.rand(n,1)
    y = torch.ones_like(x)
    x = x.to('cuda:0')
    y = y.to('cuda:0')
    cond = x ** 2 / torch.e
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    uxy = u(torch.cat([x,y], dim=1))
    return LOSS(autogradient(uxy, y,2), cond)

def loss_down(u):
    n = num_points_side
    x = torch.rand(n,1)
    y = torch.zeros_like(x)
    x = x.to('cuda:0')
    y = y.to('cuda:0')
    cond = torch.ones_like(x)
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    uxy = u(torch.cat([x, y], dim=1))
    return LOSS(uxy,cond)

def loss_up(u):
    n = num_points_side
    x = torch.rand(n, 1)
    y = torch.ones_like(x)
    x = x.to('cuda:0')
    y = y.to('cuda:0')
    cond = torch.exp(-x)
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    uxy = u(torch.cat([x, y], dim=1))
    return LOSS(uxy,cond)

def loss_left(u):
    n = num_points_side
    y = torch.rand(n,1)
    x = torch.zeros_like(y)
    x = x.to('cuda:0')
    y = y.to('cuda:0')
    cond = torch.ones_like(x)
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    uxy = u(torch.cat([x, y], dim=1))
    return LOSS(uxy,cond)

def loss_right(u):
    n = num_points_side
    y = torch.rand(n,1)
    x = torch.ones_like(y)
    x = x.to('cuda:0')
    y = y.to('cuda:0')
    cond = torch.exp(-y)
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    uxy = u(torch.cat([x, y], dim=1))
    return LOSS(uxy, cond)

def loss_data(u):
    n = num_points_side
    x = torch.rand(n,1)
    y = torch.rand(n, 1)
    x = x.to('cuda:0')
    y = y.to('cuda:0')
    cond = torch.exp(-x*y)
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    uxy = u(torch.cat([x, y], dim=1))
    return LOSS(uxy, cond)

u = MLP().cuda()
optimizer = torch.optim.Adam(u.parameters(), lr=0.001)
epochs = 5000
for epoch in simple_progress(range(epochs)):
    optimizer.zero_grad()
    l = loss_interior(u)  + loss_down(u) + loss_left(u) + loss_right(u)  + loss_up(u)
    #l = loss_data(u) + loss_interior(u)
    #l = loss_interior(u)
    l.backward()
    optimizer.step()

u = u.cpu()

num_cells = 100
x_coords = torch.linspace(0, 1, num_cells)
xm, ym = torch.meshgrid(x_coords, x_coords)
xx = xm.reshape(-1,1)
yy = ym.reshape(-1,1)
xy = torch.cat([xx, yy], dim=1)
u_pred = u(xy)
u_real = torch.exp(-yy*xx)
u_error = torch.abs(u_pred - u_real)
mse = torch.mean(u_error.pow(2))  
rmse = torch.sqrt(mse)  
u_pred_fig = u_pred.reshape(num_cells, num_cells)
u_real_fig = u_real.reshape(num_cells, num_cells)
u_error_fig = u_error.reshape(num_cells, num_cells)
print(rmse.detach().numpy())

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(xm.detach().numpy(), ym.detach().numpy(), u_pred_fig.detach().numpy(), alpha=0.7, label='PINN Solve', color='red')
ax.plot_surface(xm.detach().numpy(), ym.detach().numpy(), u_real_fig.detach().numpy(), alpha=0.7, label='Real Solve', color='green')
ax.plot_surface(xm.detach().numpy(), ym.detach().numpy(), u_error_fig.detach().numpy(), alpha=0.7, label='Absolute Error', color='blue')

ax.legend()

ax.set_title('Combined Plots')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
