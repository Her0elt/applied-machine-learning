
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn




train = pd.read_csv('data/day_length_weight.csv', dtype='float')
train_y = train.pop('day')
train_x = torch.tensor(train.to_numpy(), dtype=torch.float)
train_y = torch.tensor(train_y.to_numpy(), dtype=torch.float).reshape(-1,1)


model= nn.Linear(train_x.shape[1], train_y.shape[1])

opt = torch.optim.SGD(model.parameters(), lr=1e-4)

loss_fn = F.mse_loss

def fit(num_epochs, model, loss_fn, opt):
    for _ in range(num_epochs):
            pred = model(train_x)
            loss = loss_fn(pred,train_y)
            loss.backward()
            opt.step()
            opt.zero_grad()
    W, b =  model.parameters()
    print("W = %s, b = %s, loss = %s" % (W.data, b.data, loss_fn(model(train_x), train_y)))
       

fit(100000, model, loss_fn, opt)

xt =train_x.t()[0]
yt =train_x.t()[1]

fig = plt.figure('Linear regression 3d')
ax = plt.axes(projection='3d', title="Model for predicting days lived by weight and length")
ax.set_xlabel('$x1$')
ax.set_ylabel('$x2$')
ax.set_zlabel('$y$')
ax.set_xticks([]) 
ax.set_yticks([])
ax.set_zticks([])
ax.w_xaxis.line.set_lw(0)
ax.w_yaxis.line.set_lw(0)
ax.w_zaxis.line.set_lw(0)

ax.quiver([0], [0], [0], 
        [torch.max(xt + 1)], [0], [0], 
        arrow_length_ratio=0.05, color='black')

ax.quiver([0], [0], [0], 
        [0], [torch.max(yt + 1)],[0], 
        arrow_length_ratio=0.05, color='black')

ax.quiver([0], [0], [0], [0], [0], 
        [torch.max(train_y + 1)],
        arrow_length_ratio=0, color='black')
# Plot
ax.scatter(xt.numpy(), yt.numpy(), train_y.numpy(), label='$(x^{(i)},y^{(i)}, z^{(i)})$')
x = torch.tensor([[torch.min(xt)], [torch.max(xt)]]) 
y = torch.tensor([[torch.min(yt)], [torch.max(yt)]]) 
xy = torch.cat((x,y),1)
print(xy)
ax.plot(x.flatten(), y.flatten(), model(xy).detach().flatten(),  label='$\\hat y = f(x) = xW+b$', color="orange")
ax.legend()
plt.show()

