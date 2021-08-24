
import torch
import pandas as pd
import matplotlib.pyplot as plt



train = pd.read_csv('data/day_length_weight.csv')
train_y = train.pop('weight')
train_x = torch.tensor(train.to_numpy(), dtype=torch.double).t()
train_y = torch.tensor(train_y.to_numpy(), dtype=torch.double).t()

print(train_x.shape)


class LinearRegressionModel:
    def __init__(self):
        self.W = torch.tensor([[0.0, 0.0]], requires_grad=True, dtype=torch.double)
        self.b = torch.tensor([[0.0]], requires_grad=True, dtype=torch.double)
    # Predictor
    def f(self, x):
        return self.W.mm(x) + self.b

    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x),y)




model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.000001)
for epoch in range(5000000):
    model.loss(train_x, train_y).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    # similar to:
    # model.W -= model.W.grad * 0.01
    # model.b -= model.b.grad * 0.01

    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(train_x, train_y)))

plt.plot(train_x, train_y, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x = torch.tensor([[torch.min(train_x)], [torch.max(train_x)]])  # x = [[1], [6]]]
plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = xW+b$')
plt.legend()
plt.show()
