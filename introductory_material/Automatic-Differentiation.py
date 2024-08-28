import torch 
import matplotlib.pyplot as plt
import numpy as np
from d2l import torch as d2l
def fx(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

def f(x):
    return ((torch.log10(x ** 2)) * torch.sin(x)) + (x ** -1)
x = torch.arange(4.0) 
print(x.grad)
x.requires_grad_(True)
print(x.grad)
y = 2 * torch.dot(x, x)
y.backward()
print(x.grad)
print(x.grad == 4 * x)
x.grad.zero_()
y = x.sum()
y.backward()
x.grad.zero_()
print(x.grad)
y = x * x
y.backward(gradient=torch.ones(len(y)))  # Faster: y.sum().backward()
print(x.grad)
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad == u)
a = torch.randn(size=(), requires_grad=True)
d = fx(a)
d.backward()
print(a.grad == d / a)
# print(a.grad)
# a.backward()
# print(a.grad)
# a.backward()
# print(a.grad)
# a.backward()
# print(a.grad)


#problem 3
x = torch.linspace(-2 * np.pi, 2 * np.pi, 100, requires_grad=True)
# y = torch.sin(x)
y = fx(x)
y.backward(torch.ones_like(x))
dy_dx = x.grad
x_np = x.detach().numpy()
y_np = y.detach().numpy()
dy_dx_np = dy_dx.detach().numpy()
plt.figure(figsize=(10, 6))
plt.plot(x_np, y_np, label="f(x) = sin(x)", color='blue')
plt.plot(x_np, dy_dx_np, label="f'(x) (computed)", color='red', linestyle='--')
plt.title("Function f(x) = sin(x) and its derivative f'(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

x = torch.linspace(0.1, 10, 100)

x.requires_grad_(True)

y = f(x)
y.backward(torch.ones_like(y))
p = x.grad

d2l.plot(x.detach(), [y.detach(), p], 'x', 'f(x)', legend=['f(x)', 'df/dx'])
z=4
desired_derivative = (2 / (x * torch.log(torch.tensor(10)))) * torch.sin(x) + torch.log10(x**2) * torch.cos(x) - (1/x **2)
print(f"{(p[z]-desired_derivative[z]).detach().numpy():.18f}")
p == desired_derivative, p[z], desired_derivative[z], p[z] == desired_derivative[z]