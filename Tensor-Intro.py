import torch
x = torch.arange(12, dtype=torch.float32)
#print(x)
elements = x.numel()
#print(elements)
y = x.reshape(3,4)
#print(y)
z =torch.zeros((2, 3, 4))
#print(z)
a = torch.ones((2,3,4))
#print(a)
b = torch.rand(3,4)
#print(b)
c = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
#print(c)
#print(c[-1])
#print(c[1:3])
c[1,2]= 17
#print(c)
c[:2,:] = 12
##print(c)
d = torch.exp(c)
#print(d)
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
#print(x+y)
#print(x-y)
#print(x*y)
#print(x/y)
#print(x**y)
X = torch.arange(12, dtype=torch.float32 ).reshape(3,4)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
z = torch.cat((X, Y), dim=0)
a = torch.cat((X, Y), dim=1)
#print(z)
#print(a)
#print(X == Y)
#print(X.sum())
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
#print(a+b)
A = X.numpy()
B = torch.from_numpy(A)
#print(type(A))
#print(type(B))
a = torch.tensor([3.5])
a.item()
#print(a)
#print(float(a))
#print(int(a))

##excersise 
X = torch.arange(12, dtype=torch.float32 ).reshape(3,4)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(X==Y)
print(X>Y)
print(X<Y)

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a = torch.arange(8).reshape((2,2,2))
print(a)
print(b)
print(a+b)
