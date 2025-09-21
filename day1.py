#Question 1
import torch
a = torch.rand(3, 3)
b = a *2
#print(b)

#Question 2
l = [[1,2,3], [4, 5, 6], [7, 8, 9]]
a = torch.tensor(l)
a = a ** 2
#print(a)

#Question 3
a = torch.ones(2, 4, 4)
b = a[:,1:3, 1:3]
#print(b)

#Question 4
a = torch.rand(1,3)
b = torch.ones(4, 3)
c = a+b
#print(a)
#print(b)
#print(c)

#Question 5
a = torch.ones(1, 4)
b = torch.zeros(4, 3)
c = torch.matmul(a, b)
#print(c)

#Question 6
a = torch.ones(3, 4, dtype=torch.float32)
#print(a.dtype)
a = a.to(torch.int64)
#print(a.dtype)