import torch
import torch.nn as nn
#question 7
x = torch.tensor([2.0, 3.0], requires_grad = True)
y = x ** 3
y.backward(torch.tensor([2.0, 1.0]))
#print(y)
#print(x.grad)

#Question 8
model = nn.Linear(2, 3)
x = torch.tensor([3.0, 4.0])
with torch.no_grad():
    op = model(x)
    #print(op)
with torch.inference_mode():
    op = model(x)
    #print(op)

#Question 9
# Step 1: Create a custom autograd Function
class MySquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x ** 2
    @staticmethod
    def backward(ctx, grad_op):
        x, = ctx.saved_tensors
        grad_op = x * 2
        return grad_op
        
# Step 2: Use the custom function
x = torch.tensor([3.0], requires_grad=True)
y = MySquare.apply(x)
# Step 3: Compute gradients
y.backward()

#print("y:", y)              
#print("x.grad:", x.grad)    

#Question 10
#Doing outplace
x = torch.tensor([1.0, 2.0, 3.0])
y = x + 8
#print(x)
#print(y)
x1 = torch.tensor([2.0, 1.0, 3.0])
x1.add_(4)            #For Inplace modifications we got many default functions like add_, mul_ etc......
#print(x1)

