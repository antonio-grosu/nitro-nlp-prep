import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



scalar = torch.tensor(7)
# print(scalar.ndim)



vector = torch.tensor([7,7])
# print(vector.ndim, vector.shape)


MATRIX = torch.tensor([[7,8 ]])
# print(MATRIX)



TENSOR = torch.tensor([[[1,2,3], [3,6,89], [4,5,6]]])
# print(TENSOR[0][0][0])

random_tensor = torch.rand(4,100,100,100)
# print(*random_tensor, random_tensor.ndim)




#tensor ca o imagine

random_image_tensor = torch.rand(size=(224,224,3)) #height px , width  px, color channel R G B =3
# print(random_image_tensor.shape)


# creez tensor de 1 si 0


zeros = torch.zeros((10))
# print(zeros)

ones = torch.ones(3,4)
# print(ones)

#range of tensors and tensors-like


# torch.range

# print(torch.arange(0,10))
#print(torch.arange(start =0, end= 10000, step = 1234))


#create tensor like faar sa definesc shape ul


ten_zero = torch.zeros_like(input = ones)
# print(ten_zero)


#tensor data types


float_32_tensor = torch.tensor([3.0,6.0,9.0], dtype=None # tipul de date
                               , device = "cpu" # device cpu cuda etc
                               , requires_grad= False # daca sa track uiasca gradiente sau nu
                               )
# print(float_32_tensor)


# cele mai mari pb cu torch si deep learning
# 1 tip de date prost pt TENSOR
# tensor cu shape gresit
# tensor pe device gresit



float_16_tensor = torch.tensor([3,6,9] , dtype=torch.float16)



# print(float_16_tensor ** float_32_tensor)


int_64_tensor = torch.tensor([3,6,9] , dtype=torch.int64)



# print(int_64_tensor * float_16_tensor)



# informatii din tensori
# tensor datatype ->  tensor.dtype
# shape -> tensor.shape
# device -> tensor.device


some_tensor = torch.rand(3,4)

# print(some_tensor, some_tensor.dtype, some_tensor.shape, some_tensor.device)




#manipulating tensors

# operatii - adunare, scadere, inmultire, impartire, inmultire matrice



tensor = torch.tensor([1,2,3])


#inmultire matrice

# elemen-wise
# matrix mul  ( dot product)

# element wise
# print(tensor * tensor)


# print(torch.matmul(tensor, tensor)) tensor(14)


tensor_2 = torch.tensor([[1,2],[4,5]]) 
# print(torch.matmul(tensor_2, tensor_2)) returneaza matrice 





# cele mai comune erori in DL sunt erorile de shape


# 2 reguli ca sa inmultesti matrice
#   1. inner dimensiunile trebuie sa fie de aceeasi marime
        # (3,2) @ *(3,2) nu va merge
        # (2,3) @ (3,2) va merge
        # (3,2) @ (2,3)   va merge
    # 2. matricea rezultat are shape ul  dimensiunilor exterioare



test_mat_1 = torch.rand(2,3)
test_mat_2 = torch.rand(3,2)
# print(test_mat_1 @ test_mat_2)
# print(tensor_2 @ tensor_2)

# erorile de shape


tensor_1 = torch.rand(3,2)
tensor_2 = torch.rand(3,2)

# print(torch.mm(tensor_1, tensor_2))  eroare



# print(tensor_2, tensor_2.T)

# print(torch.mm(tensor_1, tensor_2.T))  merge ca am inmultit cu transpusa



# print(f"Shape urile celor 2 tensori {tensor_1.shape} \n {tensor_2.shape} \n")
# print(f"Astia 2 nu se pot inmulti pt ca nu au dimensiunile interioare potrivite \n asa ca pot inmulti cu transpusa")
# print(f" tensor_1 @ tensor_2.T \n{tensor_1 @ tensor_2.T}")




#tensor aggregation min max sum etc
x = torch.arange(0,100,10, dtype=torch.float32) # daca nu il fac float atunci voi primi eroare de dtype la mean media arit
# print(torch.min(x), torch.max(x) , torch.mean(x))


# suma
# print(torch.sum(x) , x.sum())


# positiomal min si max

print(x,x.argmin()) # index ul unde  apare minimul

#argmax ac lucru
#utili pt functia soft max

#reshaping , stacking , squeezing, and unsqueezing with tensors

# view -> returns a view of an input tensor of certain shape but keep the same memory as the original tensor
# stack ->  combine multiple tensors vertical or horizontal

# squeeze -> removes all '1' dimensions from tensor
# unsqueeze ->add '1' dimensions to a target tensor
# permute -> return a view of input  with dimensions permuted

x = torch.arange(1., 10.)

#x_reshaped = x.reshape(1,9) #RuntimeError: shape '[1, 7]' is invalid for input of size 9
x_reshaped = x.reshape(9,1)
#tensor([ 0., 10., 20., 30., 40., 50., 60., 70., 80., 90.]) tensor(0) #
# tensor([[1.],
#         [2.],
#         [3.],
#         [4.],
#         [5.],
#         [6.],
#         [7.],
#         [8.],
#         [9.]])

z = x.view(9,1)
z[:,0]= 5
#print(z, x)  afecteaza si x ul pt ca view ul share uieste ac memeorie ca si tensor ul original


x_stacked = torch.stack([x,x,x,x,x] , dim=1)

# tensor([ 0., 10., 20., 30., 40., 50., 60., 70., 80., 90.]) tensor(0)
# print(x_stacked) dim = 0
# tensor([[5., 5., 5., 5., 5., 5., 5., 5., 5.],
#         [5., 5., 5., 5., 5., 5., 5., 5., 5.],
#         [5., 5., 5., 5., 5., 5., 5., 5., 5.],
#         [5., 5., 5., 5., 5., 5., 5., 5., 5.],
#         [5., 5., 5., 5., 5., 5., 5., 5., 5.]])



# torch squeeze rm all 1 dim  target tensor


x = torch.zeros(2,1,2,1,2)
# print(x.size())

y = torch.squeeze(x)
# print(y.size())


y = torch.squeeze(x,1)
# print(y.size())

# torch.Size([2, 1, 2, 1, 2])
# torch.Size([2, 2, 2])
# torch.Size([2, 2, 1, 2])


# print(f"Previous tensor {x.size()} \n Squeezed tensor {y.size()}")
# Previous tensor torch.Size([2, 1, 2, 1, 2]) 
#  Squeezed tensor torch.Size([2, 2, 1, 2])
# torch.Size([1, 2, 2, 1, 2])

# torch unsqueeze - adauga un single dim catre un target tensor la un specific dim


# adauga o dimensiune extra

z = y.unsqueeze(dim=0)
# print(z.size())



# torch permute  returneaza un view  al tensorulu cu dimensiunile permutate


#permute se foloseste la imagini

x_original = torch.rand(size=(224,224,3)) # o imagine sa zicem

# permute the original tensor to rearragen the axis or dim order

# punem color channelul pe prima dimensiune

x_permuted = x_original.permute(2,0,1) # axa 0-1, 1->2, 0->1

# print(f"Previous shape {x_original.shape} \n New shape {x_permuted.shape} ")


#indexing ( selectare date din tensori)


# indexinul din torch e similar cu cel din numpy

x = torch.arange(1,10).reshape(1,3,3)
# print(x,x.shape)

# print(x[0])
# tensor([[1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]])


# print(x[0][0])
# tensor([1, 2, 3])



# print(x[0][0][0])
# tensor(1)

#: pentru a selecta totul dintr o dimensiune


# print(x[:,:,1])
# tensor([[2, 5, 8]])



# print(x[:,1,1])
# tensor(5)


#pytorch tensors si numpy

#data in numpy, want in Pytorch tensor


array =np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
# print(array, tensor)[1. 2. 3. 4. 5. 6. 7.] tensor([1., 2., 3., 4., 5., 6., 7.], dtype=torch.float64)
# din   numpy tensorul se face pe float 64 , in loc de 32 cum e default pe tensor 


# schimb valoarea array ului

array = array+1
# print(array, tensor) -> tensorul nu se schimba
# [2. 3. 4. 5. 6. 7. 8.] tensor([1., 2., 3., 4., 5., 6., 7.], dtype=torch.float64)

tensor = torch.ones(7)
numpy_tensor = tensor.numpy()

tensor+=1
# print(tensor, numpy_tensor)
#la fel, tensorul din numpy nu se schimba, se aloca o noua zona de memorie de fiecare data cand se intampla acest 'cast' din torch in numpy sau invers


#reproducibility take the random out of the random

# how a neural network works
#1  start with random numbers
#2 tensor operations
#3 update random numbers to try and make them better 
# again and again 1 2 3


# To reduce randomness in neural network, comes the concept of random seed
# essentially what the random seed does is 'flavour' the randomness




my_tensor = torch.tensor([1,2,3],device="cuda")

print(my_tensor, my_tensor.device)
