import torch
from torch.func import vmap


from scr.constitutive_relations import Piola_Stress
from scr.Kinematics import Local_Deformation_Gradient, Global_Deformation_Gradient
from scr.Gauss_Quadratures import Gauss_Quadratures_2D


Gauss_Points, Gauss_Weights = Gauss_Quadratures_2D(2)

def Local_Weak_Form(u_fn, mu, lmbda, xi, eta):
    Local_F, Local_Grad_u  = Local_Deformation_Gradient(u_fn, xi, eta)
    Local_P = Piola_Stress(Local_F, mu, lmbda)
    return torch.tensordot(Local_P, Local_Grad_u, dims=([0, 1], [0, 1]))


def Integrated_Local_Weak_Form(u_fn, mu, lmbda, Gauss_Points, Gauss_Weights):
    Local_Weak = vmap(Local_Weak_Form, in_dims = (None, None, None, 0, 0))
    Local_Weak_Batched = Local_Weak(u_fn, mu, lmbda, Gauss_Points[:, 0], Gauss_Points[:, 1])
    Total_Local_Weak_Batched = torch.dot(Local_Weak_Batched, Gauss_Weights)

    return Total_Local_Weak_Batched


def Global_Weak_Form(u_fn, X_fn, mu, lmbda, xi, eta):

    Global_F, det_Grad_X, Global_Grad_u =  Global_Deformation_Gradient(u_fn, X_fn, xi, eta)
    Global_P = Piola_Stress(Global_F, mu, lmbda) 
    Global_Weak = torch.tensordot(Global_P, Global_Grad_u, dims = ([0,1], [0,1])) * det_Grad_X

    return Global_Weak


def Integrated_Global_Weak_Form(u_fn, X_fn, mu, lmbda, Gauss_Points, Gauss_Weights):
    Global_Weak = vmap(Global_Weak_Form, in_dims= (None, None, None, None, 0, 0))
    Global_Weak_Batched = Global_Weak(u_fn, X_fn, mu, lmbda, Gauss_Points[:, 0], Gauss_Points[:, 1])
    Total_Global_Weak_Batched = torch.dot(Global_Weak_Batched, Gauss_Weights)

    return Total_Global_Weak_Batched
