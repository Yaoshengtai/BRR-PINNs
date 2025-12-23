import torch
from abc import ABC, abstractmethod
from torch.autograd import grad
import math
import numpy as np
from PINN.parameters import *

# The ADF for a line segment
    # start point: (x1,y1)
    # end point: (x2,y2)
    # L: line segment length
    # xx,yy: collocation points, torch.tensor
def lineseg(xx,yy,x1,x2,y1,y2,L):
    f=((xx-x1)*(y2-y1)-(yy-y1)*(x2-x1))/L
    t=(L**2/4-((xx-(x1+x2)/2)**2+(yy-(y1+y2)/2)**2))/L
    phi=torch.sqrt(t**2+f**4)
    adf=torch.sqrt(f**2+((phi-t)/2)**2)
    return adf

class Approximator(ABC):
    r"""The base class of approximators. An approximator is an approximation of the differential equation's solution.
    It knows the parameters in the neural network, and how to calculate the loss function and the metrics.
    """
    @abstractmethod
    def __call__(self):
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def parameters(self):
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def calculate_loss(self):
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def calculate_metrics(self):
        raise NotImplementedError  # pragma: no cover

class SingleNetworkApproximator2DSpatial_heat(Approximator):
    #An approximator to approximate the solution of a 2D steady-state heat transfer problem

    # param single_network: A neural network
    # type single_network: `torch.nn.Module`

    # param pde: The PDEs to solve
    # type pde: list[function]

    # param boundary_conditions: A list of boundary conditions
    # type boundary_conditions: list[BoundaryCondition]

    # param args: the global arguments
    # type args: argparse.ArgumentParser.parse_args

    def __init__(self, single_network, pde, boundary_conditions, args):
        self.single_network = single_network
        self.pde = pde
        self.boundary_conditions = boundary_conditions
        self.args=args

    def __call__(self, xx, yy):
        # the forward computing
        x = torch.unsqueeze(xx, dim=1).requires_grad_()
        y = torch.unsqueeze(yy, dim=1).requires_grad_()
        xy = torch.cat((x, y), dim=1)
        uu = self.single_network(xy)

        # exactly impose diriclet boundary
        if self.args.impose!=0:
            u_up=2*xx**3-3*xx**2+1
            u_0=0

            # a rectangle, simply use the actual distance
            # for complex geometry, use function 'lineseg' to generate ADF

            uu[:,0]=u_up+(1-yy)*uu[:,0].clone()
            uu[:,1]=u_0+xx*uu[:,1].clone()
            uu[:,2]=u_0+yy*uu[:,2].clone()
            uu[:,3]=u_0+(1-xx)*uu[:,3].clone()
            
            return uu
    
    def parameters(self):
        return self.single_network.parameters()

    def calculate_loss(self, xx, yy):

        uu = self.__call__(xx, yy)

        #implement Boundary region reinforcement
        R_func=(1-yy)*(xx)*(1-xx)*yy/((1-yy)*(xx)*(1-xx)+(1-yy)*(xx)*yy+(1-yy)*(1-xx)*yy+(xx)*(1-xx)*yy+1e-20)
        R_center=1/8
        bound_optim=self.args.brr*torch.exp(abs(R_func)*(math.log(self.args.center_value)-math.log(self.args.brr))/R_center)
        
        # bound_optim=bound_optim.detach().cpu().numpy()
        # np.savetxt('bound_optim.txt',bound_optim)

        equation_mse = self.args.weight_equ1*torch.mean(abs(self.pde[0](uu[:,0], xx, yy))**2)+\
                       self.args.weight_equ2*torch.mean((abs(self.pde[1](uu[:,0], xx, yy)-uu[:,1])*bound_optim)**2)+\
                       self.args.weight_equ3*torch.mean((abs(self.pde[2](uu[:,0], xx, yy)-uu[:,2])*bound_optim)**2)+\
                       self.args.weight_equ4*torch.mean((abs(self.pde[3](uu[:,0], xx, yy)-uu[:,3])*bound_optim)**2)
                       
        # boundary_mse =  sum(self._boundary_mse(bc) for bc in self.boundary_conditions)
        
        return equation_mse

    def _boundary_mse(self, bc):
        xx, yy = next(bc.points_generator)
        uu= self.__call__(xx.requires_grad_(), yy.requires_grad_())
        loss=torch.mean(abs(bc.form(uu[:,0], xx, yy))**2)
        w=bc.weight
        loss=loss*w
        return loss


    def calculate_metrics(self, xx, yy, metrics):
        uu = self.__call__(xx, yy)
        return {
            metric_name: metric_func(uu,xx,yy)
            for metric_name, metric_func in metrics.items()
        }
    
class SingleNetworkApproximator2DSpatial_deform(Approximator):
    # An approximator to approximate the solution of a 2D steady-state deformation problem

    # param single_network: A neural network
    # type single_network: `torch.nn.Module`

    # param pde: The PDEs to solve
    # type pde: list[function]

    # param boundary_conditions: A list of boundary conditions
    # type boundary_conditions: list[BoundaryCondition]

    # param args: the global arguments
    # type args: argparse.ArgumentParser.parse_args

    def __init__(self, single_network, pde, boundary_conditions, args):
        self.single_network = single_network
        self.pde = pde
        self.boundary_conditions = boundary_conditions
        self.args=args

    def __call__(self, xx, yy):
        # the forward computing
        x = torch.unsqueeze(xx, dim=1).requires_grad_()
        y = torch.unsqueeze(yy, dim=1).requires_grad_()
        xy = torch.cat((x, y), dim=1)
        uu = self.single_network(xy)

        # exactly impose diriclet boundary
        if self.args.impose!=0:

            # a rectangle, simply use the actual distance
            # for complex geometry, use function 'lineseg' to generate ADF

            u_par_0=0
            # Excatly impose bottom boundary, z-direction displacement constraint
            uu[:,1]=u_par_0+yy*uu[:,1].clone()/h1

            # Excatly impose tau_zr on four boundaries
            # uu[:,5]=u_par_0+(h1-yy)*(xx-r1)*(r2-xx)*yy*uu[:,5].clone()\
            # /((h1-yy)*(xx-r1)*(r2-xx)+(h1-yy)*(xx-r1)*yy+(h1-yy)*(r2-xx)*yy+(xx-r1)*(r2-xx)*yy+1e-20)
            uu[:,5]=u_par_0+(h1-yy)*(xx-r1)*(r2-xx)*yy*uu[:,5].clone()/(h1*(r2-r1)**2)
            
            u_par_up=((-10+1)*(2*((xx-r1)/(r2-r1))**3-3*((xx-r1)/(r2-r1))**2+1)-1)
            # Excatly impose sigma_zz on up boundary
            uu[:,4]=u_par_up+(h1-yy)*uu[:,4].clone()/h1
            # Excatly impose sigma_rr on left and right boundaries
            uu[:,2]=u_par_up+(xx-r1)*(r2-xx)*uu[:,2].clone()/(r2-r1)**2

            return uu
    
    def parameters(self):
        return self.single_network.parameters()

    def calculate_loss(self, xx, yy):

        uu = self.__call__(xx, yy)

        #implement Boundary region reinforcement
        R_func=(1-yy)*(xx)*(1-xx)*yy/((1-yy)*(xx)*(1-xx)+(1-yy)*(xx)*yy+(1-yy)*(1-xx)*yy+(xx)*(1-xx)*yy+1e-20)
        R_center=1/(4/(r2-r1)+4/h1)  
        bound_optim=self.args.brr*torch.exp(abs(R_func)*(math.log(self.args.center_value)-math.log(self.args.brr))/R_center)
        
        # bound_optim=bound_optim.detach().cpu().numpy()
        # np.savetxt('bound_optim.txt',bound_optim)

        equation_mse = self.args.weight_equ1*torch.mean(abs(self.pde[0](uu[:,0],uu[:,1], xx, yy))**2)+\
                self.args.weight_equ2*torch.mean(abs(self.pde[1](uu[:,0],uu[:,1], xx, yy))**2)+\
                self.args.weight_equ3*torch.mean((abs(self.pde[2](uu[:,0],uu[:,1], xx, yy)-uu[:,2])*bound_optim)**2)+\
                self.args.weight_equ4*torch.mean((abs(self.pde[3](uu[:,0],uu[:,1], xx, yy)-uu[:,3])*bound_optim)**2)+\
                self.args.weight_equ5*torch.mean((abs(self.pde[4](uu[:,0],uu[:,1], xx, yy)-uu[:,4])*bound_optim)**2)+\
                self.args.weight_equ6*torch.mean((abs(self.pde[5](uu[:,0],uu[:,1], xx, yy)-uu[:,5])*bound_optim)**2)
                       
        return equation_mse


    def calculate_metrics(self, xx, yy, metrics):
        uu = self.__call__(xx, yy)
        return {
            metric_name: metric_func(uu,xx,yy)
            for metric_name, metric_func in metrics.items()
        }


def _train_2dspatial(train_generator_spatial, approximator, optimizer, metrics, shuffle, batch_size,device,args):
    xx, yy = next(train_generator_spatial)
    xx,yy=xx.to(device),yy.to(device)

    xx.requires_grad = True
    yy.requires_grad = True
    training_set_size = len(xx)
    idx = torch.randperm(training_set_size) if shuffle else torch.arange(training_set_size)
    batch_start, batch_end = 0, batch_size
    while batch_start < training_set_size:
        if batch_end > training_set_size:
            batch_end = training_set_size
        batch_idx = idx[batch_start:batch_end]
        batch_xx = xx[batch_idx].to(device)
        batch_yy = yy[batch_idx].to(device)
        batch_loss = approximator.calculate_loss(batch_xx, batch_yy)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        batch_start += batch_size
        batch_end += batch_size

    epoch_loss = approximator.calculate_loss(xx,yy)
    epoch_metrics = approximator.calculate_metrics(xx, yy, metrics)
    for k, v in epoch_metrics.items():
        epoch_metrics[k] = v.item()

    return epoch_loss, epoch_metrics


def _solve_2dspatial(
    train_generator_spatial,
    approximator, optimizer, batch_size, max_epochs, shuffle, metrics, monitor,device,args,
):
    r"""Solve a 2D steady-state problem

    :param train_generator_spatial:
        A generator to generate 2D spatial points for training.
    :type train_generator_spatial: generator
    :param approximator:
        An approximator for 2D time-state problem.
    :type approximator: `temporal.SingleNetworkApproximator2DSpatial`, `temporal.SingleNetworkApproximator2DSpatialSystem`, or a custom `temporal.Approximator`
    :param optimizer:
        The optimization method to use for training.
    :type optimizer: `torch.optim.Optimizer`
    :param batch_size:
        The size of the mini-batch to use.
    :type batch_size: int
    :param max_epochs:
        The maximum number of epochs to train.
    :type max_epochs: int
    :param shuffle:
        Whether to shuffle the training examples every epoch.
    :type shuffle: bool
    :param metrics:
        Metrics to keep track of during training.
        The metrics should be passed as a dictionary where the keys are the names of the metrics,
        and the values are the corresponding function.
        The input functions should be the same as `pde` (of the approximator) and the output should be a numeric value.
        The metrics are evaluated on both the training set and validation set.
    :type metrics: dict[string, callable]
    :param monitor:
        The monitor to check the status of nerual network during training.
    :type monitor: `temporal.Monitor2DSpatial` or `temporal.MonitorMinimal`
    """
    return _solve_spatial_temporal(
        train_generator_spatial, approximator, optimizer, batch_size, max_epochs, shuffle, metrics, monitor,device,args,
        train_routine=_train_2dspatial
    )

def _solve_spatial_temporal(
    train_generator_spatial, approximator, optimizer, batch_size, max_epochs, shuffle, metrics, monitor,device,args,
    train_routine
):
    history = {'train_loss': []}
    for metric_name, _ in metrics.items():
        history['train_' + metric_name] = []
    
    with open(args.save_dict+'-train_log.txt', 'w') as file:
        file.write("....... begin training ....... \n")
    for epoch in range(max_epochs):
        train_epoch_loss, train_epoch_metrics = train_routine(
            train_generator_spatial, approximator, optimizer, metrics, shuffle, batch_size,device,args,
        )
        history['train_loss'].append(train_epoch_loss.detach().cpu())
        
        
        for metric_name, metric_value in train_epoch_metrics.items():
            history['train_' + metric_name].append(metric_value)

        if monitor and epoch % monitor.check_every == 0:
            monitor.check(approximator, history,epoch)

        with open(args.save_dict+'-train_log.txt', 'w') as file:
            last_items = {key: values[-1] if values else None for key, values in history.items()}
            for key, value in last_items.items():
                file.write(f"{key}: {value}\n")
            file.write("Already calculate for "+ str(epoch) + "/"+str(max_epochs)+'\n')


    return approximator, history
