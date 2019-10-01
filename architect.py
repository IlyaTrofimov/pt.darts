""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import torch
from scipy.sparse.linalg import cg, LinearOperator
import numpy as np

class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, w_momentum, w_weight_decay, hessian_vector_type = 1):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay
        self.hessian_vector_type = hessian_vector_type

    def virtual_step(self, trn_X, trn_y, xi, w_optim):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        loss = self.net.loss(trn_X, trn_y) # L_trn(w)

        # compute gradient
        gradients = torch.autograd.grad(loss, self.net.weights())

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay*w))

            # synchronize alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)

    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # compute gradient

        # def ddot(a, b):
        #    return sum(u.flatten() @ v.flatten() for u, v in zip(a, b))

        if self.hessian_vector_type in [0, 1, 2]:
            # do virtual step (calc w`)
            self.virtual_step(trn_X, trn_y, xi, w_optim)

            # calc unrolled loss
            val_loss = self.v_net.loss(val_X, val_y) # L_val(w`)

            v_alphas = tuple(self.v_net.alphas())
            v_weights = tuple(self.v_net.weights())
            v_grads = torch.autograd.grad(val_loss, v_alphas + v_weights)
            dalpha = v_grads[:len(v_alphas)] # d_a L_val(w')
            dw = v_grads[len(v_alphas):]     # d_w L_val(w')

        #
        # DARTS
        #
        if self.hessian_vector_type == 1:
            hessian_vector = self.compute_hessian_vector(dw, trn_X, trn_y)

        #
        # Exact hessian*grad
        #
        elif self.hessian_vector_type == 2:
            tr_loss = self.net.loss(trn_X, trn_y) # L_train(w)

            grad_L_train_w = torch.autograd.grad(tr_loss, self.net.weights(), create_graph = True)

            hessian_vector = torch.autograd.grad(grad_L_train_w, self.net.alphas(),
                                                 grad_outputs = dw, retain_graph = False)

        #
        # HOAG (train_hessian_aw * inverse_train_hessian_ww * test_grad_w)
        #
        elif self.hessian_vector_type == 3:
            def flatten(t):  # No .cpu()
                return torch.cat([x.flatten() for x in t], dim=0)

            v_alphas = tuple(self.net.alphas())
            v_weights = tuple(self.net.weights())

            # step 1. calc test alpha- and weight-grad
            val_loss = self.net.loss(val_X, val_y)  # L_val(w)
            val_grad = torch.autograd.grad(
                val_loss,
                v_alphas + v_weights,
                create_graph=False)

            val_grad_a = val_grad[:len(v_alphas)]                    # d_a L_val
            val_grad_w = flatten(val_grad[len(v_alphas):]).double()  # d_w L_val

            # step 2. cg on the train weight-hessian and test weight-grad
            trn_loss = self.net.loss(trn_X, trn_y)  # L_train(w)
            trn_grad_w = flatten(torch.autograd.grad(
                trn_loss,
                v_weights,
                create_graph=True)) # d_w L_train(w)

            def calc_huge_hessian_vector(z):
                return flatten(torch.autograd.grad(
                    trn_grad_w,
                    v_weights,
                    grad_outputs=torch.from_numpy(z).to(trn_grad_w),
                    retain_graph=True)
                ).cpu()

            LinOp = LinearOperator(
                (len(trn_grad_w), len(trn_grad_w)),
                matvec=calc_huge_hessian_vector,
                dtype=np.dtype('float64'))

            # why these tol and maxiter?
            inv_hess_vect, info = cg(LinOp, val_grad_w.cpu().numpy(), tol=1e-3, maxiter=5)
            inv_hess_vect = torch.from_numpy(inv_hess_vect).to(trn_grad_w)

            # step 3. get \nabla^2_{\alpha\omega} L_train(\omega^*(\alpha), \alpha) q
            # Supply `grad()` with `q` as the `grad_outputs` for
            #  the Jacobian-vector product (see grad's docs).
            if True:
                hessian_vector = torch.autograd.grad(
                    trn_grad_w,
                    v_alphas,
                    grad_outputs=inv_hess_vect,
                    retain_graph=False)

            else:
                hessian_vector = self.compute_hessian_vector(inv_hess_vect, trn_X, trn_y)

            dalpha = val_grad_a

        else:
            hessian_vector = 0

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            if self.hessian_vector_type == 0:
                for alpha, da in zip(self.net.alphas(), dalpha):
                    alpha.grad = da
            elif self.hessian_vector_type in [1, 2]:
                for alpha, da, h in zip(self.net.alphas(), dalpha, hessian_vector):
                    alpha.grad = da - xi*h
            elif self.hessian_vector_type == 3:
                for alpha, da, h in zip(self.net.alphas(), dalpha, hessian_vector):
                    alpha.grad = da - h


    def compute_hessian_vector(self, dw, trn_X, trn_y):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_pos = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2. * eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_neg = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian
