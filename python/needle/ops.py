"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        return out_grad * self.scalar * a ** (self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad / b, out_grad * (-a / (b * b))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return (a / self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad / self.scalar,)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes:
            ax0, ax1 = self.axes[0], self.axes[1]
        else:
            ax0, ax1 = a.ndim - 2, a.ndim - 1
        permute_axes = list(range(a.ndim))
        permute_axes[ax0], permute_axes[ax1] = ax1, ax0
        return a.permute(permute_axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, axes=self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a.compact(), self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        # print("self.shape: ", self.shape)
        # print("reshaping out grad to a.shape: ", out_grad.shape, a.shape)
        return reshape(out_grad, a.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        sum_axes = []

        original_shape = a.shape
        j = len(original_shape) - 1

        broadcasted_shape = out_grad.shape

        for i in reversed(range(len(broadcasted_shape))):
            if j < 0 or broadcasted_shape[i] != original_shape[j]:
                sum_axes.append(i)

            j -= 1

        return reshape(summation(out_grad, axes=tuple(sum_axes)), (original_shape))
        ### END YOUR SOLUTION
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if isinstance(self.axes, tuple):
            for axis in reversed(sorted(self.axes)):
                a = a.sum(axis=axis)
            return a
        return array_api.summation(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        target_shape = list(node.inputs[0].shape)
        if self.axes is None:
            axes = range(len(target_shape))
        elif isinstance(self.axes, tuple):
            axes = self.axes
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            raise ValueError("Incorrect axes type in summation! axes must be int, tuple or None")
        
        for axis in axes:
            target_shape[axis] = 1
        return out_grad.reshape(target_shape).broadcast_to(a.shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        if len(a.shape) > len(b.shape):
            axis = tuple(i for i in range(len(a.shape) - len(b.shape)))
            return out_grad @ b.transpose(), (a.transpose() @ out_grad).sum(axes=axis)

        if len(b.shape) > len(a.shape):
            axis = tuple(i for i in range(len(b.shape) - len(a.shape)))
            return (out_grad @ b.transpose()).sum(axes=axis), a.transpose() @ out_grad

        return out_grad @ b.transpose(), a.transpose() @ out_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * -1
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        # one = broadcast_to(Tensor([1]), out_grad.shape)
        return divide(out_grad, a)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        return exp(a) * out_grad
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        output = node.realize_cached_data()
        return out_grad * Tensor(output > 0, device=out_grad.device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        # if isinstance(axes, int):
        #     axes=(axes,)
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Z_shape = list(Z.shape)

        if self.axes is not None:
            if isinstance(self.axes, int):
                Z_shape[self.axes] = 1
            else:
                for i in self.axes:
                    Z_shape[i] = 1
        else:
            Z_shape = [1] * len(Z.shape)

        # print(Z_shape)
        self.Z_shape = Z_shape

        max_z = Z.max(axis=self.axes)
        self.max_z = max_z
        exp_z = array_api.exp(Z - max_z.reshape(Z_shape).broadcast_to(Z.shape))
        return array_api.log(array_api.summation(exp_z, axis=self.axes)) + max_z
    
        # max_z_original = Z.max(axis=self.axes, keepdims=True) 
        # max_z_reduce = Z.max(axis=self.axes)
        # return array_api.log(array_api.summation(array_api.exp(Z - max_z_original.broadcast_to(Z.shape)), axis=self.axes)) + max_z_reduce 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z, = node.inputs
        # print(Z.shape)
        # print(out_grad.shape)
        # print(self.Z_shape)

        out_grad_reshaped = out_grad.reshape(self.Z_shape)
        # print(out_grad_reshaped.shape)
        max_z = Tensor(self.max_z, device=Z.device)
        diff = Z - broadcast_to(max_z.reshape(self.Z_shape), Z.shape)

        exp_diff = exp(diff)
        sum_exp_diff = summation(exp_diff, axes=self.axes)

        dZ = exp_diff / broadcast_to(sum_exp_diff.reshape(self.Z_shape), Z.shape) # reshape then broadcast
        return dZ * broadcast_to(out_grad_reshaped, Z.shape)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.tanh()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        return (1 - tanh(a) ** 2) * out_grad
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        target_shape = list(args[0].shape)
        target_shape.insert(self.axis, len(args))
        # print(target_shape)
        out = array_api.empty(target_shape, device=args[0].device)
        # out = init.zeros(*target_shape)
        # print(type(out))
        slices = [slice(0, s) for s in target_shape]
        # print(slices)
        for i, arr in enumerate(args):
            slices[self.axis] = slice(i, i + 1)
            out[tuple(slices)] = arr
        return out
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        num_arr = A.shape[self.axis]
        target_shape = list(A.shape)
        target_shape.pop(self.axis)
        splited = []
        slices = [slice(0, s) for s in A.shape]
        for i in range(num_arr):
            slices[self.axis] = slice(i, i + 1)
            splited.append(A[tuple(slices)].compact().reshape(target_shape))

        return tuple(splited)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        for axis in self.axes:
            new_shape[axis] *= (self.dilation + 1)

        out = array_api.full(new_shape, 0, device=a.device)
        selection = [slice(0, n) for n in new_shape]
        for axis in self.axes:
            selection[axis] = slice(0, new_shape[axis], self.dilation + 1)

        out[tuple(selection)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        selection = [slice(0, n) for n in a.shape]
        for axis in self.axes:
            selection[axis] = slice(0, a.shape[axis], self.dilation + 1)

        return a[tuple(selection)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        pad = [(self.padding, self.padding) if i == 1 or i == 2 else (0, 0) for i in range(len(A.shape))]
        A = A.pad(pad)

        N, H, W, C_in = A.shape
        assert C_in == B.shape[2], "C_in dimension of image and kernel weights doesn't match"
        assert B.shape[0] == B.shape[1], "kernel size height and width not equal, currently not supported"
        K, _, _, C_out = B.shape
        Ns, Hs, Ws, C_ins = A.strides
        inner_dim = K * K * C_in
        H_out, W_out = (H - K) // self.stride + 1, (W - K) // self.stride + 1

        im2col = A.as_strided((N, H_out, W_out, K, K, C_in), (Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, C_ins)).compact()
        # print("A shape: ", A.shape)
        # print("B shape: ", B.shape)
        # print(f"H out {H_out}, W out {W_out}, self.stride {self.stride}")
        # print("im2col shape: ", im2col.shape)
        out = im2col.reshape((N * H_out * W_out, inner_dim)) @ B.compact().reshape((inner_dim, C_out))
        return out.compact().reshape((N, H_out, W_out, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs
        K, _, _, _ = W.shape

        if self.stride > 1:
            out_grad = dilate(out_grad, (1, 2), self.stride-1)

        W_modified = transpose(flip(W, axes=(0, 1)), (2, 3))
        X_grad = conv(out_grad, W_modified, padding=K-1-self.padding)

        X_permuted = transpose(X, (0, 3))
        out_grad_permuted = transpose(transpose(out_grad, (0, 1)), (1, 2))
        W_grad = conv(X_permuted, out_grad_permuted, padding=self.padding)
        W_grad = transpose(transpose(W_grad, (0, 1)), (1, 2))

        return X_grad, W_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



