import math
import numpy as np
from torch.autograd import Function, Variable
import torch

TensorT = torch.Tensor


class QuantFunc(Function):
    @staticmethod
    def forward(ctx, x, k):
        assert 0. <= x.min() and x.max() <= 1., f"x max: {x.max()}, x min: {x.min()}"

        k_up = float(2 ** k) - 1.
        x_fix = torch.round(x * k_up)

        return x_fix / k_up

    @staticmethod
    def backward(ctx, df):
        dx = dk = None
        dx = df.clone()

        return dx, dk


class LinearQuantFunc(Function):
    @staticmethod
    def forward(ctx, x, k):
        eps = 1e-8
        k_up = float(2 ** k) - 1.
        x_max = _batch_max(x)
        x_scale = 2. * x_max + eps
        x = x / x_scale + .5
        x = torch.round(x * k_up) / k_up
        x = (x - .5) * x_scale

        return x

    @staticmethod
    def backward(ctx, df):
        dx = dk = None
        dx = df.clone()

        return dx, dk


def clamp(input, min, max, inplace=False):
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def get_percentile_min_max(input, lower_percentile, upper_percentile, output_tensor=False):
    # sorted_input, _ = torch.sort(input.view(-1), dim=0)
    # input = input.view(-1)
    input_length = input.shape[0]
    lower_index = round(input_length * lower_percentile * 0.01)
    upper_index = round(input_length * upper_percentile * 0.01)
    lower_bound, _ = torch.topk(input, lower_index, largest=False, sorted=False)
    upper_bound, _ = torch.topk(input, input_length - upper_index, largest=True, sorted=False)
    if not output_tensor:
        lower_bound = torch.max(lower_bound).item()
        upper_bound = torch.min(upper_bound).item()
    elif output_tensor:
        lower_bound = torch.max(lower_bound)
        upper_bound = torch.min(upper_bound)
    # lower_bound = lower_bound[-1].item()
    # upper_bound = upper_bound[-1].item()
    # lower_bound = sorted_input[lower_index].squeeze()
    # upper_bound = sorted_input[upper_index].squeeze()
    # print("lower_bound = ", lower_bound)
    # print("upper_bound = ", upper_bound)
    return lower_bound, upper_bound


def linear_quantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    # scale and zero_point can be broadcasting to the same shape as input
    # the * and - here are element-wise operations
    return torch.round(scale * input - zero_point)


def linear_quantize_clamp(input, scale, zero_point, clamp_min, clamp_max, inplace=False):
    output = linear_quantize(input, scale, zero_point, inplace)
    return clamp(output, clamp_min, clamp_max, inplace)


def asymmetric_linear_quantization_params(num_bits, saturation_min, saturation_max,
                                          integral_zero_point=True, signed=False):
    n = 2 ** num_bits - 1

    # indicator = 0
    #
    # for i in (saturation_max - saturation_min):
    #   if i.abs() < 0.0000000001:
    #       # print("all zero warning")
    #       indicator = 1
    #
    # if indicator == 1:
    scale = n / torch.clamp((saturation_max - saturation_min), min=0.0000000001)
    # print(scale)

    # else:

    # scale = n / (saturation_max - saturation_min)

    zero_point = scale * saturation_min

    if integral_zero_point:
        if isinstance(zero_point, torch.Tensor):
            zero_point = zero_point.round()
        else:
            zero_point = float(round(zero_point))
    if signed:
        zero_point += 2 ** (num_bits - 1)
    return scale, zero_point


def linear_dequantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    # scale and zero_point can be broadcasting to the same shape as input
    # the + and / here are element-wise operations
    return (input + zero_point) / scale


def affine_quant_func(x, k, lower_bound, upper_bound):
    """ Quantize input variables via affine methods.

    input type: TensorT, int, float, float
    output type: float, TensorT, int

    Returns:
            - delta: magnitude of quantized values;
            - quant_idx: same shape with x;
            - shift_idx: quantized value w.r.t. real value 0.

    """
    assert lower_bound <= upper_bound, "got lower_bound = {}, while upper_bound = {}".format(
        lower_bound, upper_bound)

    # asymmetic quantization, 2 ** k - 1 rather than 2 ** (k-1) - 1
    delta = (upper_bound - lower_bound) / (2. ** k - 1.)
    x = torch.clamp(x, lower_bound, upper_bound)

    quant_idx = torch.round((x - lower_bound) / delta)
    shift_idx = math.floor(abs(lower_bound) / delta)

    return delta, quant_idx, shift_idx


def nudge_min_max(k, x_min, x_max):
    """ This function applies a small shift on data range to make sure 0 is quantized to exact 0.

    k is int type, x_min and x_max are float type.
    0 is important since there are lots of 0 in data, and it doesn't require operations.

    """
    assert x_min <= x_max, "got x_min = {}, while x_max = {}".format(
        x_min, x_max)

    modified_min, modified_max = x_min, x_max

    if 0. <= x_min:
        modified_min = 0.
    elif x_max <= 0.:
        modified_max = 0.
    else:
        modified_range = modified_max - modified_min
        delta = modified_range / (2. ** k - 1.)
        mismatch = abs(modified_min) % delta

        if mismatch < (delta / 2.):
            nudge = mismatch
        else:
            nudge = mismatch - delta

        modified_min += nudge
        modified_max += nudge

    return modified_min, modified_max


def get_tensor_min_max(t, per_dim=None):
    if per_dim is None:
        return t.min(), t.max()
    if per_dim > t.dim():
        raise ValueError('Got per_dim={0}, but tensor only has {1} dimensions', per_dim, t.dim())
    view_dims = [t.shape[i] for i in range(per_dim + 1)] + [-1]
    tv = t.view(*view_dims)
    return tv.min(dim=-1)[0], tv.max(dim=-1)[0]


def symmetric_quant_func(x, k, x_mag):
    """
    inputs: TensorT, int, float
    outputs: TensorT, float, TensorT

    """
    assert 0 < x_mag

    x_min, x_max = -x_mag, x_mag
    idx = (x_min <= x) * (x <= x_max)
    x = torch.clamp(x, x_min, x_max)
    n = 2 ** (k - 1) - 1
    q_d = x_max / n
    q_i = torch.round(x / q_d)

    return q_i, q_d, idx


class AsymmetricQuantFunction(Function):
    @staticmethod
    def forward(ctx, x, k, x_min=None, x_max=None, per_channel=False, percentile_mode=False):
        if x_min is None or x_max is None:
            x_min, x_max = x.min(), x.max()

        if per_channel:
            # lower_bound = torch.zeros(x.data.size()[1])
            # upper_bound = torch.zeros(x.data.size()[1])
            # delta = torch.zeros(x.data.size()[1])
            # quant_idx = torch.zeros(x.data.size())
            # # 	# print(3, time.time())
            # for i in range(x.data.size()[1]):
            #     lower_bound[i], upper_bound[i] = nudge_min_max(k, x_min[i].item(), x_max[i].item())
            #     delta[i], quant_idx[:, i], shift_idx = affine_quant_func(x[:, i], k, lower_bound[i], upper_bound[i])
            # quant_x = delta.to(x.device) * quant_idx.to(x.device) + x_min
            # print( "original", quant_x, quant_x.shape )

            scale, zero_point = asymmetric_linear_quantization_params(k, x_min, x_max)
            # print("scale.shape = ", scale.shape)
            # print("x.shape = ", x.shape)

            new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)

            # Need to clamp x if percentile mode is True
            if percentile_mode:
                n = 2 ** k - 1
                new_quant_x = torch.clamp(new_quant_x, 0, n)

            # print("new_quant_x.shape = ", new_quant_x.shape)
            quant_x = linear_dequantize(new_quant_x, scale, zero_point, inplace=False)
            # print("quant_x.shape = ", quant_x.shape)
            # print( "new", quant_x, quant_x.shape )
            # print( quant_x == new_quant_x )
            # exit()
            # 	print(quant_x)
            """
            quant_x = torch.zeros(x.data.size()).cuda()
            for i in range(x.data.size()[1]):
                lower_bound, upper_bound = nudge_min_max(
                    k, x_min[i].item(), x_max[i].item())
                delta, quant_idx_i, shift_idx = affine_quant_func(
                    x[:, i], k, lower_bound, upper_bound)
                quant_x[:, i] = delta * quant_idx_i + x_min[i].item()
                #     x[:, i, :, :], k, lower_bound, upper_bound)
                # quant_x[:, i, :, :] = delta * quant_idx_i + x_min[i].item()
            # print(4, time.time())
            """
            return torch.autograd.Variable(quant_x)

        else:
            ## use advanced quantizer
            # scale, zero_point = asymmetric_linear_quantization_params(k, x_min, x_max)
            # new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
            # quant_x = linear_dequantize(new_quant_x, scale, zero_point, inplace=False)
            # return quant_x

            # use portable quantizer
            lower_bound, upper_bound = nudge_min_max(k, x_min, x_max)
            delta, quant_idx, shift_idx = affine_quant_func(
                x, k, lower_bound, upper_bound)
            return delta * quant_idx + x_min

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None, None, None, None


class SymmetricQuantFunction(Function):
    @staticmethod
    def forward(ctx, x, k, magnitude=None):
        if magnitude is None:
            magnitude = x.abs().max()
        x_i, x_d, idx = symmetric_quant_func(x, k, magnitude)
        ctx.idx = idx

        return x_d * x_i

    @staticmethod
    def backward(ctx, grad_output):
        dk = d_magnitude = None
        idx = ctx.idx

        dx = grad_output * Variable(idx.float())

        return dx, dk, d_magnitude


class QuantGradFunc(Function):
    @staticmethod
    def forward(ctx, x, k):
        """
        Args:
                x: input activation in forward pass.
                k: number of bits used to quantize backward gradient.
        """
        ctx.bit_g = k

        return x

    @staticmethod
    def backward(ctx, grad_output):
        dx = None
        dk = None
        k = ctx.bit_g

        if k == 32:
            dx = grad_output.clone()
        else:
            dtype = torch.cuda.FloatTensor if grad_output.is_cuda else torch.FloatTensor
            n = 0.5 / float(2 ** k - 1)
            bias = Variable(
                dtype(grad_output.shape).uniform_(-n, n), requires_grad=False)
            df_max = _batch_max(grad_output)
            df = grad_output / df_max
            df = torch.clamp(df * 0.5 + 0.5 + bias, 0., 1.)
            df_fix = _quantize(df, k)
            dx = df_max * (df_fix - 0.5) * 2.

        return dx, dk


def _quantize(x, k):
    """

    inputs: Variable, int
    output: Variable

    Args:
            x: input variables within [0, 1].
            k: number of bit to represent fixed x.

    Returns:
            Quantized input x, with same shape and dtype.

    """
    assert 0. <= x.data.min() and x.data.max() <= 1.

    x_up = float(2 ** k) - 1
    x_fix = torch.round(x * x_up)

    return x_fix / x_up


def _batch_max(x):
    """ Instance-level maximum within a batch.
    identical to tf.reduce_max(x, axis=(1... x.dim))

    input: TensorT
    output: TensorT

    """
    reduce_idx = (x.shape[0],) + (1,) * (x.dim() - 1)
    x_max = x.abs().view(x.shape[0], -1).max(1)[0].view(reduce_idx)

    return x_max
