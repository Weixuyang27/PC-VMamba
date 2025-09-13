import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter, Softmax
from einops import rearrange, repeat
import logging

try:
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn_no_out_proj
except ImportError:
    mamba_inner_fn_no_out_proj = None


class TransformerCluster(nn.Module):
    def __init__(self, dim, w_size=7, clusters=7, num_heads=8):
        super().__init__()
        self.dim = dim
        self.w_size = w_size
        self.clusters = clusters
        self.num_heads = num_heads

        # Transformer components
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        # Projection layers
        self.f = nn.Conv2d(self.dim // 2, self.dim // 16, kernel_size=1)
        self.v = nn.Conv2d(self.dim // 2, self.dim // 16, kernel_size=1)
        self.p = nn.Conv2d(self.dim // 16, self.dim, kernel_size=1)

    def forward(self, x_in):
        # Reshape input
        x = rearrange(x_in, "b e (Wg w) (Hg h)-> (b Wg Hg) e w h", Wg=self.w_size, Hg=self.w_size)

        # Split features
        x1, x2 = x.chunk(2, dim=1)
        f = self.f(x1)
        v = self.v(x2)

        # Reshape for transformer
        b, c, h, w = f.shape
        f_flat = rearrange(f, 'b c h w -> b (h w) c')
        v_flat = rearrange(v, 'b c h w -> b (h w) c')

        # Apply transformer
        f_norm = self.norm1(f_flat)
        attn_output, _ = self.attention(f_norm, f_norm, f_norm)
        f_transformed = f_flat + attn_output

        f_transformed = self.norm2(f_transformed)
        f_transformed = f_transformed + self.ffn(f_transformed)

        # Reshape back
        f_transformed = rearrange(f_transformed, 'b (h w) c -> b c h w', h=h, w=w)

        # Combine with value features
        out = f_transformed + v
        out = self.p(out)

        # Reshape to original format
        out = rearrange(out, "(b Wg Hg) e w h -> b e (Wg w) (Hg h)", Wg=self.w_size, Hg=self.w_size)
        out = out.flatten(2)

        return out


class DepthWiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(DepthWiseConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class PAM_Module(nn.Module):
    """ Position attention module"""

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = DepthWiseConv2d(in_dim, in_dim, kernel_size=1)
        self.key_conv = DepthWiseConv2d(in_dim, in_dim, kernel_size=1)
        self.value_conv = DepthWiseConv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()

        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x

        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()

        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        return out


class DA_Block(nn.Module):
    def __init__(self, dim, w_size=7):
        super(DA_Block, self).__init__()
        inter_channels = dim // 32  # 减少中间通道数
        self.w_size = w_size

        # 定义初始卷积层
        self.conv5a = nn.Sequential(
            DepthWiseConv2d(dim, inter_channels, 3, padding=1),
            nn.ReLU()
        )

        self.conv5c = nn.Sequential(
            DepthWiseConv2d(dim, inter_channels, 3, padding=1),
            nn.ReLU()
        )

        # 定义位置注意力模块 (PAM) 和通道注意力模块 (CAM)
        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)

        # SA 和 SC 输出后的卷积层
        self.conv51 = nn.Sequential(
            DepthWiseConv2d(inter_channels, inter_channels, 3, padding=1),
            nn.ReLU()
        )
        self.conv52 = nn.Sequential(
            DepthWiseConv2d(inter_channels, inter_channels, 3, padding=1),
            nn.ReLU()
        )

        # 最后的卷积操作
        self.conv6 = nn.Sequential(
            nn.Dropout2d(0.05, False),
            DepthWiseConv2d(inter_channels, dim, 1),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Dropout2d(0.05, False),
            DepthWiseConv2d(inter_channels, dim, 1),
            nn.ReLU()
        )

        self.conv8 = nn.Sequential(
            nn.Dropout2d(0.05, False),
            DepthWiseConv2d(dim, dim, 1),
            nn.ReLU()
        )

    def forward(self, x):
        # 获取输入的 batch_size 和空间尺寸
        if len(x.shape) == 3:  # [B, L, C]
            b, l, c = x.shape
            h = w = int(math.sqrt(l))
            x = x.transpose(1, 2).reshape(b, c, h, w)
        else:  # [B, C, H, W]
            b, c, h, w = x.shape

        # 通过卷积和注意力机制进行特征提取
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output1 = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output2 = self.conv7(sc_conv)

        feat_sum = sa_output1 + sc_output2

        # 最后的卷积操作
        sasc_output = self.conv8(feat_sum)

        # 重塑输出为符合预期的形状
        if len(x.shape) == 3:
            sasc_output = sasc_output.flatten(2).transpose(1, 2)  # [B, L, C]
        else:
            sasc_output = sasc_output.flatten(2)  # [B, C, L]

        return sasc_output


class MultiScan(nn.Module):
    ALL_CHOICES = ('h', 'h_flip', 'v', 'v_flip', 'c2', 'c5')

    def __init__(self, dim, choices=None, token_size=(14, 14), win_size=8):
        super().__init__()
        self.token_size = token_size
        self.win_size = win_size
        if choices is None:
            self.choices = MultiScan.ALL_CHOICES
            self.norms = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=False) for _ in self.choices])
            self.weights = nn.Parameter(1e-3 * torch.randn(len(self.choices), 1, 1, 1))
            self._iter = 0
            self.logger = logging.getLogger()
            self.search = True
        else:
            self.choices = choices
            self.search = False
        numbers = [item[1:] for item in self.choices if item.startswith('c') and item[1:].isdigit()]
        if numbers:
            number = int(numbers[0])
            # self.local_cluster = TransformerCluster(dim=dim, w_size=self.win_size, clusters=number)
            self.local_cluster = DA_Block(dim=dim, w_size=self.win_size)

    def forward(self, xs):
        """
        Input @xs: [[B, L, D], ...]
        """
        if self.search:
            weights = self.weights.softmax(0)
            xs = [norm(x) for norm, x in zip(self.norms, xs)]
            xs = torch.stack(xs) * weights
            x = xs.sum(0)
            if self._iter % 200 == 0:
                if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    self.logger.info(str(weights.detach().view(-1).tolist()))
            self._iter += 1
        else:
            x = torch.stack(xs).sum(0)
        return x

    def multi_scan(self, x):
        """
        Input @x: shape [B, L, D]
        """
        xs = []
        for direction in self.choices:
            xs.append(self.scan(x, direction))
        return xs

    def multi_reverse(self, xs):
        new_xs = []
        for x, direction in zip(xs, self.choices):
            new_xs.append(self.reverse(x, direction))
        return new_xs

    def scan(self, x, direction='h'):
        """
        Input @x: shape [B, L, D] or [B, C, H, W]
        Return torch.Tensor: shape [B, D, L]
        """
        H, W = self.token_size
        if len(x.shape) == 3:
            if direction == 'h':
                return x.transpose(-2, -1)
            elif direction == 'h_flip':
                return x.transpose(-2, -1).flip([-1])
            elif direction == 'v':
                return rearrange(x, 'b (h w) d -> b d (w h)', h=H, w=W)
            elif direction == 'v_flip':
                return rearrange(x, 'b (h w) d -> b d (w h)', h=H, w=W).flip([-1])
            elif direction.startswith('c'):
                # K = int(direction[1:].split('_')[0])
                # flip = direction.endswith('flip')
                # return LocalScanTriton.apply(x.transpose(-2, -1), K, flip, H, W)
                return self.local_cluster(x)
            else:
                raise RuntimeError(f'Direction {direction} not found.')
        elif len(x.shape) == 4:
            if direction == 'h':
                return x.flatten(2)
            elif direction == 'h_flip':
                return x.flatten(2).flip([-1])
            elif direction == 'v':
                return rearrange(x, 'b d h w -> b d (w h)', h=H, w=W)
            elif direction == 'v_flip':
                return rearrange(x, 'b d h w -> b d (w h)', h=H, w=W).flip([-1])
            elif direction.startswith('c'):
                # K = int(direction[1:].split('_')[0])
                # flip = direction.endswith('flip')
                # return LocalScanTriton.apply(x, K, flip, H, W).flatten(2)
                return self.local_cluster(x)
            else:
                raise RuntimeError(f'Direction {direction} not found.')

    def reverse(self, x, direction='h'):
        """
        Input @x: shape [B, D, L]
        Return torch.Tensor: shape [B, D, L]
        """
        H, W = self.token_size
        if direction == 'h':
            return x
        elif direction == 'h_flip':
            return x.flip([-1])
        elif direction == 'v':
            return rearrange(x, 'b d (h w) -> b d (w h)', h=H, w=W)
        elif direction == 'v_flip':
            return rearrange(x.flip([-1]), 'b d (h w) -> b d (w h)', h=H, w=W)
        elif direction.startswith('c'):
            return x
        else:
            raise RuntimeError(f'Direction {direction} not found.')

    def __repr__(self):
        scans = ', '.join(self.choices)
        return super().__repr__().replace(self.__class__.__name__, f'{self.__class__.__name__}[{scans}]')


class BiAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.125, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        # self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        # self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()

    def forward(self, x):
        ori_x = x
        x = self.norm(x)
        x_global = x.mean(1, keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        # x_local = self.act_fn(self.local_reduce(x))

        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)  # [B, 1, C]

        attn = c_attn  # * s_attn  # [B, N, C]
        return ori_x * attn


class MultiMamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,  # Fused kernel options
            layer_idx=None,
            device=None,
            dtype=None,
            bimamba_type="none",
            directions=None,
            token_size=(14, 14),
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.token_size = token_size

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.activation = "silu"
        self.act = nn.SiLU()

        self.multi_scan = MultiScan(self.d_inner, choices=directions, token_size=token_size)
        '''new for search'''
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        for i in range(len(self.multi_scan.choices)):
            setattr(self, f'A_log_{i}', nn.Parameter(A_log))
            getattr(self, f'A_log_{i}')._no_weight_decay = True

            conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            setattr(self, f'conv1d_{i}', conv1d)

            x_proj = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            setattr(self, f'x_proj_{i}', x_proj)

            dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            # Initialize special dt projection to preserve variance at initialization
            dt_init_std = self.dt_rank ** -0.5 * dt_scale
            if dt_init == "constant":
                nn.init.constant_(dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError

            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

            setattr(self, f'dt_proj_{i}', dt_proj)

            D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            D._no_weight_decay = True
            setattr(self, f'D_{i}', D)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.attn = BiAttn(self.d_inner)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        xz = self.in_proj(hidden_states)

        xs = self.multi_scan.multi_scan(xz)  # [[BDL], [BDL], ...]
        outs = []
        for i, xz in enumerate(xs):
            # xz = rearrange(xz, "b l d -> b d l")
            A = -torch.exp(getattr(self, f'A_log_{i}').float())
            conv1d = getattr(self, f'conv1d_{i}')
            x_proj = getattr(self, f'x_proj_{i}')
            dt_proj = getattr(self, f'dt_proj_{i}')
            D = getattr(self, f'D_{i}')

            out = mamba_inner_fn_no_out_proj(
                xz,
                conv1d.weight,
                conv1d.bias,
                x_proj.weight,
                dt_proj.weight,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                D,
                delta_bias=dt_proj.bias.float(),
                delta_softplus=True,
            )
            outs.append(out)

        outs = self.multi_scan.multi_reverse(outs)
        outs = [self.attn(rearrange(out, 'b d l -> b l d')) for out in outs]
        out = self.multi_scan(outs)
        out = F.linear(out, self.out_proj.weight, self.out_proj.bias)

        return out


try:
    import selective_scan_cuda_oflex
except:
    selective_scan_cuda_oflex = None


class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None


class MultiVMamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,  # Fused kernel options
            layer_idx=None,
            device=None,
            dtype=None,
            bimamba_type="none",
            directions=None,
            token_size=(14, 14),
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.token_size = token_size

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.activation = "silu"
        self.act = nn.SiLU()

        self.multi_scan = MultiScan(self.d_inner, choices=directions, token_size=token_size)
        '''new for search'''
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        for i in range(len(self.multi_scan.choices)):
            setattr(self, f'A_log_{i}', nn.Parameter(A_log))
            getattr(self, f'A_log_{i}')._no_weight_decay = True

            x_proj = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            setattr(self, f'x_proj_{i}', x_proj)

            conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            setattr(self, f'conv1d_{i}', conv1d)

            dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            # Initialize special dt projection to preserve variance at initialization
            dt_init_std = self.dt_rank ** -0.5 * dt_scale
            if dt_init == "constant":
                nn.init.constant_(dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError

            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

            setattr(self, f'dt_proj_{i}', dt_proj)

            D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            D._no_weight_decay = True
            setattr(self, f'D_{i}', D)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.attn = BiAttn(self.d_inner)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch_size, seq_len, dim = hidden_states.shape
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=2)
        z = self.act(z)

        xs = self.multi_scan.multi_scan(x)
        outs = []
        for i, xz in enumerate(xs):
            xz = rearrange(xz, "b l d -> b d l")
            A = -torch.exp(getattr(self, f'A_log_{i}').float())
            x_proj = getattr(self, f'x_proj_{i}')
            conv1d = getattr(self, f'conv1d_{i}')
            dt_proj = getattr(self, f'dt_proj_{i}')
            D = getattr(self, f'D_{i}')

            xz = conv1d(xz)[:, :, :seq_len]
            xz = self.act(xz)

            N = A.shape[-1]
            R = dt_proj.weight.shape[-1]

            x_dbl = F.linear(rearrange(xz, 'b d l -> b l d'), x_proj.weight)
            dts, B, C = torch.split(x_dbl, [R, N, N], dim=2)
            dts = F.linear(dts, dt_proj.weight)

            dts = rearrange(dts, 'b l d -> b d l')
            B = rearrange(B, 'b l d -> b 1 d l')
            C = rearrange(C, 'b l d -> b 1 d l')
            D = D.float()
            delta_bias = dt_proj.bias.float()

            out = SelectiveScanOflex.apply(xz.contiguous(), dts.contiguous(), A.contiguous(), B.contiguous(),
                                           C.contiguous(), D.contiguous(), delta_bias, True, True)

            outs.append(rearrange(out, "b d l -> b l d"))

        outs = self.multi_scan.multi_reverse(outs)
        outs = [self.attn(out) for out in outs]
        out = self.multi_scan(outs)
        out = out * z
        out = self.out_proj(out)

        return out


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


def fea_num_reverse(x, H, W):
    K = 7
    if len(x.shape) == 4:
        B, C, H, W = x.shape
    elif len(x.shape) == 3:
        B, C, _ = x.shape
        assert H is not None and W is not None, "x must be BCHW format to infer the H W"
    else:
        raise RuntimeError(f"Unsupported shape of x: {x.shape}")
    B, C, H, W = int(B), int(C), int(H), int(W)

    Hg, Wg = math.ceil(H / K), math.ceil(W / K)
    Hb, Wb = Hg * K, Wg * K

    x = rearrange(x, "b c (w h ) -> b c w h", w=Wb, h=Hb)
    x = x[:, :, :W, :H]
    x = x.flatten(2)

    return x




