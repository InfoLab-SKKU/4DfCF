import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# 4DfCF_


NEG_INF = -1000000


class Mlp(nn.Module):
    r"""2-layer MLP"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




class PositionalEmbedding(nn.Module):
    def __init__(self, dim, patch_dim):
        super().__init__()
        self.dim = dim
        self.patch_dim = patch_dim  # Expecting a tuple (D, H, W, T)
        d, h, w, t = patch_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, d, h, w, dim))
        self.time_embed = nn.Parameter(torch.zeros(1, t, 1, 1, 1, dim))
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.time_embed, std=0.02)

    def forward(self, x):
        B, D, H, W, T, C = x.shape
        x = x + self.pos_embed.permute(0, 2, 3, 4, 1).contiguous()  # Rearrange to match dimensions
        x = x + self.time_embed.permute(0, 5, 1, 2, 3, 4).contiguous()  # Add time embedding
        return x


class Attention(nn.Module):
    def __init__(self, dim, group_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.group_size = group_size  # Assuming (D, H, W, T)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Positional embeddings
        self.pos_embedding = PositionalEmbedding(dim, group_size)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.pos_embedding(x)  # Add positional embeddings
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # Matrix multiplication for attention scores
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self):
        return f'dim={self.dim}, group_size={self.group_size}, num_heads={self.num_heads}'

    def flops(self, N):
        flops = N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops


class CrossFormerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, group_size=4, interval=8, lsda_flag=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_patch_size=1,
                 pad_type=0, use_extra_conv=True, use_cpe=False, no_mask=False, adaptive_interval=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (D, H, W, T)
        self.num_heads = num_heads
        self.group_size = group_size
        self.interval = interval
        self.lsda_flag = lsda_flag
        self.mlp_ratio = mlp_ratio
        self.num_patch_size = num_patch_size
        self.pad_type = pad_type
        self.use_extra_conv = use_extra_conv
        self.use_cpe = use_cpe

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, group_size=to_2tuple(self.group_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            position_bias=(not use_cpe))

        if self.use_extra_conv:
            self.ex_conv = nn.Conv3d(dim, dim, kernel_size=(3, 3, 3), padding=(1, 1, 1), groups=dim)
            self.ex_ln = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm_cpe = norm_layer(dim) if self.use_cpe else None

        # Calculate effective group size and padding
        effective_group_size = self.group_size * self.interval if lsda_flag else self.group_size
        self.padding = (effective_group_size - input_resolution[0] % effective_group_size) % effective_group_size

    def forward(self, x):
        B, D, H, W, T, C = self.input_resolution
        x = x.view(B, D, H, W, T, C)  # Assume x is reshaped properly to (B, D, H, W, T, C)

        # Padding for group size compatibility
        if self.padding > 0:
            x = F.pad(x, (0, 0, 0, self.padding, 0, self.padding, 0, self.padding), "constant", 0)

        _, Dp, Hp, Wp, Tp, _ = x.shape

        # Reshape for attention based on SDA or LDA
        if self.lsda_flag == 0:  # SDA
            x = x.view(B, Dp // self.group_size, self.group_size, Hp // self.group_size, self.group_size,
                       Wp // self.group_size, self.group_size, T, C)
            x = x.permute(0, 1, 3, 5, 7, 2, 4, 6, 8).contiguous()
            x = x.reshape(B, -1, self.group_size ** 3 * T, C)
        else:  # LDA
            I = self.interval
            G = Gh = Gw = self.group_size
            Rh, Rw = Hp // (Gh * I), Wp // (Gw * I)

            x = x.view(B, Rh, Gh, I, Rw, Gw, I, T, C)
            x = x.permute(0, 1, 4, 7, 3, 6, 2, 5, 8).contiguous()
            x = x.reshape(B, -1, Gh * Gw * T, C)

        # Attention processing
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_path(x)

        # Reverse reshaping
        x = x.view(B, Dp // self.group_size, Hp // self.group_size, Wp // self.group_size, Tp, self.group_size,
                   self.group_size, self.group_size, self.dim)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4, 8).reshape(B, Dp, Hp, Wp, Tp, self.dim)

        # Removing padding
        if self.padding > 0:
            x = x[:, :D, :H, :W, :T, :]

        # Extra convolution layer
        if self.use_extra_conv:
            x = self.ex_conv(x)
            x = self.ex_ln(x)

        # MLP and final normalization
        x = self.mlp(self.norm2(x)) + x

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"group_size={self.group_size}, lsda_flag={self.lsda_flag}, mlp_ratio={self.mlp_ratio}, " \
               f"interval={self.interval}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # LSDA
        nW = H * W / self.group_size / self.group_size
        flops += nW * self.attn.flops(self.group_size * self.group_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer for 4D fMRI images with maintained time dimension.

    Args:
        input_resolution (tuple[int]): Resolution of input feature in the format (D, H, W).
        time_dim (int): Number of time steps.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, time_dim, dim, norm_layer=nn.LayerNorm, patch_size=[2]):
        super().__init__()
        self.input_resolution = input_resolution  # Spatial dimensions (D, H, W)
        self.time_dim = time_dim  # Time dimension
        self.dim = dim
        self.reductions = nn.ModuleList()
        self.patch_size = patch_size
        self.norm = norm_layer(dim)

        # Assuming same patch size for D, H, W dimensions
        stride = patch_size[0]
        padding = (stride - 1) // 2
        out_dim = dim * 2  # Doubling the dimension

        # Create a 3D convolution to merge patches in spatial dimensions
        self.reductions.append(nn.Conv3d(dim, out_dim, kernel_size=patch_size,
                                         stride=patch_size, padding=padding))

    def forward(self, x):
        """
        x: B, T, D, H, W, C
        """
        B, T, D, H, W, C = x.shape
        assert D == self.input_resolution[0] and H == self.input_resolution[1] and W == self.input_resolution[2], \
            "input feature has wrong size"

        # Reformat the tensor to match (B*T, C, D, H, W) for 3D convolution
        x = x.permute(0, 5, 2, 3, 4, 1).contiguous()  # Reorder dimensions to (B, C, D, H, W, T)
        x = x.view(B * T, C, D, H, W)  # Merge batch and time for convolution

        x = self.norm(x)  # Normalize before reduction
        x = self.reductions[0](x)  # Apply 3D convolution

        # Calculate new dimensions
        D_new, H_new, W_new = D // self.patch_size[0], H // self.patch_size[0], W // self.patch_size[0]
        x = x.view(B, T, self.dim * 2, D_new, H_new, W_new)  # Separate batch and time, increase channels

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, time_dim={self.time_dim}, dim={self.dim}"

    def flops(self):
        D, H, W = self.input_resolution
        out_dim = self.dim * 2
        flops = self.time_dim * D * H * W * self.dim  # Normalization flops
        # flops for 3D convolutions
        flops += self.time_dim * (D // self.patch_size[0]) * (H // self.patch_size[0]) * (W // self.patch_size[0]) * \
                 (self.patch_size[0] ** 3) * self.dim * out_dim
        return flops


class Stage(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, group_size, interval,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 patch_size_end=[4], num_patch_size=None, use_cpe=False, pad_type=0,
                 no_mask=False, adaptive_interval=False, use_acl=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (D, H, W, T)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Building blocks with specific attention mechanism (SDA or LDA)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            lsda_flag = 0 if (i % 2 == 0) else 1  # Alternate between SDA and LDA
            use_extra_conv = ((i + 1) % 3 == 0) and (i < depth - 1) and use_acl

            block_group_size = group_size if isinstance(group_size, list) else [group_size] * depth
            block_interval = interval if isinstance(interval, list) else [interval] * depth

            self.blocks.append(CrossFormerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, group_size=block_group_size[i], interval=block_interval[i],
                lsda_flag=lsda_flag, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                num_patch_size=num_patch_size,
                use_extra_conv=use_extra_conv,
                use_cpe=use_cpe,
                pad_type=pad_type,
                no_mask=no_mask,
                adaptive_interval=adaptive_interval
            ))

        # Optional downsampling layer at the end of the stage
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer,
                                         patch_size=patch_size_end, num_input_patch_size=num_patch_size)
        else:
            self.downsample = None

    def forward(self, x):
        # Pass through each block
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        # Apply downsampling if it exists
        if self.downsample is not None:
            x = self.downsample(x)

        return x

    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        # Calculate FLOPs by summing each block and the optional downsampling layer
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" 4D fMRI image to Patch Embedding for spatial dimensions only

    Args:
        img_size (tuple): Image size in (D, H, W, T).
        patch_size (tuple): Patch token size for (D, H, W).
        in_chans (int): Number of input image channels. Usually 1 for grayscale images.
        embed_dim (int): Number of linear projection output channels.
        norm_layer (nn.Module, optional): Normalization layer.
    """

    def __init__(self, img_size=(96, 96, 96, 20), patch_size=(4, 4, 4), in_chans=1, embed_dim=24, norm_layer=None):
        super().__init__()
        self.img_size = img_size  # Tuple of (D, H, W, T)
        self.patch_size = patch_size
        self.patches_resolution = [img_size[i] // patch_size[i] for i in range(3)]  # Only for D, H, W
        self.num_patches = np.prod(self.patches_resolution) * img_size[
            3]  # Total patches multiplied by number of time steps
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Using 3D convolutions to handle spatial dimensions
        self.spatial_proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, D, H, W, T = x.shape
        assert D == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2], \
            "Input spatial dimensions don't match model."
        assert T == self.img_size[3], "Input temporal dimension does not match."

        # Apply 3D Convolution to spatial dimensions
        x = x.view(B * T, C, D, H, W)  # Merge batch and time for independent spatial processing
        x = self.spatial_proj(x)  # Output shape: (B*T, embed_dim, D', H', W')
        x = x.view(B, T, self.embed_dim, -1)  # Reshape to separate batch and time: (B, T, embed_dim, D'*H'*W')
        x = x.permute(0, 2, 3, 1).contiguous()  # Reorder to (B, embed_dim, D'*H'*W', T)

        if self.norm:
            x = self.norm(x)

        return x

    def flops(self):
        Dp, Hp, Wp = (self.img_size[i] // self.patch_size[i] for i in range(3))
        # FLOPs for 3D convolution
        flops_conv3d = (self.patch_size[0] * self.patch_size[1] * self.patch_size[2] *
                        self.in_chans * self.embed_dim * Dp * Hp * Wp * self.img_size[3])
        return flops_conv3d


class CrossFormer(nn.Module):
    r""" Adapted CrossFormer for handling 4D fMRI data.

    Args:
        img_size (tuple[int]): Input image size (D, H, W, T).
        patch_size (tuple[int]): Patch size for (D, H, W).
        in_chans (int): Number of input image channels.
        num_classes (int): Number of classes for the classification head.
        embed_dim (int): Patch embedding dimension.
        depths (tuple[int]): Depth of each stage.
        num_heads (tuple[int]): Number of attention heads in different layers.
        group_size (tuple[int]): Group size for each stage.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate.
        drop_path_rate (float): Stochastic depth rate.
        norm_layer (nn.Module): Normalization layer.
        ape (bool): If True, add absolute position embedding to the patch embedding.
        patch_norm (bool): If True, add normalization after patch embedding.
        use_checkpoint (bool): Whether to use checkpointing to save memory.
        use_cpe (bool): Whether to use conditional positional encoding.
        pad_type (bool): 0 to pad in one direction, otherwise 1.
    """

    def __init__(self, img_size=(96, 96, 96, 20), patch_size=(4, 4, 4), in_chans=1, num_classes=2,
                 embed_dim=24, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 group_size=[7, 7, 7, 7], crs_interval=[8, 4, 2, 1], mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, merge_size=[[2], [2], [2]], use_cpe=False,
                 pad_type=0, no_mask=False,
                 adaptive_interval=False, use_acl=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # Patch Embedding adapted for 4D input
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        # Positional embedding, if used
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *img_size, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build stages
        self.layers = nn.ModuleList()
        previous_dim = embed_dim
        for i_layer, (layer_depth, nh, gs, ci) in enumerate(zip(depths, num_heads, group_size, crs_interval)):
            layer = Stage(
                dim=previous_dim,
                input_resolution=img_size,  # Pass the entire 4D resolution
                depth=layer_depth, num_heads=nh, group_size=gs, interval=ci,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)

        # Final normalization and classifier head
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        # Assume input x has shape (B, C, D, H, W, T) where C is channel, D, H, W are spatial dimensions and T is the time dimension
        x = self.patch_embed(x)  # Apply the patch embedding layer

        if self.ape:
            # Add position embedding, assuming absolute_pos_embed has been reshaped or broadcasted appropriately
            x = x + self.absolute_pos_embed

        x = self.pos_drop(x)  # Apply dropout to the positional embeddings

        for layer in self.layers:
            x = layer(x)  # Pass the output through each layer in the transformer

        # After processing through all layers, reshape x from (B, T, N, D) to (B, T * N, D) for pooling, assuming that T is the time dimension
        # This step flattens the time and sequence length dimensions into a single dimension to apply global pooling across all tokens and time steps
        B, T, N, D = x.shape
        x = x.view(B, T * N, D)  # Reshape for pooling

        x = self.norm(x)  # Apply normalization

        x = self.avgpool(
            x.transpose(1, 2))  # Apply adaptive average pooling across the combined sequence-time dimension
        x = x.flatten(1)  # Flatten to prepare for the classification head

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
