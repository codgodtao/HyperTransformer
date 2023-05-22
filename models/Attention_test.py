from typing import Union

from mmcv.ops import MultiScaleDeformableAttention
# This function implements the multi-head attention
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor


def get_encoder_reference_points(
        spatial_shapes: Tensor, valid_ratios: Tensor,device: Union[torch.device, str]) -> Tensor:
    """Get the reference points used in encoder.

    Args:
        spatial_shapes (Tensor): Spatial shapes of features in all levels,
            has shape (num_levels, 2), last dimension represents (h, w).
        valid_ratios (Tensor): The ratios of the valid width and the valid
            height relative to the width and the height of features in all
            levels, has shape (bs, num_levels, 2).
        device (obj:`device` or str): The device acquired by the
            `reference_points`.

    Returns:
        Tensor: Reference points used in decoder, has shape (bs, length,
        num_levels, 2).
    """

    reference_points_list = []
    for lvl, (H, W) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H - 0.5, H, dtype=torch.float32,device=device),
            torch.linspace(
                0.5, W - 0.5, W, dtype=torch.float32,device=device))
        ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H)
        ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W)
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    # [bs, sum(hw), num_level, 2]
    reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    return reference_points

class NoAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self):
        super().__init__()

    def forward(self, v, k, q, mask=None):
        output = v
        return output


class ScaledDotProductAttentionOnly(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, v, k, q, mask=None):
        b, c, h, w = q.size(0), q.size(1), q.size(2), q.size(3)

        # Reshaping K,Q, and Vs...
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)

        # Compute attention
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # Normalization (SoftMax)
        attn = F.softmax(attn, dim=-1)

        # Attention output
        output = torch.matmul(attn, v)

        # Reshape output to original format
        output = output.view(b, c, h, w)
        return output


# This function implements the multi-head attention
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, v, k, q, mask=None):
        # Compute attention
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # Normalization (SoftMax)
        attn = F.softmax(attn, dim=-1)

        # Attention output
        output = torch.matmul(attn, v)
        return output




class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module for Hyperspectral Pansharpening (Image Fusion) '''

    def __init__(self, n_head, in_pixels, linear_dim, num_features):
        super().__init__()
        # Parameters
        self.n_head = n_head  # No of heads
        self.in_pixels = in_pixels  # No of pixels in the input image
        self.linear_dim = linear_dim  # Dim of linear-layer (outputs)

        # Linear layers

        self.w_qs = nn.Linear(in_pixels, n_head * linear_dim, bias=False)  # Linear layer for queries
        self.w_ks = nn.Linear(in_pixels, n_head * linear_dim, bias=False)  # Linear layer for keys
        self.w_vs = nn.Linear(in_pixels, n_head * linear_dim, bias=False)  # Linear layer for values
        self.fc = nn.Linear(n_head * linear_dim, in_pixels, bias=False)  # Final fully connected layer

        # Scaled dot product attention
        self.attention = ScaledDotProductAttention(temperature=in_pixels ** 0.5)

        # Batch normalization layer
        self.OutBN = nn.BatchNorm2d(num_features=num_features)

    def forward(self, v, k, q, mask=None):
        # Reshaping matrixes to 2D
        # q = b, c_q, h*w
        # k = b, c_k, h*w
        # v = b, c_v, h*w
        b, c, h, w = q.size(0), q.size(1), q.size(2), q.size(3)
        n_head = self.n_head
        linear_dim = self.linear_dim

        # Reshaping K, Q, and Vs...
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)

        # Save V
        output = v

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(b, c, n_head, linear_dim)
        k = self.w_ks(k).view(b, c, n_head, linear_dim)
        v = self.w_vs(v).view(b, c, n_head, linear_dim)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        # Computing ScaledDotProduct attention for each head
        v_attn = self.attention(v, k, q, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        v_attn = v_attn.transpose(1, 2).contiguous().view(b, c, n_head * linear_dim)
        v_attn = self.fc(v_attn)

        output = output + v_attn
        # output  = v_attn

        # Reshape output to original image format
        output = output.view(b, c, h, w)

        # We can consider batch-normalization here,,,
        # Will complete it later
        output = self.OutBN(output)
        return output






if __name__ == '__main__':
    lv1_dim = 16 ** 2
    lv2_dim = (2 * 16) ** 2
    lv3_dim = (4 * 16) ** 2

    scaleddotattention = ScaledDotProductAttentionOnly(lv1_dim)
    v, k, q = torch.randn([2,256,32,32]),torch.randn([2,256,32,32]),torch.randn([2,256,32,32])
    result = scaleddotattention(v,k,q)
    print(result.shape)
    q = q.view(2,32 * 32,  256)
    k = k.view(2,32 * 32, 256)
    v = v.view(2,32 * 32,  256)
    attention2 = MultiScaleDeformableAttention(num_levels=1,num_heads=8,batch_first=True)
    #(bs, num_levels, 2)
    reference_point = get_encoder_reference_points(torch.tensor([[32,32]]),torch.ones([2,1,2]))

    #(bs, num_query, num_levels, 2)
    result2=attention2(q,k,v,spatial_shapes=torch.tensor([[32,32]]),level_start_index=torch.tensor([0,32*32]),
                       reference_points=reference_point)
    print(result.shape,result2.shape)

