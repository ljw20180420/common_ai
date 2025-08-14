import os
import pickle
import torch
from torch import nn

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, einsum

from .utils import Residual


class ProteinBertCrossAttention(nn.Module):
    def __init__(
        self, dim_token: int, dim_global: int, heads: int, dim_head: int
    ) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_global = dim_global
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(
            nn.Linear(dim_global, heads * dim_head, bias=False),
            nn.Tanh(),
        )
        self.to_k = nn.Sequential(
            nn.Linear(dim_token, heads * dim_head, bias=False),
            nn.Tanh(),
        )
        self.to_v = nn.Sequential(
            nn.Linear(dim_token, dim_global, bias=False),
            nn.GELU(),
            Rearrange("b s (h d) -> b s h d", h=heads, d=dim_global // heads),
        )

    def forward(self, tokens: torch.Tensor, annotation: torch.Tensor) -> torch.Tensor:
        q = self.to_q(annotation)
        k = self.to_k(tokens)
        v = self.to_v(tokens)

        sim = (
            einsum(
                rearrange(
                    q, "b () (h d_h) -> b h d_h", h=self.heads, d_h=self.dim_head
                ),
                rearrange(
                    k, "b s (h d_h) -> b s h d_h", h=self.heads, d_h=self.dim_head
                ),
                "b h d_h, b s h d_h -> b s h",
            )
            * self.scale
        )
        attn = sim.softmax(dim=1)
        out = einsum(attn, v, "b s h, b s h d -> b h d")
        out = rearrange(out, "b h d -> b () (h d)", h=self.heads)
        return out


class ProteinBertLayer(nn.Module):
    def __init__(
        self,
        dim_token: int,
        dim_global: int,
        narrow_conv_kernel: int,
        wide_conv_kernel: int,
        wide_conv_dilation: int,
        attn_heads: int,
        attn_dim_head: int,
    ) -> None:
        super().__init__()

        self.narrow_conv = nn.Sequential(
            Rearrange("b s d -> b d s", d=dim_token),
            nn.Conv1d(
                dim_token,
                dim_token,
                narrow_conv_kernel,
                padding=narrow_conv_kernel // 2,
            ),
            Rearrange("b d s -> b s d", d=dim_token),
            nn.GELU(),
        )

        self.wide_conv = nn.Sequential(
            Rearrange("b s d -> b d s", d=dim_token),
            nn.Conv1d(
                dim_token,
                dim_token,
                wide_conv_kernel,
                dilation=wide_conv_dilation,
                padding=wide_conv_kernel // 2 * wide_conv_dilation,
            ),
            Rearrange("b d s -> b s d", d=dim_token),
            nn.GELU(),
        )

        self.extract_global_info = nn.Sequential(
            nn.Linear(dim_global, dim_token),
            nn.GELU(),
        )

        self.local_feedforward = nn.Sequential(
            nn.LayerNorm(dim_token, eps=1e-3),
            Residual(
                nn.Sequential(
                    nn.Linear(dim_token, dim_token),
                    nn.GELU(),
                )
            ),
            nn.LayerNorm(dim_token, eps=1e-3),
        )

        self.global_attend_local = ProteinBertCrossAttention(
            dim_token=dim_token,
            dim_global=dim_global,
            heads=attn_heads,
            dim_head=attn_dim_head,
        )

        self.global_dense = nn.Sequential(
            nn.Linear(dim_global, dim_global),
            nn.GELU(),
        )

        self.global_feedforward = nn.Sequential(
            nn.LayerNorm(dim_global, eps=1e-3),
            Residual(
                nn.Sequential(
                    nn.Linear(dim_global, dim_global),
                    nn.GELU(),
                )
            ),
            nn.LayerNorm(dim_global, eps=1e-3),
        )

    def forward(
        self, tokens: torch.Tensor, annotation: torch.Tensor
    ) -> tuple[torch.Tensor]:
        # process local (protein sequence)
        narrow_out = self.narrow_conv(tokens)
        wide_out = self.wide_conv(tokens)
        global_info = self.extract_global_info(annotation)

        tokens = tokens + narrow_out + wide_out + global_info
        tokens = self.local_feedforward(tokens)

        # process global (annotations)
        annotation = (
            annotation
            + self.global_dense(annotation)
            + self.global_attend_local(tokens, annotation)
        )
        annotation = self.global_feedforward(annotation)

        return tokens, annotation


class ProteinBertLayerCross(nn.Module):
    def __init__(
        self,
        dim_token: int,
        dim_global: int,
        narrow_conv_kernel: int,
        wide_conv_kernel: int,
        wide_conv_dilation: int,
        attn_heads: int,
        attn_dim_head: int,
    ) -> None:
        super().__init__()
        assert (
            dim_token % 2 == 0 and dim_global % 2 == 0
        ), "token and global dimension cannot be odd"

        self.ptpg = ProteinBertLayer(
            dim_token,
            dim_global,
            narrow_conv_kernel,
            wide_conv_kernel,
            wide_conv_dilation,
            attn_heads,
            attn_dim_head,
        )

        self.ptdg = ProteinBertLayer(
            dim_token,
            dim_global,
            narrow_conv_kernel,
            wide_conv_kernel,
            wide_conv_dilation,
            attn_heads,
            attn_dim_head,
        )

        self.dtpg = ProteinBertLayer(
            dim_token,
            dim_global,
            narrow_conv_kernel,
            wide_conv_kernel,
            wide_conv_dilation,
            attn_heads,
            attn_dim_head,
        )

        self.dtdg = ProteinBertLayer(
            dim_token,
            dim_global,
            narrow_conv_kernel,
            wide_conv_kernel,
            wide_conv_dilation,
            attn_heads,
            attn_dim_head,
        )

        self.protein_local_feedforward = nn.Sequential(
            nn.LayerNorm(2 * dim_token, eps=1e-3),
            Residual(
                nn.Sequential(
                    nn.Linear(2 * dim_token, dim_token),
                    nn.GELU(),
                )
            ),
            nn.LayerNorm(dim_token, eps=1e-3),
        )

        self.protein_global_feedforward = nn.Sequential(
            nn.LayerNorm(2 * dim_global, eps=1e-3),
            Residual(
                nn.Sequential(
                    nn.Linear(2 * dim_global, dim_global),
                    nn.GELU(),
                )
            ),
            nn.LayerNorm(dim_global, eps=1e-3),
        )

        self.DNA_local_feedforward = nn.Sequential(
            nn.LayerNorm(2 * dim_token, eps=1e-3),
            Residual(
                nn.Sequential(
                    nn.Linear(2 * dim_token, dim_token),
                    nn.GELU(),
                )
            ),
            nn.LayerNorm(dim_token, eps=1e-3),
        )

        self.DNA_global_feedforward = nn.Sequential(
            nn.LayerNorm(2 * dim_global, eps=1e-3),
            Residual(
                nn.Sequential(
                    nn.Linear(2 * dim_global, dim_global),
                    nn.GELU(),
                )
            ),
            nn.LayerNorm(dim_global, eps=1e-3),
        )

    def forward(
        self,
        protein_tokens: torch.Tensor,
        protein_annotation: torch.Tensor,
        DNA_tokens: torch.Tensor,
        DNA_annotation: torch.Tensor,
    ):
        protein_tokens_ptpg, protein_annotation_ptpg = self.ptpg(
            protein_tokens, protein_annotation
        )
        protein_tokens_ptdg, DNA_annotation_ptdg = self.ptdg(
            protein_tokens, DNA_annotation
        )
        DNA_tokens_dtpg, protein_annotation_dtpg = self.dtpg(
            DNA_tokens, protein_annotation
        )
        DNA_tokens_dtdg, DNA_annotation_dtdg = self.dtdg(DNA_tokens, DNA_annotation)
        protein_tokens = self.protein_local_feedforward(
            torch.cat(
                (
                    protein_tokens_ptpg,
                    protein_tokens_ptdg,
                )
            ),
            dim=2,
        )
        protein_annotation = self.protein_global_feedforward(
            torch.cat(
                (
                    protein_annotation_ptpg,
                    protein_annotation_dtpg,
                )
            ),
            dim=2,
        )
        DNA_tokens = self.DNA_local_feedforward(
            torch.cat(
                (
                    DNA_tokens_dtpg,
                    DNA_tokens_dtdg,
                )
            ),
            dim=2,
        )
        DNA_annotation = self.DNA_global_feedforward(
            torch.cat(
                (
                    DNA_annotation_ptdg,
                    DNA_annotation_dtdg,
                )
            ),
            dim=2,
        )
        return protein_tokens, protein_annotation, DNA_tokens, DNA_annotation


class ProteinBert(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        dim_token: int,
        dim_global: int,
        depth: int,
        narrow_conv_kernel: int,
        wide_conv_kernel: int,
        wide_conv_dilation: int,
        attn_heads: int,
        attn_dim_head: int,
    ) -> None:
        super().__init__()
        self.dim_global = dim_global
        self.token_emb = nn.Embedding(num_tokens, dim_token)
        self.global_bias = nn.Parameter(torch.zeros(dim_global))
        self.active_global = nn.GELU()

        self.layers = nn.ModuleList(
            [
                ProteinBertLayer(
                    dim_token=dim_token,
                    dim_global=dim_global,
                    narrow_conv_kernel=narrow_conv_kernel,
                    wide_conv_dilation=wide_conv_dilation,
                    wide_conv_kernel=wide_conv_kernel,
                    attn_heads=attn_heads,
                    attn_dim_head=attn_dim_head,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, id: torch.Tensor) -> tuple[torch.Tensor]:
        batch_size = id.shape[0]

        tokens = self.token_emb(id)

        annotation = repeat(
            self.active_global(self.global_bias),
            "d -> b () d",
            b=batch_size,
            d=self.dim_global,
        )

        for layer in self.layers:
            tokens, annotation = layer(tokens, annotation)

        return tokens, annotation

    def load_pretrain_weights(self, weights_file: os.PathLike):
        with open(weights_file, "rb") as fd:
            _, model_weights, _ = pickle.load(fd)
        self.global_bias.data = torch.from_numpy(model_weights[1])
        self.token_emb.weight.data = torch.from_numpy(model_weights[2])

        for i, layer in enumerate(self.layers):
            # torch Linear weight is (out_feature, in_feature)
            # tensorflow Linear weight is (in_feature, out_feature)
            # EinMix weight is (in_feature, out_feature), the same as tensorflow
            # EinMix bias is (1, ..., 1, out_feature)
            layer.extract_global_info[0].weight.data = rearrange(
                torch.from_numpy(model_weights[i * 23 + 3]), "in out -> out in"
            )
            layer.extract_global_info[0].bias.data = torch.from_numpy(
                model_weights[i * 23 + 4]
            )
            # torch Conv weight is (out_channel, in_channel, kernel_dim1, kernel_dim2, ...)
            # tensorflow Conv weight is (kernel_dim1, kernel_dim2, ..., in_channel, out_channel)
            layer.narrow_conv[1].weight.data = torch.from_numpy(
                model_weights[i * 23 + 5]
            ).permute(2, 1, 0)
            layer.narrow_conv[1].bias.data = torch.from_numpy(model_weights[i * 23 + 6])
            layer.wide_conv[1].weight.data = torch.from_numpy(
                model_weights[i * 23 + 7]
            ).permute(2, 1, 0)
            layer.wide_conv[1].bias.data = torch.from_numpy(model_weights[i * 23 + 8])
            layer.local_feedforward[0].weight.data = torch.from_numpy(
                model_weights[i * 23 + 9]
            )
            layer.local_feedforward[0].bias.data = torch.from_numpy(
                model_weights[i * 23 + 10]
            )
            layer.local_feedforward[1].module[0].weight.data = rearrange(
                torch.from_numpy(model_weights[i * 23 + 11]), "in out -> out in"
            )
            layer.local_feedforward[1].module[0].bias.data = torch.from_numpy(
                model_weights[i * 23 + 12]
            )
            layer.local_feedforward[2].weight.data = torch.from_numpy(
                model_weights[i * 23 + 13]
            )
            layer.local_feedforward[2].bias.data = torch.from_numpy(
                model_weights[i * 23 + 14]
            )
            layer.global_dense[0].weight.data = rearrange(
                torch.from_numpy(model_weights[i * 23 + 15]), "in out -> out in"
            )
            layer.global_dense[0].bias.data = torch.from_numpy(
                model_weights[i * 23 + 16]
            )
            layer.global_attend_local.to_q[0].weight.data = rearrange(
                torch.from_numpy(model_weights[i * 23 + 17]),
                "n d hd -> (n hd) d",
            )
            layer.global_attend_local.to_k[0].weight.data = rearrange(
                torch.from_numpy(model_weights[i * 23 + 18]),
                "n d hd -> (n hd) d",
            )
            layer.global_attend_local.to_v[0].weight.data = rearrange(
                torch.from_numpy(model_weights[i * 23 + 19]),
                "n d hd -> (n hd) d",
            )
            layer.global_feedforward[0].weight.data = torch.from_numpy(
                model_weights[i * 23 + 20]
            )
            layer.global_feedforward[0].bias.data = torch.from_numpy(
                model_weights[i * 23 + 21]
            )
            layer.global_feedforward[1].module[0].weight.data = rearrange(
                torch.from_numpy(model_weights[i * 23 + 22]), "in out -> out in"
            )
            layer.global_feedforward[1].module[0].bias.data = torch.from_numpy(
                model_weights[i * 23 + 23]
            )
            layer.global_feedforward[2].weight.data = torch.from_numpy(
                model_weights[i * 23 + 24]
            )
            layer.global_feedforward[2].bias.data = torch.from_numpy(
                model_weights[i * 23 + 25]
            )
