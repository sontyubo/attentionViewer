import torch
import torch.nn as nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helper functions
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    # heads, dim_heads = Multi-Head Self-Attentionのパラメータ
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        # 標準化
        # LayerNorm = (x_i - μ) / (σ + eps) * gamma + beta
        self.norm = nn.LayerNorm(dim)

        # 関数を代入
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)  # dropoutの確率でドロップアウト

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)  # qkv = (q, k, v)
        # Q/K/Vをヘッド数に分割
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # QとKの内積

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    """
    :param dim: int: トークンの次元数
    :param depth: int: トランスフォーマーのレイヤー数
    :param heads: int: ヘッドの数
    :param dim_head: int: 1つのヘッドの次元数
    :param mlp_dim: int: MLP層の中間ユニット数
    :param dropout: float: ドロップアウト率
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])  # レイヤーをリストで保持
        for _ in range(depth):  # depth回各レイヤーを追加
            """
            [Attention, FeedForward]
            [Attention, FeedForward]
            [Attention, FeedForward]
            ...
            （depth 回繰り返し）
            """
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for layer in self.layers:
            attn, ff = layer
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class VIT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropput=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        image_height, image_wirdh = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_wirdh % patch_width == 0, (
            "image dimensions must be divisible by the patch size"
        )

        num_patches = (image_height // patch_height) * (image_wirdh // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {"cls", "mean"}, (
            "pool type must be either cls (cls token) or mean (mean pooling)"
        )

        self.to_patch_embedding = nn.Sequential(
            # 画像を小さなパッチに切り出し、それぞれを1つのベクトル（flattenされたパッチ）」
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            # Transformerに渡せるように特徴ベクトルの次元を変換
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 画像の代表ベクトル
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropput)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
