import torch
import torch.nn as nn

# 適当なテンソルを定義（N=1, C=2, H=2, W=2）
x = torch.tensor(
    [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], dtype=torch.float32
)

print("入力テンソル:\n", x)

# GroupNorm を定義（2チャンネルを 2グループに分ける）
gn = nn.GroupNorm(num_groups=2, num_channels=2, eps=1e-5, affine=False)

# GroupNorm 出力を得る
output = gn(x)
print("\nPyTorch GroupNorm 出力:\n", output)


# 手動でGroupNormを計算
def manual_groupnorm(x, num_groups, eps=1e-5):
    N, C, H, W = x.shape
    G = num_groups
    x = x.view(N, G, -1)  # (N, G, C*H*W // G)

    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    x_norm = x_norm.view(N, C, H, W)

    print("\n手動計算:")
    print("Mean:\n", mean)
    print("Variance:\n", var)
    print("Normalized Output:\n", x_norm)
    return x_norm


manual_output = manual_groupnorm(x, num_groups=2)

# PyTorch の結果との比較（ほぼ同じはず）
print("\n差分 (PyTorch - 手動):\n", output - manual_output)
