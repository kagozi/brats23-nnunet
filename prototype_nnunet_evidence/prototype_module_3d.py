
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeMemory3D(nn.Module):
    def __init__(self, feature_dim, num_classes, prototypes_per_class):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.prototypes_per_class = prototypes_per_class
        self.num_prototypes = num_classes * prototypes_per_class

        self.prototypes = nn.Parameter(
            torch.randn(self.num_prototypes, feature_dim)
        )

        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, x):
        b, c, d, h, w = x.shape

        x_flat = x.permute(0, 2, 3, 4, 1).reshape(b, d * h * w, c)

        x_norm = F.normalize(x_flat, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)

        similarity = torch.matmul(x_norm, p_norm.t())

        similarity = similarity.view(
            b,
            d,
            h,
            w,
            self.num_classes,
            self.prototypes_per_class,
        )

        similarity = similarity.permute(0, 4, 5, 1, 2, 3).contiguous()

        class_evidence = similarity.max(dim=2).values
        top_proto_indices = similarity.view(
            b,
            self.num_classes * self.prototypes_per_class,
            d,
            h,
            w,
        ).argmax(dim=1)

        return {
            "proto_similarity": similarity,
            "class_evidence": class_evidence,
            "top_proto_indices": top_proto_indices,
            "prototypes": self.prototypes,
        }


class PrototypeCrossAttention3D(nn.Module):
    def __init__(self, feature_dim, num_classes, prototypes_per_class, num_heads=4):
        super().__init__()

        self.feature_dim = feature_dim

        self.memory = PrototypeMemory3D(
            feature_dim=feature_dim,
            num_classes=num_classes,
            prototypes_per_class=prototypes_per_class,
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        b, c, d, h, w = x.shape

        proto_out = self.memory(x)
        prototypes = proto_out["prototypes"]

        query = x.permute(0, 2, 3, 4, 1).reshape(b, d * h * w, c)

        proto_tokens = prototypes.unsqueeze(0).expand(b, -1, -1)

        attended, _ = self.attn(
            query=query,
            key=proto_tokens,
            value=proto_tokens,
        )

        attended = self.norm(attended + query)

        attended = attended.view(b, d, h, w, c)
        attended = attended.permute(0, 4, 1, 2, 3).contiguous()

        proto_out["proto_features"] = attended

        return proto_out


class PrototypeFusionBlock3D(nn.Module):
    def __init__(self, feature_dim, num_classes, prototypes_per_class, num_heads=4):
        super().__init__()

        self.prototype_attention = PrototypeCrossAttention3D(
            feature_dim=feature_dim,
            num_classes=num_classes,
            prototypes_per_class=prototypes_per_class,
            num_heads=num_heads,
        )

        self.gate = nn.Sequential(
            nn.Conv3d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.InstanceNorm3d(feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        proto_out = self.prototype_attention(x)
        proto_features = proto_out["proto_features"]

        gate = self.gate(torch.cat([x, proto_features], dim=1))

        fused = gate * proto_features + (1.0 - gate) * x

        proto_out["fused_features"] = fused
        proto_out["gate"] = gate

        return proto_out
