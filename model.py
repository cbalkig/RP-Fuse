import torch
import torch.nn as nn
import timm
import os


class SliceAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        attn_scores = self.attn(x)
        weights = torch.softmax(attn_scores, dim=1)
        return torch.sum(x * weights, dim=1)


class PredictiveResNet(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=2, dropout_rate=0.5,
                 medical_weights_path=None, in_chans=1):
        super(PredictiveResNet, self).__init__()
        self.in_chans = in_chans

        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            in_chans=in_chans,
            features_only=True
        )

        if medical_weights_path and os.path.exists(medical_weights_path):
            self._load_medical_weights(medical_weights_path)

        with torch.no_grad():
            dummy = torch.randn(1, in_chans, 128, 128)
            feats = self.backbone(dummy)
            self.feat_dim = feats[-1].shape[1]

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.aggregator = SliceAttention(self.feat_dim)

        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

    def _load_medical_weights(self, path):
        print(f"[Model] Loading ResNet Weights: {path}")
        state_dict = torch.load(path, map_location='cpu')

        conv1_key = 'conv1.weight'

        if conv1_key in state_dict:
            weight = state_dict[conv1_key]
            if weight.shape[1] == 3:
                state_dict[conv1_key] = weight.sum(dim=1, keepdim=True)
                print("[Model] Adapted conv1 to 1-channel.")

        self.backbone.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        if x.dim() == 5:
            b, c, d, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)

            feats = self.backbone(x)[-1]

            feats = self.global_pool(feats).flatten(1)

            feats = feats.view(b, d, -1)

            agg_feats = self.aggregator(feats)
        else:
            feats = self.backbone(x)[-1]
            feats = self.global_pool(feats).flatten(1)
            agg_feats = self.aggregator(feats.unsqueeze(1))

        return self.fc(agg_feats)


def create_Model_3D(model_config=None):
    dropout = 0.5
    weights_path = None
    model_name = 'resnet50'

    if isinstance(model_config, dict):
        model_name = model_config.get("BACKBONE", 'resnet50')
        dropout = model_config.get("DROPOUT_RATE", 0.5)
        weights_path = model_config.get("WEIGHTS_PATH", None)

    return PredictiveResNet(
        model_name=model_name,
        dropout_rate=dropout,
        medical_weights_path=weights_path
    )
