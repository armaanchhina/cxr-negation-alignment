from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoModel


class ProjectionHead(nn.Module):
    """
    Two-layer projection head used to map encoder features into the shared
    contrastive embedding space.
    """

    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super().__init__()
        hidden_dim = embedding_dim * 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class MultimodalCXRModel(nn.Module):
    """
    Multimodal chest X-ray retrieval model with:
    - Bio_ClinicalBERT text encoder
    - DenseNet-121 image encoder
    - learned projection heads for each modality
    - learned temperature for contrastive training
    """

    def __init__(
        self,
        text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        embedding_dim: int = 512,
        image_encoder_name: str = "densenet121",
        pretrained_image_encoder: bool = True,
    ) -> None:
        super().__init__()

        if image_encoder_name != "densenet121":
            raise ValueError(f"Unsupported image encoder: {image_encoder_name}")

        self.text_model_name = text_model_name
        self.embedding_dim = embedding_dim
        self.image_encoder_name = image_encoder_name

        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_hidden_dim = self.text_encoder.config.hidden_size

        if pretrained_image_encoder:
            weights = models.DenseNet121_Weights.DEFAULT
        else:
            weights = None

        image_backbone = models.densenet121(weights=weights)
        image_hidden_dim = image_backbone.classifier.in_features
        image_backbone.classifier = nn.Identity()
        self.image_encoder = image_backbone

        self.text_projection = ProjectionHead(
            input_dim=text_hidden_dim,
            embedding_dim=embedding_dim,
        )
        self.image_projection = ProjectionHead(
            input_dim=image_hidden_dim,
            embedding_dim=embedding_dim,
        )

        initial_temperature = 0.07
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / initial_temperature)))

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_features = outputs.last_hidden_state[:, 0, :]
        projected = self.text_projection(cls_features)
        return F.normalize(projected, p=2, dim=1)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image)
        projected = self.image_projection(image_features)
        return F.normalize(projected, p=2, dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_embeddings = self.encode_image(image)
        text_embeddings = self.encode_text(input_ids, attention_mask)
        return image_embeddings, text_embeddings