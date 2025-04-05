from mmengine.model import BaseModule
from mmengine.registry import MODELS

from mmdet.models import Mask2FormerTransformerDecoder
from mmdet.utils import ConfigType
from mmengine.model import normal_init, xavier_init

import re
from torch import nn
import torch

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
import torch.nn.functional as F


@MODELS.register_module()
class IpslVisionTower(BaseModule):
    def __init__(self, model_path, vision_select_layer=-2, vision_select_feature="patch", **kwargs):
        super().__init__()

        self.model_path = model_path
        self.select_layer = vision_select_layer
        self.select_feature = vision_select_feature

        self.image_processor = CLIPImageProcessor.from_pretrained(self.model_path)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.model_path)

        self.vision_tower.requires_grad_(False)

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls':
            image_features = image_features[:, 0]
        elif self.select_feature == 'all':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        images = F.interpolate(images, size=(336, 336), mode='bilinear', align_corners=False)
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                               output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


@MODELS.register_module()
class IpslSemanticProjector(BaseModule):
    def __init__(self, projector_type, vision_hidden_size, llm_hidden_size, model_path, **kwargs):
        super().__init__()
        if projector_type == 'linear':
            self.projector = nn.Linear(vision_hidden_size, llm_hidden_size)

        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(vision_hidden_size, llm_hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(llm_hidden_size, llm_hidden_size))
            self.projector = nn.Sequential(*modules)

        print(f'Loding pretrained Semtantic projector from {model_path}.')
        w = torch.load(model_path)
        w = {k.split('mm_projector.')[-1]: v for k, v in w.items()}
        self.projector.load_state_dict(w, strict=True)

    def forward(self, img_emds):
        return self.projector(img_emds)

    def init_weights(self):
        xavier_init(self.projector)


@MODELS.register_module()
class IpslModule(BaseModule):
    def __init__(
            self,
            vision_hidden_size,
            llm_hidden_size,
            llm_vocab_size,
            transformer_decoder_hidden_size,
            vision_tower_cfg: ConfigType = dict(
                model_path=None,
                vision_select_layer=-2,
                vision_select_feature="patch", ),
            semantic_projector_cfg: ConfigType = dict(
                projector_type="linear"
            ),
            transformer_decoder: ConfigType = ...,
            **kwargs
    ):
        super().__init__()
        # self.llm_emds = nn.Embedding(
        #     num_embeddings=llm_vocab_size,
        #     embedding_dim=llm_hidden_size,
        # )
        self.vision_tower = MODELS.build(vision_tower_cfg)
        semantic_projector_cfg.update(
            vision_hidden_size=vision_hidden_size,
            llm_hidden_size=llm_hidden_size,
        )
        self.semantic_projector = MODELS.build(semantic_projector_cfg)
        self.transformer_decoder = Mask2FormerTransformerDecoder(
            **transformer_decoder)
        self.transformer_decoder_projector = nn.Sequential(
            nn.Linear(llm_hidden_size, transformer_decoder_hidden_size),
            nn.GELU(),
            nn.Linear(transformer_decoder_hidden_size, transformer_decoder_hidden_size),
        )

    def forward(self, img_tensors, object_queries):
        """
        Args:
            img_tensors: [B, C, H, W]
        Returns:
        """
        # print(img_tensors.shape, object_queries.shape)
        img_emds = self.vision_tower(img_tensors)
        # print(img_emds.shape)
        semantic_img_emds = self.semantic_projector(img_emds)
        # print(semantic_img_emds.shape)
        semantic_img_emds = self.transformer_decoder_projector(semantic_img_emds)
        # print(semantic_img_emds.shape)
        for _ in self.transformer_decoder.layers:
            object_queries = self.transformer_decoder(
                query=object_queries,
                key=semantic_img_emds,
                value=semantic_img_emds,
                query_pos=None,
                key_pos=None,
                key_padding_mask=None,
            )
        # print(object_queries.shape)
        # exit(0)
        return object_queries[0]

    def init_weights(self):
        xavier_init(self.transformer_decoder_projector)
