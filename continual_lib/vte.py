import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
import torchvision.transforms.v2.functional as Fv2
from continual_lib.utils.base_continual_learner import BaseContinualLearner


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = logits.softmax(dim=-1)
    return -(p * (p.log() + 1e-12)).sum(dim=-1)


class Model(BaseContinualLearner):
    REQ_NON_AUG_INPUTS = False

    def __init__(
        self,
        args, backbone, head, loss, device,
        selection_p: float = 0.10,
        num_augs: int = 63,                 # 1 + 63 = 64 views
        crop_scale=(0.08, 1.0),             # wider scale helps ImageNet-R
        crop_ratio=(3/4, 4/3),
        microbatch: int | None = None,
        seed: int | None = 123,              # fix seed for eval runs
        **kwargs,
    ):
        super().__init__(args, backbone, head, loss, device)
        self.selection_p = selection_p
        self.num_augs = num_augs
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        self.microbatch = microbatch

        if seed is not None:
            torch.manual_seed(seed)

        self.backbone.requires_grad_(False).eval()
        self.head.requires_grad_(False).eval()

        mean = torch.tensor(list(map(float, self.backbone.module.mean)), device=device)
        std  = torch.tensor(list(map(float, self.backbone.module.std)),  device=device)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std",  std,  persistent=False)

        self.text_features = None
        self._cached_texts_key = None
        with torch.no_grad():
            # cache but DON'T use it for view ranking
            self.logit_scale = self.head.module.text_encoder.logit_scale.exp().to(device)

    # ---------- GPU/Tensor augmentation - OPTIMIZED PARALLEL VERSION ----------
    @torch.no_grad()
    def _augment_views_gpu_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """
        Input `images`: [B,C,H,W], normalized with self.mean/std.
        Returns [B*(1+num_augs), C, 224, 224], normalized.
        Fully parallelized on GPU - no Python loops over batch or augmentations.
        """
        device = images.device
        B, C, H, W = images.shape

        # De-normalize to linear space
        imgs_denorm = images * self.std.view(1, -1, 1, 1) + self.mean.view(1, -1, 1, 1)

        # Create an 'original' 224 view via center-crop-from-resize
        orig_224 = Fv2.resize(imgs_denorm, size=[256, 256], antialias=True)
        orig_224 = Fv2.center_crop(orig_224, output_size=[224, 224])
        orig_224 = (orig_224 - self.mean.view(1, -1, 1, 1)) / self.std.view(1, -1, 1, 1)

        if self.num_augs == 0:
            return orig_224

        # Repeat images for all augmentations: [B*num_augs, C, H, W]
        imgs_repeated = imgs_denorm.unsqueeze(1).repeat(1, self.num_augs, 1, 1, 1).view(B * self.num_augs, C, H, W)
        
        # Sample all RRC parameters in parallel
        N = B * self.num_augs
        scale_min, scale_max = self.crop_scale
        ratio_min, ratio_max = self.crop_ratio
        
        # Sample random scales and aspect ratios
        scales = torch.empty(N, device=device).uniform_(scale_min, scale_max)
        log_ratio = torch.empty(N, device=device).uniform_(
            torch.log(torch.tensor(ratio_min)), 
            torch.log(torch.tensor(ratio_max))
        )
        ratios = torch.exp(log_ratio)
        
        # Compute crop dimensions
        areas = H * W * scales
        crop_h = torch.sqrt(areas / ratios).clamp(1, H).long()
        crop_w = torch.sqrt(areas * ratios).clamp(1, W).long()
        
        # Sample random crop positions
        i = (torch.rand(N, device=device) * (H - crop_h + 1).float()).long()
        j = (torch.rand(N, device=device) * (W - crop_w + 1).float()).long()
        
        # Create normalized grid coordinates for grid_sample - FULLY VECTORIZED
        # Calculate affine transformation matrices for all crops in parallel
        # Normalized coordinates in [-1, 1]
        y_start = 2 * i.float() / (H - 1) - 1
        y_end = 2 * (i + crop_h - 1).float() / (H - 1) - 1
        x_start = 2 * j.float() / (W - 1) - 1
        x_end = 2 * (j + crop_w - 1).float() / (W - 1) - 1
        
        # Build theta matrices in parallel [N, 2, 3]
        theta = torch.zeros(N, 2, 3, device=device)
        theta[:, 0, 0] = (x_end - x_start) / 2
        theta[:, 0, 2] = (x_end + x_start) / 2
        theta[:, 1, 1] = (y_end - y_start) / 2
        theta[:, 1, 2] = (y_end + y_start) / 2
        
        # Generate sampling grid and extract crops
        grid = F.affine_grid(theta, [N, C, 224, 224], align_corners=True)
        crops = F.grid_sample(imgs_repeated, grid, mode='bilinear', 
                             padding_mode='border', align_corners=True)
        
        # Apply horizontal flips in parallel
        flip_mask = torch.rand(N, device=device) < 0.5
        crops = torch.where(
            flip_mask.view(-1, 1, 1, 1),
            torch.flip(crops, dims=[3]),  # flip width dimension
            crops
        )
        
        # Normalize all crops at once
        crops = (crops - self.mean.view(1, -1, 1, 1)) / self.std.view(1, -1, 1, 1)
        
        # Combine original and augmented views: [B*(1+num_augs), C, 224, 224]
        all_views = torch.cat([orig_224, crops], dim=0)
        return all_views

    # ---------- text cache ----------
    def _maybe_build_text_features(self, texts, batch_size: int):
        key = tuple(texts) if isinstance(texts, (list, tuple)) else texts
        if (self.text_features is None) or (key != self._cached_texts_key):
            with torch.no_grad():
                tf = self.head.module.embed_text(texts, batch_size).to(self.device)
                tf = tf / (tf.norm(dim=-1, keepdim=True) + 1e-12)
            self.text_features = tf
            self._cached_texts_key = key

    @torch.no_grad()
    def forward(self, images, **kwargs):
        """
        Returns ensembled, L2-normalized image features: [B, D].
        kwargs['texts'] must match your zeroshot setup (same templates & order).
        """
        self.backbone.eval()
        self.head.eval()

        texts = kwargs['texts']
        self._maybe_build_text_features(texts, self.args.experiment.evaluation.batch_size)
        text_features = self.text_features                 # [K,D]

        B = images.shape[0]
        V = 1 + self.num_augs

        # 1) Build views (fully parallelized)
        all_aug_images = self._augment_views_gpu_tensor(images)   # [B*V,C,224,224]

        # 2) Encode (micro-batch if needed)
        def _encode_in_chunks(x, chunk):
            if not chunk or x.shape[0] <= chunk:
                return self.backbone(x)
            parts = [self.backbone(x[i:i+chunk]) for i in range(0, x.shape[0], chunk)]
            return torch.cat(parts, dim=0)

        with torch.cuda.amp.autocast():
            img_features = _encode_in_chunks(all_aug_images, self.microbatch)  # [B*V,D]

        img_features = img_features / (img_features.norm(dim=-1, keepdim=True) + 1e-12)

        # 3) View selection — rank by entropy of *unscaled* cosine logits
        cosine_logits = img_features @ text_features.T            # [B*V,K]
        logits = cosine_logits.view(B, V, -1)
        view_entropy = entropy_from_logits(logits)                 # [B,V]

        num_select = max(1, int(V * self.selection_p))             # default: 6/64
        top_idx = torch.argsort(view_entropy, dim=1, descending=False)[:, :num_select]

        # 4) Ensemble selected features
        img_features = img_features.view(B, V, -1)                 # [B,V,D]
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, img_features.size(-1))
        selected = torch.gather(img_features, dim=1, index=gather_idx)  # [B,S,D]
        ensembled = selected.mean(dim=1)                           # [B,D]
        ensembled = ensembled / (ensembled.norm(dim=-1, keepdim=True) + 1e-12)
        return ensembled