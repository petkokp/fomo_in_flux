import torch
import torch.nn as nn
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

    # ---------- GPU/Tensor augmentation (no pre-resize, no clamp) ----------
    @torch.no_grad()
    def _augment_views_gpu_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """
        Input `images`: [B,C,H,W], normalized with self.mean/std.
        Returns [B*(1+num_augs), C, 224, 224], normalized.
        """
        device = images.device
        B, C, H, W = images.shape

        # De-normalize to linear space in [possibly <0, >1]; no clamp
        imgs_denorm = images * self.std.view(1, -1, 1, 1) + self.mean.view(1, -1, 1, 1)

        # Create an 'original' 224 view via center-crop-from-resize (to match zeroshot)
        # If already 224, this is a no-op except re-normalization.
        orig_224 = Fv2.resize(imgs_denorm, size=[256, 256], antialias=True)
        orig_224 = Fv2.center_crop(orig_224, output_size=[224, 224])
        orig_224 = (orig_224 - self.mean.view(1, -1, 1, 1)) / self.std.view(1, -1, 1, 1)

        views = [orig_224]  # [B,C,224,224]

        # Prepare an RRC param sampler (we'll apply per-image)
        rrc = v2.RandomResizedCrop(size=(224, 224), scale=self.crop_scale, ratio=self.crop_ratio, antialias=True)

        for _ in range(self.num_augs):
            crops = []
            for i in range(B):
                # Sample params on the ORIGINAL image (no pre-resize)
                i0, j0, h0, w0 = rrc.get_params(imgs_denorm[i], scale=self.crop_scale, ratio=self.crop_ratio)
                ci = Fv2.resized_crop(imgs_denorm[i], i0, j0, h0, w0, size=[224, 224], antialias=True)
                # Random horizontal flip
                if torch.rand((), device=device) < 0.5:
                    ci = Fv2.horizontal_flip(ci)
                crops.append(ci.unsqueeze(0))
            crops = torch.cat(crops, dim=0)  # [B,C,224,224]
            crops = (crops - self.mean.view(1, -1, 1, 1)) / self.std.view(1, -1, 1, 1)
            views.append(crops)

        all_views = torch.cat(views, dim=0).to(device, non_blocking=True)  # [B*(1+num_augs), C, 224, 224]
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

        # 1) Build views
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
        #    (more stable than applying logit_scale for ranking)
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
