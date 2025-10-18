# batclip.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from continual_lib.utils.base_continual_learner import BaseContinualLearner


class Model(BaseContinualLearner):
    """
    BATCLIP: Bimodal Online Test-Time Adaptation for CLIP
    
    Performs adaptation during forward pass (test-time), not during observe (training).
    """
    
    REQ_NON_AUG_INPUTS = False
    
    def __init__(self, args, backbone, head, loss, device, **kwargs):
        super(Model, self).__init__(args, backbone, head, loss, device)
        
        # Cache for text features
        self.text_features = None
        self._cached_texts_key = None
        
        # Freeze all parameters initially
        self.backbone.requires_grad_(False)
        self.head.requires_grad_(False)
        
        # Set up LayerNorm parameters to optimize
        self._set_optim_params()
        
        # Initialize optimizer for test-time adaptation
        self._init_optimizer()
        
        # Initialize mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler()
    
    def _set_optim_params(self):
        """Only optimize LayerNorm parameters (~0.044% of total params)."""
        self.to_optimize = []
        
        # Vision encoder LayerNorms
        vision_ln_params = [
            p for n, p in self.backbone.named_parameters()
            if 'ln' in n.lower() or 'norm' in n.lower()
        ]
        for p in vision_ln_params:
            p.requires_grad = True
        
        if vision_ln_params:
            self.to_optimize.append({"params": vision_ln_params})
        
        # Text encoder LayerNorms
        if hasattr(self.head, 'module') and hasattr(self.head.module, 'text_encoder'):
            text_ln_params = [
                p for n, p in self.head.module.text_encoder.named_parameters()
                if 'ln' in n.lower() or 'norm' in n.lower()
            ]
            for p in text_ln_params:
                p.requires_grad = True
            
            if text_ln_params:
                self.to_optimize.append({"params": text_ln_params})
    
    def _init_optimizer(self):
        """Initialize optimizer for test-time adaptation."""
        # Get optimizer settings from config
        optimizer_name = self.args.experiment.optimizer.name
        lr = self.args.experiment.optimizer.lr
        weight_decay = self.args.experiment.optimizer.weight_decay
        
        # Set learning rate and weight decay for all param groups
        for param_group in self.to_optimize:
            if 'lr' not in param_group:
                param_group['lr'] = lr
            if 'weight_decay' not in param_group:
                param_group['weight_decay'] = weight_decay
        
        # Create optimizer based on config
        if optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(self.to_optimize)
        elif optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.to_optimize)
        elif optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(self.to_optimize)
        else:
            # Default to AdamW
            self.optimizer = torch.optim.AdamW(self.to_optimize)
    
    def _maybe_build_text_features(self, texts, batch_size):
        """Cache text features to avoid recomputing."""
        key = tuple(texts) if isinstance(texts, (list, tuple)) else texts
        
        if self.text_features is None or key != self._cached_texts_key:
            with torch.no_grad():
                text_features = self.head.module.embed_text(texts, batch_size)
                self.text_features = F.normalize(text_features, dim=-1).to(self.device)
            self._cached_texts_key = key
    
    def compute_class_prototypes(self, features, pseudo_labels, num_classes):
        """
        Vectorized prototype computation using scatter operations.
        """
        B, D = features.shape
        
        # Create one-hot encoding: [B, C]
        one_hot = F.one_hot(pseudo_labels, num_classes).float()
        
        # Count samples per class: [C]
        class_counts = one_hot.sum(dim=0).clamp(min=1e-6)
        
        # Sum features per class: [C, D]
        class_sums = one_hot.T @ features
        
        # Average to get prototypes: [C, D]
        prototypes = class_sums / class_counts.unsqueeze(1)
        
        # Normalize prototypes
        prototypes = F.normalize(prototypes, dim=-1)
        
        return prototypes
    
    def projection_matching_loss(self, prototypes, text_features):
        """Maximize dot product between prototypes and text features."""
        projections = (prototypes * text_features).sum(dim=-1).mean()
        return -projections
    
    def separability_loss(self, prototypes):
        """Maximize pairwise distances using vectorized cosine similarity."""
        # Compute pairwise cosine similarity: [C, C]
        sim_matrix = prototypes @ prototypes.T
        
        # Mask out diagonal
        C = sim_matrix.shape[0]
        mask = ~torch.eye(C, dtype=torch.bool, device=sim_matrix.device)
        
        # Get off-diagonal similarities
        pairwise_sims = sim_matrix[mask]
        
        if pairwise_sims.numel() > 0:
            return -(1 - pairwise_sims.mean())
        
        return torch.tensor(0.0, device=prototypes.device)
    
    def forward(self, images, **kwargs):
        """
        Forward pass with test-time adaptation.
        
        If image_features_only=True, skip adaptation and just return features.
        """
        image_features_only = kwargs.get('image_features_only', False)
        
        # If only features are requested (e.g., for evaluation metrics),
        # skip adaptation and just return features
        if image_features_only:
            with torch.no_grad():
                self.backbone.eval()
                image_features = self.backbone(images)
                return F.normalize(image_features, dim=-1)
        
        texts = kwargs.get('texts')
        experiment = kwargs.get('experiment')
        batch_size = images.shape[0]
        num_classes = len(texts) if texts is not None else experiment.total_num_classes
        
        # Cache text features
        self._maybe_build_text_features(texts, batch_size)
        text_features = self.text_features
        
        # Set model to train mode for LayerNorm adaptation
        # (only LayerNorm params have requires_grad=True)
        self.backbone.train()
        if hasattr(self.head, 'module') and hasattr(self.head.module, 'text_encoder'):
            self.head.module.text_encoder.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            # Forward pass through vision encoder
            image_features = self.backbone(images)
            image_features = F.normalize(image_features, dim=-1)
            
            # Compute logits
            logit_scale = getattr(self.head.module.text_encoder, "logit_scale", 1.0)
            if hasattr(logit_scale, 'exp'):
                logit_scale = logit_scale.exp()
            
            logits = logit_scale * (image_features @ text_features.T)
            
            # Get pseudo-labels
            pseudo_labels = logits.detach().argmax(dim=-1)
            
            # Compute class prototypes
            prototypes = self.compute_class_prototypes(
                image_features, pseudo_labels, num_classes
            )
            
            # Three loss components (all vectorized)
            loss_ent = -(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            loss_pm = self.projection_matching_loss(prototypes, text_features)
            loss_sp = self.separability_loss(prototypes)
            
            # Total loss: Lent - Lpm - Lsp (equation 6 from paper)
            loss = loss_ent + loss_pm + loss_sp
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Optional: Gradient clipping
        if self.args.experiment.optimizer.clip_grad_norm > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for group in self.to_optimize for p in group['params']],
                self.args.experiment.optimizer.clip_grad_norm
            )
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Set back to eval mode for consistent behavior
        self.backbone.eval()
        if hasattr(self.head, 'module') and hasattr(self.head.module, 'text_encoder'):
            self.head.module.text_encoder.eval()
        
        # Return predictions (after adaptation)
        with torch.no_grad():
            # Recompute features and logits after adaptation
            image_features_updated = self.backbone(images)
            image_features_updated = F.normalize(image_features_updated, dim=-1)
            return image_features_updated
            logits_updated = logit_scale * (image_features_updated @ text_features.T)
        
        return {
            'features': image_features_updated,
            'logits': logits_updated
        }
    
    def observe(self, images, targets, **kwargs):
        """
        For TTA, we don't do traditional training via observe.
        This is just a compatibility stub.
        """
        return 0.0
    
    @property
    def checkpoint(self):
        return {
            "self": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict()
        }
    
    def load_from_checkpoint(self, state_dict):
        self.load_state_dict(state_dict["self"])
        if "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        if "scaler" in state_dict:
            self.scaler.load_state_dict(state_dict["scaler"])