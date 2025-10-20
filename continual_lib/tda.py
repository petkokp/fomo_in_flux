import torch
import torch.nn as nn
import torch.nn.functional as F
from continual_lib.utils.base_continual_learner import BaseContinualLearner


class Model(BaseContinualLearner):
    """
    Training-free Dynamic Adapter (TDA) for Test-Time Adaptation.
    
    Implements the method from "Efficient Test-Time Adaptation of Vision-Language Models"
    using positive and negative caches for pseudo-label refinement.
    """
    REQ_NON_AUG_INPUTS = False

    def __init__(
        self,
        args, backbone, head, loss, device,
        pos_shot_capacity: int = 3,           # k for positive cache (per class)
        neg_shot_capacity: int = 2,           # k for negative cache (per class)
        alpha: float = 2.0,                   # residual ratio (weight for cache)
        beta: float = 5.0,                    # sharpness ratio
        tau_l: float = 0.2,                   # lower entropy threshold for negative cache
        tau_h: float = 0.5,                   # upper entropy threshold for negative cache
        neg_threshold: float = 0.03,          # threshold for negative pseudo-labeling
        **kwargs,
    ):
        super().__init__(args, backbone, head, loss, device)
        
        # Hyperparameters
        self.pos_shot_capacity = pos_shot_capacity
        self.neg_shot_capacity = neg_shot_capacity
        self.alpha = alpha
        self.beta = beta
        self.tau_l = tau_l
        self.tau_h = tau_h
        self.neg_threshold = neg_threshold
        
        # Freeze the model for test-time adaptation
        self.backbone.requires_grad_(False).eval()
        self.head.requires_grad_(False).eval()
        
        # Initialize caches as None (will be built during inference)
        self.pos_cache_keys = None      # [N*k, D] positive features
        self.pos_cache_values = None    # [N*k, N] positive one-hot labels
        self.pos_cache_entropies = None # [N*k] entropy values
        self.pos_cache_counts = None    # [N] count per class
        
        self.neg_cache_keys = None      # [N*˜k, D] negative features
        self.neg_cache_values = None    # [N*˜k, N] negative labels
        self.neg_cache_entropies = None # [N*˜k] entropy values
        self.neg_cache_counts = None    # [N] count per class
        
        # Cache for text features
        self.text_features = None
        self._cached_texts_key = None
        
        # Store logit scale
        with torch.no_grad():
            self.logit_scale = self.head.module.text_encoder.logit_scale.exp().to(device)
        
        # Track current number of classes to detect changes
        self.current_num_classes = None

    def _maybe_build_text_features(self, texts, batch_size: int):
        """Cache text features to avoid recomputation."""
        key = tuple(texts) if isinstance(texts, (list, tuple)) else texts
        if (self.text_features is None) or (key != self._cached_texts_key):
            with torch.no_grad():
                tf = self.head.module.embed_text(texts, batch_size).to(self.device)
                tf = tf / (tf.norm(dim=-1, keepdim=True) + 1e-12)
            self.text_features = tf
            self._cached_texts_key = key

    def _init_caches(self, num_classes: int, feature_dim: int):
        """Initialize empty caches. Reinitializes if number of classes changes."""
        # Check if we need to reinitialize due to class count change
        needs_init = (
            self.pos_cache_keys is None or 
            self.current_num_classes is None or
            self.current_num_classes != num_classes
        )
        
        if needs_init:
            self.current_num_classes = num_classes
            
            self.pos_cache_keys = torch.zeros(
                num_classes * self.pos_shot_capacity, feature_dim, 
                device=self.device
            )
            self.pos_cache_values = torch.zeros(
                num_classes * self.pos_shot_capacity, num_classes,
                device=self.device
            )
            self.pos_cache_entropies = torch.full(
                (num_classes * self.pos_shot_capacity,), 
                float('inf'), device=self.device
            )
            self.pos_cache_counts = torch.zeros(num_classes, device=self.device, dtype=torch.long)
            
            self.neg_cache_keys = torch.zeros(
                num_classes * self.neg_shot_capacity, feature_dim,
                device=self.device
            )
            self.neg_cache_values = torch.zeros(
                num_classes * self.neg_shot_capacity, num_classes,
                device=self.device
            )
            self.neg_cache_entropies = torch.full(
                (num_classes * self.neg_shot_capacity,),
                float('inf'), device=self.device
            )
            self.neg_cache_counts = torch.zeros(num_classes, device=self.device, dtype=torch.long)

    def _compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy from logits."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy

    def _adaptation_function(self, affinity: torch.Tensor) -> torch.Tensor:
        """
        Adaptation function: A(z) = α * exp(-β(1-z))
        
        Args:
            affinity: Similarity scores (e.g., cosine similarity)
        Returns:
            Weighted affinity
        """
        return self.alpha * torch.exp(-self.beta * (1 - affinity))

    def _update_positive_cache(self, features: torch.Tensor, logits: torch.Tensor):
        """
        Update positive cache with high-confidence predictions.
        
        Args:
            features: [B, D] normalized image features
            logits: [B, N] CLIP logits
        """
        B, N = logits.shape
        D = features.shape[1]
        
        # Get pseudo labels and entropy
        probs = F.softmax(logits, dim=-1)
        pseudo_labels = torch.argmax(probs, dim=-1)  # [B]
        entropies = self._compute_entropy(logits)    # [B]
        
        # Process each sample
        for b in range(B):
            cls_idx = pseudo_labels[b].item()
            entropy = entropies[b].item()
            feature = features[b]
            
            # One-hot encode label
            label_onehot = torch.zeros(N, device=self.device)
            label_onehot[cls_idx] = 1.0
            
            # Get the range for this class in cache
            start_idx = cls_idx * self.pos_shot_capacity
            end_idx = start_idx + self.pos_shot_capacity
            
            current_count = self.pos_cache_counts[cls_idx].item()
            
            if current_count < self.pos_shot_capacity:
                # Add new entry
                insert_idx = start_idx + current_count
                self.pos_cache_keys[insert_idx] = feature
                self.pos_cache_values[insert_idx] = label_onehot
                self.pos_cache_entropies[insert_idx] = entropy
                self.pos_cache_counts[cls_idx] += 1
            else:
                # Replace highest entropy entry if current is lower
                class_entropies = self.pos_cache_entropies[start_idx:end_idx]
                max_entropy_pos = torch.argmax(class_entropies)
                max_entropy = class_entropies[max_entropy_pos].item()
                
                if entropy < max_entropy:
                    replace_idx = start_idx + max_entropy_pos
                    self.pos_cache_keys[replace_idx] = feature
                    self.pos_cache_values[replace_idx] = label_onehot
                    self.pos_cache_entropies[replace_idx] = entropy

    def _update_negative_cache(self, features: torch.Tensor, logits: torch.Tensor):
        """
        Update negative cache with uncertain predictions.
        
        Args:
            features: [B, D] normalized image features
            logits: [B, N] CLIP logits
        """
        B, N = logits.shape
        
        # Get probabilities and entropy
        probs = F.softmax(logits, dim=-1)            # [B, N]
        entropies = self._compute_entropy(logits)     # [B]
        
        # Select samples with entropy in [tau_l, tau_h]
        mask = (entropies >= self.tau_l) & (entropies <= self.tau_h)
        
        if not mask.any():
            return
        
        # Filter selected samples
        selected_features = features[mask]      # [M, D]
        selected_probs = probs[mask]            # [M, N]
        selected_entropies = entropies[mask]    # [M]
        
        # Generate negative pseudo labels: -1 where prob < threshold, else 0
        neg_labels = torch.where(
            selected_probs < self.neg_threshold,
            torch.full_like(selected_probs, -1.0),
            torch.zeros_like(selected_probs)
        )  # [M, N]
        
        # For each selected sample, try to add to cache
        for idx in range(selected_features.shape[0]):
            feature = selected_features[idx]
            neg_label = neg_labels[idx]
            entropy = selected_entropies[idx].item()
            
            # Find the class with highest probability (most uncertain about)
            cls_idx = torch.argmax(selected_probs[idx]).item()
            
            # Get the range for this class in negative cache
            start_idx = cls_idx * self.neg_shot_capacity
            end_idx = start_idx + self.neg_shot_capacity
            
            current_count = self.neg_cache_counts[cls_idx].item()
            
            if current_count < self.neg_shot_capacity:
                # Add new entry
                insert_idx = start_idx + current_count
                self.neg_cache_keys[insert_idx] = feature
                self.neg_cache_values[insert_idx] = neg_label
                self.neg_cache_entropies[insert_idx] = entropy
                self.neg_cache_counts[cls_idx] += 1
            else:
                # Replace highest entropy entry if current is lower
                class_entropies = self.neg_cache_entropies[start_idx:end_idx]
                max_entropy_pos = torch.argmax(class_entropies)
                max_entropy = class_entropies[max_entropy_pos].item()
                
                if entropy < max_entropy:
                    replace_idx = start_idx + max_entropy_pos
                    self.neg_cache_keys[replace_idx] = feature
                    self.neg_cache_values[replace_idx] = neg_label
                    self.neg_cache_entropies[replace_idx] = entropy

    def _predict_from_cache(self, features: torch.Tensor, cache_keys: torch.Tensor, 
                           cache_values: torch.Tensor, cache_counts: torch.Tensor) -> torch.Tensor:
        """
        Predict using cache model.
        
        Args:
            features: [B, D] query features
            cache_keys: [N*k, D] cached features
            cache_values: [N*k, N] cached labels
            cache_counts: [N] number of samples per class
        Returns:
            predictions: [B, N]
        """
        # Only use filled cache entries
        mask = cache_counts > 0
        if not mask.any():
            return torch.zeros(features.shape[0], cache_values.shape[1], device=self.device)
        
        # Compute affinity: [B, D] @ [D, N*k] = [B, N*k]
        affinity = features @ cache_keys.T
        
        # Apply adaptation function
        weighted_affinity = self._adaptation_function(affinity)  # [B, N*k]
        
        # Compute predictions: [B, N*k] @ [N*k, N] = [B, N]
        predictions = weighted_affinity @ cache_values
        
        return predictions

    @torch.no_grad()
    def forward(self, images, **kwargs):
        """
        Forward pass for test-time adaptation.
        
        Args:
            images: [B, C, H, W] input images
            kwargs: should contain 'texts' for adaptation (optional)
        Returns:
            Image features [B, D] - adapted if texts provided, otherwise unadapted
        """
        self.backbone.eval()
        self.head.eval()
        
        # Extract and normalize image features
        img_features = self.backbone(images)  # [B, D]
        img_features = img_features / (img_features.norm(dim=-1, keepdim=True) + 1e-12)
        
        # If texts are not provided, return unadapted features
        # (e.g., for custom head evaluation)
        texts = kwargs.get('texts', None)
        if texts is None:
            return img_features
            
        self._maybe_build_text_features(texts, self.args.experiment.evaluation.batch_size)
        
        text_features = self.text_features  # [N, D]
        N = text_features.shape[0]
        D = img_features.shape[1]
        
        # Initialize caches if needed (will reinitialize if num_classes changed)
        self._init_caches(N, D)
        
        # Compute CLIP predictions (unscaled for entropy calculation)
        clip_logits = img_features @ text_features.T  # [B, N]
        
        # Update caches with current batch
        self._update_positive_cache(img_features, clip_logits)
        self._update_negative_cache(img_features, clip_logits)
        
        # Get predictions from caches
        pos_predictions = self._predict_from_cache(
            img_features, self.pos_cache_keys, 
            self.pos_cache_values, self.pos_cache_counts
        )  # [B, N]
        
        neg_predictions = self._predict_from_cache(
            img_features, self.neg_cache_keys,
            self.neg_cache_values, self.neg_cache_counts
        )  # [B, N]
        
        # Combine predictions: CLIP + positive cache + negative cache
        adapted_logits = clip_logits + pos_predictions + neg_predictions  # [B, N]
        
        # Compute adapted features by projecting back from adapted logits
        # This allows the adapted predictions to be used with custom heads
        # Method: renormalize the weighted combination with text features
        adapted_features = adapted_logits @ text_features  # [B, N] @ [N, D] = [B, D]
        adapted_features = adapted_features / (adapted_features.norm(dim=-1, keepdim=True) + 1e-12)
        
        return adapted_features

    @property
    def checkpoint(self):
        """Return checkpoint dict."""
        return {
            "self": self.state_dict(),
            "pos_cache_keys": self.pos_cache_keys,
            "pos_cache_values": self.pos_cache_values,
            "pos_cache_entropies": self.pos_cache_entropies,
            "pos_cache_counts": self.pos_cache_counts,
            "neg_cache_keys": self.neg_cache_keys,
            "neg_cache_values": self.neg_cache_values,
            "neg_cache_entropies": self.neg_cache_entropies,
            "neg_cache_counts": self.neg_cache_counts,
        }

    def load_from_checkpoint(self, state_dict):
        """Load from checkpoint."""
        self.load_state_dict(state_dict["self"])
        if "pos_cache_keys" in state_dict:
            self.pos_cache_keys = state_dict["pos_cache_keys"]
            self.pos_cache_values = state_dict["pos_cache_values"]
            self.pos_cache_entropies = state_dict["pos_cache_entropies"]
            self.pos_cache_counts = state_dict["pos_cache_counts"]
            self.neg_cache_keys = state_dict["neg_cache_keys"]
            self.neg_cache_values = state_dict["neg_cache_values"]
            self.neg_cache_entropies = state_dict["neg_cache_entropies"]
            self.neg_cache_counts = state_dict["neg_cache_counts"]