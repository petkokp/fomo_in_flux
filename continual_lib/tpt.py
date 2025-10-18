import torch
import torch.nn as nn
import numpy as np
import kornia.augmentation as K
from continual_lib.utils.base_continual_learner import BaseContinualLearner

# --- Helper functions from the reference code ---

def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx]

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def select_confident_samples_idx(logits, top):
    """
    Selects the indices of the most confident samples based on prediction entropy.
    """
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    # Return the indices of the top 'k' samples with the lowest entropy
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return idx
# --- END OF THE DEFINITIVE FIX ---

# --- The Main TPT Class ---

class Model(BaseContinualLearner):
    """
    Test-Time Prompt Tuning (TPT).
    This method adapts a prompt for each test sample via entropy minimization.
    """
    REQ_NON_AUG_INPUTS = False

    def __init__(self, args, backbone, head, loss, device, steps: int = 1, selection_p: float = 0.1, tpt_lr: float = 0.001, **kwargs):
        super(Model, self).__init__(args, backbone, head, loss, device)

        self.steps = steps
        self.selection_p = selection_p
        self.tpt_lr = tpt_lr
        
        # TPT creates its own augmentations on the fly.
        self.augment = nn.Sequential(
            K.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
            K.RandomGrayscale(p=0.2),
        ).to(self.device)

        # Freeze the entire model by default
        self.backbone.requires_grad_(False)
        self.head.requires_grad_(False)
        
        # Initialize the prompts (will be attached later)
        self.prompts = None
        self.initial_prompt_state = None
        self.optimizer = None
        
        # We call begin_task here to set up the prompts immediately
        # This is safe because TPT does not use the training stream
        self.begin_task()

    def begin_task(self, **kwargs):
        """Initializes the prompts and the TPT-specific optimizer."""
        if self.prompts is None:
            prompt_length = self.args.continual.prompt.prompt_length # Re-use prompt length config
            embedding_dim = self.head.module.text_encoder.token_embedding.weight.shape[1]
            prompt_vectors = torch.empty(prompt_length, embedding_dim, device=self.device)
            torch.nn.init.normal_(prompt_vectors, std=0.02)
            
            # Register prompts on the head and set requires_grad=True
            self.head.module.register_parameter("prompts", torch.nn.Parameter(prompt_vectors))
            self.head.module.prompts.requires_grad = True
            self.prompts = self.head.module.prompts

            # Store the initial state for resetting
            self.initial_prompt_state = self.prompts.data.clone()

            # TPT manages its own optimizer
            self.optimizer = torch.optim.AdamW([self.prompts], lr=self.tpt_lr)

    def reset_prompts(self):
        """Resets the prompts to their initial state."""
        self.prompts.data.copy_(self.initial_prompt_state)

    def observe(self, images, targets, **kwargs):
        """TPT does not learn from the training stream. This is a no-op."""
        return 0.0

    @torch.no_grad()
    def forward(self, images, **kwargs):
        self.backbone.eval()
        self.head.eval()
        
        all_classnames = kwargs['texts']
        batch_ensembled_features = []

        for i in range(images.shape[0]):
            image = images[i].unsqueeze(0)
            augmented_images = torch.cat([image] + [self.augment(image) for _ in range(15)], dim=0)

            with torch.cuda.amp.autocast():
                img_features = self.backbone(augmented_images)
                text_features = self.head.module.embed_text(all_classnames, self.args.experiment.evaluation.batch_size)
                
                logit_scale = self.head.module.text_encoder.logit_scale.exp()
                normalized_img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                logits = logit_scale * normalized_img_features @ text_features.T
                
                confident_idx = select_confident_samples_idx(logits, self.selection_p)
                
                ensembled_feature = img_features[confident_idx].mean(dim=0, keepdim=True)
                ensembled_feature = ensembled_feature / ensembled_feature.norm(dim=-1, keepdim=True)
                
                batch_ensembled_features.append(ensembled_feature)

        return torch.cat(batch_ensembled_features, dim=0)