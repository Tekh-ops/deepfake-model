# src/training/continual_learning.py

import torch

class EWCTrainer:
    """
    Elastic Weight Consolidation for continual learning.
    Allows updating models on new deepfake types without
    forgetting performance on previously seen types.
    """
    def __init__(self, model, dataloader, ewc_lambda=5000):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.fisher_matrix = {}
        self.optimal_params = {}
        self._compute_fisher(dataloader)
    
    def _compute_fisher(self, dataloader):
        """Compute diagonal Fisher information matrix."""
        self.model.eval()
        
        for name, param in self.model.named_parameters():
            self.fisher_matrix[name] = torch.zeros_like(param)
            self.optimal_params[name] = param.data.clone()
        
        for images, labels in dataloader:
            self.model.zero_grad()
            logits, _ = self.model(images.cuda())
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels.float().cuda()
            )
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_matrix[name] += \
                        param.grad.data.pow(2)
        
        # Normalize by dataset size
        n = len(dataloader.dataset)
        for name in self.fisher_matrix:
            self.fisher_matrix[name] /= n
    
    def ewc_loss(self) -> torch.Tensor:
        """EWC regularization term to add to training loss."""
        loss = torch.tensor(0.0).cuda()
        for name, param in self.model.named_parameters():
            if name in self.fisher_matrix:
                fisher = self.fisher_matrix[name]
                optimal = self.optimal_params[name]
                loss += (fisher * (param - optimal).pow(2)).sum()
        return (self.ewc_lambda / 2) * loss