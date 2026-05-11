# src/evaluation/adversarial_test.py

import torch
from torchattacks import PGD, FGSM

def adversarial_robustness_evaluation(model, test_loader, 
                                       device='cuda'):
    """
    Test model accuracy under PGD and FGSM attacks.
    A robust detector should maintain >70% accuracy 
    under epsilon=8/255 L∞ perturbation.
    """
    model.eval()
    
    attacks = {
        'FGSM_8': FGSM(model, eps=8/255),
        'PGD_10': PGD(model, eps=8/255, alpha=2/255, steps=10),
        'PGD_50': PGD(model, eps=8/255, alpha=1/255, steps=50),
    }
    
    results = {}
    
    for attack_name, attack in attacks.items():
        correct = 0
        total = 0
        
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            adv_images = attack(images, labels)
            
            with torch.no_grad():
                logits, _ = model(adv_images)
                preds = (torch.sigmoid(logits) > 0.5).long()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        results[attack_name] = correct / total
        print(f"{attack_name} accuracy: {results[attack_name]:.4f}")
    
    return results