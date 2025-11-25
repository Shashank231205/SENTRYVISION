import torch.nn.functional as F

def compare(f1, f2):
    """
    f1, f2 are global embeddings from PE-Core
    """
    return F.cosine_similarity(f1, f2).mean().item()
