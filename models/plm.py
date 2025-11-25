import torch
from plm_model import PerceptionLanguageModel  # from perception_models repo

def load_plm():
    """
    Loads PLM-3B for reasoning over video embeddings.
    """
    print("Loading PLM-3B…")

    model = PerceptionLanguageModel.from_pretrained(
        "facebook/PLM-3B",
        torch_dtype=torch.float16,
        device_map="cuda"
    )

    model.eval()
    return model
