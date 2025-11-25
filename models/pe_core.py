import torch
import core.vision_encoder.pe as pe

def load_pe_core():
    """
    Loads PE-Core model for global video embeddings.
    Best choice: PE-Core-L14-336
    """
    model_name = "PE-Core-L14-336"
    print(f"Loading {model_name} from HF…")

    model = pe.CLIP.from_config(model_name, pretrained=True)
    model = model.cuda()
    model.eval()
    return model
