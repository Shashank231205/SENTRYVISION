import torch
import core.vision_encoder.pe as pe

def load_pe_spatial():
    """
    PE-Spatial model — extracts dense spatial features (masks, boxes).
    """
    model_name = "PE-Spatial-L14-448"
    print(f"Loading {model_name} from HF…")

    model = pe.Spatial.from_config(model_name, pretrained=True)
    model = model.cuda()
    model.eval()
    return model
