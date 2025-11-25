# Adapter wrapper for PLM (Perception Language Model)

from perception_language_model.modeling_plm import PLMForVideo

class PerceptionLanguageModel:

    @staticmethod
    def from_pretrained(name, torch_dtype, device_map):
        return PLMForVideo.from_pretrained(
            name,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
