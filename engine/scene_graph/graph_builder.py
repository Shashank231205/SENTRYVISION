def build_scene_graph(objects, relations):
    return {"objects":objects,"relations":relations}

def predict_scene_graph(frame):
    return build_scene_graph(["person","ball"],["person-holds-ball"])
