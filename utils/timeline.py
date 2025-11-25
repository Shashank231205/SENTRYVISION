import json

def generate_timeline(events):
    """
    events: list of strings
    """
    timeline = []
    for i, e in enumerate(events):
        timeline.append({
            "time": f"{i*3}-{(i+1)*3}s",
            "event": e.strip()
        })
    return timeline


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
