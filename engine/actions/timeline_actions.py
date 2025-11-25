def actions_to_timeline(actions):
    timeline = []
    if not actions: return timeline
    last = actions[0]
    start = 0
    for i, act in enumerate(actions):
        if act != last:
            timeline.append({"action": last, "start": start, "end": i})
            start = i
            last = act
    timeline.append({"action": last, "start": start, "end": len(actions)-1})
    return timeline
