def slide(video_tensor, window=64, stride=32):
    frames = video_tensor.shape[1]
    chunks = []

    for i in range(0, frames - window + 1, stride):
        chunks.append(video_tensor[:, i:i+window])

    return chunks
