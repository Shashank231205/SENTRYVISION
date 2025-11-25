def answer_question(video_tensor, question, plm):
    if not question or len(question.strip()) == 0:
        question = "Describe the video in detail."

    rewrite_prompt = (
        "Rewrite the user's question to be maximally precise, grounded in the visual domain, "
        "clarifying ambiguities and making it specific to video understanding. Do not answer, "
        "only rewrite.\n\nQuestion: " + question + "\nRewritten:"
    )

    with torch.no_grad():
        refined_q = plm.generate(
            video=None,
            prompt=rewrite_prompt,
            max_new_tokens=60,
            temperature=0.05
        ).strip()

    meta_prompt = (
        "You are a high-level video-reasoning model. Use multi-frame causal reasoning, "
        "temporal event alignment, object-action correlations, spatial grounding, "
        "motion trajectories, and visual consistency checking. "
        "You must:\n"
        "1. Identify relevant frames\n"
        "2. Extract objects, motions, interactions\n"
        "3. Infer intentions, outcomes, sequence of events\n"
        "4. Provide an answer supported by evidence\n"
        "5. Provide timestamp-level references when possible\n"
        "6. Provide a confidence score (0–100)\n"
        "7. If uncertain, state the reason and give the most probable interpretation.\n"
        "8. Avoid hallucinating unseen objects or events.\n\n"
        "Rewritten Question: " + refined_q + "\n\n"
        "Answer with the format:\n"
        "• Direct Answer\n"
        "• Evidence from frames\n"
        "• Temporal reasoning\n"
        "• Spatial reasoning\n"
        "• Uncertainty (if any)\n"
        "• Confidence score (%)\n\n"
        "Final Answer:\n"
    )

    with torch.no_grad():
        out = plm.generate(
            video=video_tensor,
            prompt=meta_prompt,
            max_new_tokens=500,
            temperature=0.12
        ).strip()

    return out
