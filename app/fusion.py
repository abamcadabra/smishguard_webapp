def or_fusion(text_pred: int, url_pred: int) -> tuple[int, str]:
    final_pred = 1 if (text_pred == 1 or url_pred == 1) else 0

    if text_pred == 1 and url_pred == 1:
        reason = "both_models_flagged"
    elif text_pred == 1:
        reason = "text_model_flagged"
    elif url_pred == 1:
        reason = "url_model_flagged"
    else:
        reason = "no_model_flagged"

    return final_pred, reason