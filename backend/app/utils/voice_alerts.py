def format_alerts_for_speech(alerts):
    if not alerts:
        return "No alerts"
    sentences = []
    for a in alerts:
        sentences.append(f"{a.get('label','object')} detected with confidence {a.get('confidence',0)}")
    return ". ".join(sentences)
