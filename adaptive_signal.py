def get_signal_time(risk_score):

    if risk_score < 20:
        return 5

    elif risk_score < 50:
        return 10

    else:
        return 20