def calculate_risk(
    helmet=False,
    triple_riding=False,
    mobile_usage=False,
    overspeed=False
):

    risk = 0

    if not helmet:
        risk += 40

    if triple_riding:
        risk += 25

    if mobile_usage:
        risk += 20

    if overspeed:
        risk += 15

    if risk > 100:
        risk = 100

    return risk