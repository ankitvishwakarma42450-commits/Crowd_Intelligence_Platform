def calculate_risk(
    occupancy,
    density_status,
    movement_rate=0,
    crowd_growth_rate=0
):
    """
    Calculate crowd risk based on:
    - Occupancy percentage
    - Density status
    - Movement rate
    - Crowd growth rate
    """

    risk_score = 0

    # -----------------------------
    # Occupancy contribution
    # -----------------------------

    if occupancy >= 100:
        risk_score += 50

    elif occupancy >= 80:
        risk_score += 35

    elif occupancy >= 60:
        risk_score += 20

    elif occupancy >= 40:
        risk_score += 10


    # -----------------------------
    # Density contribution
    # -----------------------------

    if density_status == "CRITICAL":
        risk_score += 40

    elif density_status == "HIGH":
        risk_score += 25

    elif density_status == "MODERATE":
        risk_score += 10


    # -----------------------------
    # Movement contribution
    # -----------------------------

    if movement_rate >= 80:
        risk_score += 20

    elif movement_rate >= 50:
        risk_score += 10


    # -----------------------------
    # Crowd growth contribution
    # -----------------------------

    if crowd_growth_rate >= 50:
        risk_score += 20

    elif crowd_growth_rate >= 25:
        risk_score += 10


    # -----------------------------
    # Limit score
    # -----------------------------

    risk_score = min(risk_score, 100)


    # -----------------------------
    # Determine risk level
    # -----------------------------

    if risk_score >= 75:
        risk_level = "CRITICAL"

    elif risk_score >= 50:
        risk_level = "HIGH"

    elif risk_score >= 25:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"


    return risk_score, risk_level