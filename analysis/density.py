def calculate_density(person_count, zone_area, max_capacity):
    """
    Calculate crowd density and determine the crowd status.

    person_count  = Number of people detected in the zone
    zone_area     = Area of the monitoring zone
    max_capacity  = Maximum acceptable people in the zone
    """

    # Avoid division by zero
    if zone_area <= 0:
        return 0, "INVALID"

    # Calculate people per unit area
    density = person_count / zone_area

    # Calculate occupancy percentage
    occupancy = (person_count / max_capacity) * 100

    # Determine crowd status
    if occupancy < 50:
        status = "NORMAL"

    elif occupancy < 80:
        status = "MODERATE"

    elif occupancy < 100:
        status = "HIGH"

    else:
        status = "CRITICAL"

    return density, status