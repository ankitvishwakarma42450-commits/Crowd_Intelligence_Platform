import math


def calculate_movement(previous_position, current_position):
    """
    Calculate movement direction and distance
    between two positions.
    """

    if previous_position is None or current_position is None:
        return "UNKNOWN", 0

    previous_x, previous_y = previous_position
    current_x, current_y = current_position

    # Calculate movement
    dx = current_x - previous_x
    dy = current_y - previous_y

    # Calculate distance
    distance = math.sqrt(
        (dx ** 2) + (dy ** 2)
    )

    # Ignore tiny movements caused by detection noise
    if distance < 5:
        direction = "STATIONARY"

    elif abs(dx) > abs(dy):

        if dx > 0:
            direction = "RIGHT"
        else:
            direction = "LEFT"

    else:

        if dy > 0:
            direction = "DOWN"
        else:
            direction = "UP"

    return direction, distance