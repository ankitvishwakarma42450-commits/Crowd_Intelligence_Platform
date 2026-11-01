from movement import calculate_movement


# Previous and current positions
previous_position = (100, 200)
current_position = (160, 205)


direction, distance = calculate_movement(
    previous_position,
    current_position
)


print("Previous Position:", previous_position)
print("Current Position:", current_position)
print("Direction:", direction)
print("Distance:", round(distance, 2))