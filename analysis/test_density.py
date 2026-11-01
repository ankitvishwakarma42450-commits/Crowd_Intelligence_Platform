from density import calculate_density


# Example values
people = 8
zone_area = 20
max_capacity = 10

density, status = calculate_density(
    people,
    zone_area,
    max_capacity
)

print("People:", people)
print("Density:", round(density, 2))
print("Status:", status)