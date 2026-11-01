from risk import calculate_risk


# Example scenario
occupancy = 85
density_status = "HIGH"
movement_rate = 60
crowd_growth_rate = 30


risk_score, risk_level = calculate_risk(
    occupancy,
    density_status,
    movement_rate,
    crowd_growth_rate
)


print("Occupancy:", occupancy, "%")
print("Density Status:", density_status)
print("Movement Rate:", movement_rate, "%")
print("Crowd Growth Rate:", crowd_growth_rate, "%")
print("Risk Score:", risk_score)
print("Risk Level:", risk_level)