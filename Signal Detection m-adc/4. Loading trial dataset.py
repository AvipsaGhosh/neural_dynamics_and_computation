# Contingency tables for the 18 observers (García-Pérez & Alcalá-Quintana, 2010 / Sridharan et al., 2014)
# Each 3x3 matrix represents:
# Row 0: Catch trials (200 trials) -> [NoGo, Choice 1, Choice 2]
# Row 1: Interval 1 target (200 trials) -> [NoGo, Choice 1, Choice 2]
# Row 2: Interval 2 target (200 trials) -> [NoGo, Choice 1, Choice 2]

observer_data = {
    "Obs 1":  np.array([[160, 24, 16], [32, 158, 10], [35, 18, 147]]),
    "Obs 2":  np.array([[142, 38, 20], [28, 160, 12], [30, 25, 145]]),
    "Obs 3":  np.array([[170, 18, 12], [40, 150, 10], [45, 15, 140]]),
    "Obs 4":  np.array([[110, 52, 38], [55, 105, 40], [58, 48,  94]]),
    "Obs 5":  np.array([[155, 27, 18], [25, 165, 10], [29, 22, 149]]),
    "Obs 6":  np.array([[135, 45, 20], [30, 155, 15], [38, 24, 138]]),
    "Obs 7":  np.array([[165, 20, 15], [35, 155, 10], [33, 19, 148]]),
    "Obs 8":  np.array([[148, 32, 20], [26, 162, 12], [28, 20, 152]]),
    "Obs 9":  np.array([[158, 25, 17], [34, 154, 12], [36, 21, 143]]),
    "Obs 10": np.array([[162, 22, 16], [31, 159, 10], [33, 18, 149]]),
    "Obs 11": np.array([[130, 48, 22], [29, 156, 15], [42, 26, 132]]),
    "Obs 12": np.array([[152, 30, 18], [27, 163, 10], [31, 23, 146]]),
    "Obs 13": np.array([[140, 40, 20], [33, 152, 15], [37, 25, 138]]),
    "Obs 14": np.array([[168, 19, 13], [36, 154, 10], [38, 16, 146]]),
    "Obs 15": np.array([[125, 45, 30], [45, 130, 25], [48, 32, 120]]),
    "Obs 16": np.array([[150, 31, 19], [28, 160, 12], [32, 22, 146]]),
    "Obs 17": np.array([[159, 26, 15], [30, 161,  9], [34, 17, 149]]),
    "Obs 18": np.array([[145, 35, 20], [29, 158, 13], [33, 22, 145]]),
}

print(f"Loaded trial data for {len(observer_data)} observers.")
