def deadline_aware_trainer_selection(M, t_round_u, t_round_e, t_round_m, alpha):
    """
    Deadline aware and slicing based local trainers' selection algorithm.

    Args:
    M (list): Set of all near-RT-RICs.
    t_round_u (float): Round time for slice u.
    t_round_e (float): Round time for slice e.
    t_round_m (float): Round time for slice m.
    alpha (float): Weighting factor for the estimated time.

    Returns:
    list: Combined set of selected trainers.
    """
    
    # Initialize sets
    N_u, N_e, N_m = [], [], []
    
    # Slicing times for each slice
    t_rounds = {"u": t_round_u, "e": t_round_e, "m": t_round_m}

    # Loop through slices
    for slice_type, t_round in t_rounds.items():
        N = M.copy()  # Copy the set of near-RT-RICs for each slice
        while len(N) > 0:
            # Select the trainer with minimum combined time
            x = min(N, key=lambda n: 0.5 * (n["t_k_minus_1"] + alpha * n["t_k_estimated"]))

            # Calculate the total time for the current trainer
            t = x["t_1"] + x["t_agg"] + x["t_k"]
            
            # Remove selected trainer from the set
            N.remove(x)

            # Check if the trainer can be selected based on the deadline
            if t < t_round:
                t += x["t_k"]
                # Add the selected trainer to the corresponding slice set
                if slice_type == "u":
                    N_u.append(x)
                elif slice_type == "e":
                    N_e.append(x)
                elif slice_type == "m":
                    N_m.append(x)

    # Return the combined set of selected trainers
    return N_u + N_e + N_m


# Example usage with dummy data
M = [
    {"id": 1, "t_k_minus_1": 2.0, "t_k_estimated": 1.5, "t_1": 0.5, "t_agg": 0.8, "t_k": 1.2},
    {"id": 2, "t_k_minus_1": 2.2, "t_k_estimated": 1.6, "t_1": 0.7, "t_agg": 0.9, "t_k": 1.3},
    # Add more trainers as needed...
]

t_round_u = 10.0  # Example round time for slice u
t_round_e = 12.0  # Example round time for slice e
t_round_m = 15.0  # Example round time for slice m
alpha = 0.7  # Example alpha value

selected_trainers = deadline_aware_trainer_selection(M, t_round_u, t_round_e, t_round_m, alpha)
print("Selected trainers:", selected_trainers)
