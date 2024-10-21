#Local Trainers' Selection
Inputs:
    M: List of all near-RT-RICs, each represented as a dictionary with their timing attributes.
    t_round_u, t_round_e, t_round_m: Round times for the different slices (u, e, and m).
    alpha: Weighting factor for the estimated time.

Steps:
  For each slice (u, e, and m), the algorithm selects trainers based on the minimum combined time using the formula from the pseudocode.
  Selected trainers are added to the corresponding slice sets (N_u, N_e, N_m).
  Trainers whose cumulative time exceeds the deadline (t_round^i) are discarded.

Output: The combined set of selected trainers across all slices is returned.


#MCORANFed
    Compress Gradient: The function compress_gradient() implements equation (21), where the gradient is compressed with momentum and a compression function CC.
    Decompress Gradient: The function decompress_gradient() decompresses the gradients received from local trainers.
    Aggregate Weights: This function implements equation (18) to aggregate the weights based on the participants' data sizes.
    Global Loss Function: Implements equation (20), calculating the global loss by aggregating the local loss functions.
    Main Loop:
        For each iteration, a subset of trainers is selected, and their local gradients are trained, compressed, transmitted, decompressed, and aggregated.
        The global loss and global accuracy are calculated in each iteration.
        The final trained model is returned after all iterations.

Key Parameters:
    MM: Number of participants.
    KϵKϵ​: Number of iterations.
    θθ: Accuracy threshold for local training.
    CC: Compression function.
    γγ: Momentum parameter.


#Joint Optimization Problem
    Variables:
        a_t: Binary variable indicating whether a participant is selected at time t.
        b_t: Continuous variable representing the bandwidth allocation for participant m at time t.
        f_m: Frequency of computation for each participant.
    Objective Function:
        The cost function is defined based on the given expression, which includes transmission cost, computation cost, and delay terms.
    Constraints:
        Bandwidth constraint ensures the sum of allocated bandwidth does not exceed the total available bandwidth (B).
        The sum of bandwidth fractions for each participant equals 1.
        b_t is constrained between b_min and 1.
        Frequency for each participant must be greater than or equal to f_min.
        Binary constraint for a_t is automatically handled by using boolean=True in the variable definition.
    Solver:
        The problem is solved using CVXPY, and the optimal values for a_t, b_t, and f_m are printed along with the optimal cost.

Dependencies:
You will need to install CVXPY

