import numpy as np
import math
import random
from typing import List, Tuple, Dict, Set

class Preprocessor:
    """
    Implements Algorithm 1: Preprocessing stage for subgraph of voltage X.
    """

    def assign_boxes(self, indices: np.ndarray, boxes: List[int]) -> Dict[int, int]:
        """
        Procedure ASSIGNBOXES (Lines 43-49).
        Randomly assigns each vertex index to a box chosen from the provided boxes.
        """
        mapping = {}
        # Line 44: for each i in indices
        for i in indices:
            # Line 45: Randomly choose b in boxes
            b = random.choice(boxes)
            # Line 46: v_i <- b
            mapping[i] = b
        return mapping

    def run_setup(self, desired_degrees: List[int], input_diameter: int) -> Tuple[np.ndarray, np.ndarray, Set[int], Set[int]]:
        """
        Executes the SETUP procedure (Algorithm 1).

        Args:
            desired_degrees: Vector d of desired degrees.
            input_diameter: The desired diameter delta.

        Returns:
            Tuple containing:
            - d_prime: The inflated degree sequence (numpy array).
            - v: The vertex-box sequence (numpy array, where v[i] is the box ID for node i).
            - D: Set of diameter path vertices.
            - S: Set of subdiameter path vertices.
        """
        # Work with a list for easy inflation, convert to numpy later
        d = list(desired_degrees)
        
        # Line 2: eta <- |{d in d : d > 0}|
        eta = sum(1 for x in d if x > 0)
        
        # Line 3: delta adjustment
        # Formula: delta <- round(delta - 2 * log(eta / (delta + 1)))
        # We use natural log (log) as is standard in theoretical CS papers unless log10 is specified.
        if input_diameter + 1 > 0 and eta > 0:
            term = 2 * math.log(eta / (input_diameter + 1))
            delta = int(round(input_diameter - term))
        else:
            delta = input_diameter
            
        # Ensure delta is valid (at least 1)
        delta = max(1, delta)

        # --- Lines 4-8: Inflate degree sequence ---
        # "Inflate degree seq. until expected number of non-isolated vertices matches"
        while True:
            # Condition: |d| - sum_{d_i in d} exp(-d_i) <= eta
            # Note: |d| grows as we append to it.
            current_n = len(d)
            # Calculate sum of exp(-d_i) for all d_i in current d
            exp_sum = sum(math.exp(-x) for x in d)
            
            lhs = current_n - exp_sum
            
            # Line 5 Check
            if lhs <= eta:
                # If condition met, we proceed to next section
                # (The loop continues WHILE the condition is <= eta? 
                # Wait, the paper says "while ... <= eta do ...". 
                # Usually we want to REACH a target. 
                # If LHS represents expected non-isolated nodes, it starts low and grows?
                # Actually |d| grows by 1. exp_sum grows by small amount (exp(-d)).
                # So LHS grows. We want to STOP when it gets big enough?
                # The pseudocode says "while LHS <= eta". So we loop as long as it is small.
                pass 
            else:
                # Break if LHS > eta
                break

            # Line 6: Randomly select nonzero d from d
            non_zeros = [x for x in d if x > 0]
            if not non_zeros:
                break # Avoid infinite loop if all zeros
            
            chosen_d = random.choice(non_zeros)
            
            # Line 7: d <- (d, d_new)
            d.append(chosen_d)

        d = np.array(d)
        n_total = len(d)

        # --- Lines 9-11: Randomly distribute non-isolated vertices into boxes ---
        # Line 10: I_O <- {i : d_i >= 1}
        I_O = np.where(d >= 1)[0]
        
        # Line 11: B <- {1, ..., delta + 1}
        # We use 0-based indexing: {0, ..., delta}
        B = list(range(delta + 1))

        # --- Lines 13-19: Select Box Subset C ---
        max_d = np.max(d) if len(d) > 0 else 0
        C = []

        # Line 15: if eta / (delta + 1) < max(d) then
        if (delta + 1) > 0 and (eta / (delta + 1)) < max_d:
            # Line 16: Randomly select C subset of B with |C| = eta / max(d)
            # We assume integer division or rounding for the size. 
            # Given "max(d)" is usually small integer, this size might be small.
            c_size = int(math.ceil(eta / max_d)) 
            c_size = max(1, min(c_size, len(B))) # Clamp size
            C = random.sample(B, c_size)
        else:
            # Line 18: C = B
            C = list(B)

        # Line 20: v = ASSIGNBOXES(I_O, C)
        # Initialize v with -1 (representing None/Empty)
        v = np.full(n_total, -1, dtype=int)
        
        assignments = self.assign_boxes(I_O, C)
        for idx, box_id in assignments.items():
            v[idx] = box_id

        # --- Lines 21-31: Diameter Path Selection ---
        # Line 22: if |{i : d_i >= 3}| >= delta + 1
        candidates_ge_3 = np.where(d >= 3)[0]
        
        if len(candidates_ge_3) >= (delta + 1):
            # Line 23
            I_P = candidates_ge_3
        else:
            # Line 25
            I_P = np.where(d >= 2)[0]

        # Line 27: Randomly choose D subset I_P with |D| = delta + 1
        if len(I_P) < (delta + 1):
            raise ValueError(f"Cannot satisfy diameter {delta}: Not enough high-degree nodes.")
            
        D = set(random.sample(list(I_P), delta + 1))

        # Lines 28-31: Assign each i in D to distinct box b in B
        # Note: B has size delta+1. D has size delta+1. This is a bijection.
        available_boxes_D = list(B)
        random.shuffle(available_boxes_D)
        
        for i in D:
            # Line 29: Choose b (we use the shuffled list to ensure distinctness)
            b = available_boxes_D.pop()
            # Line 30: v_i <- b, B <- B \ b (handled by pop)
            v[i] = b

        # --- Lines 32-40: Subdiameter Path Selection ---
        # Line 33: alpha calculation
        # I_P \ D
        I_P_minus_D = list(set(I_P) - D)
        alpha = min(delta + 1, len(I_P_minus_D)) - 1
        
        # Line 34: beta calculation
        # beta <- floor((delta+1)/2) - floor((alpha-1)/2)
        term_beta_1 = math.floor((delta + 1) / 2)
        term_beta_2 = math.floor((alpha - 1) / 2)
        beta = term_beta_1 - term_beta_2
        
        # Line 35: B <- {beta, ..., beta + alpha}
        # In 0-based index: range from beta to beta + alpha (inclusive)
        # We need to map 1-based logic to 0-based carefully. 
        # If paper B was {1..delta+1}, sub-range is consistent.
        # We used 0..delta. So indices match directly.
        # Python range is exclusive at end, so +1
        B_sub_indices = list(range(beta, beta + alpha + 1))
        
        # Filter B_sub to ensure it's within valid box range (0 to delta)
        B_sub = [b for b in B_sub_indices if 0 <= b <= delta]

        # Line 36: Randomly choose S subset I_P \ D with |S| = alpha + 1
        S = set()
        if len(I_P_minus_D) >= (alpha + 1):
             S = set(random.sample(I_P_minus_D, alpha + 1))

        # Lines 37-40: Assign each i in S to distinct box in B_sub
        available_boxes_S = list(B_sub)
        random.shuffle(available_boxes_S)
        
        for i in S:
            if available_boxes_S:
                b = available_boxes_S.pop()
                v[i] = b

        return d, v, D, S