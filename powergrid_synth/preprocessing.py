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
            if not boxes:
                continue 
            b = random.choice(boxes)
            # Line 46: v_i <- b
            mapping[int(i)] = b
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
        if input_diameter + 1 > 0 and eta > 0:
            term = 2 * math.log(eta / (input_diameter + 1))
            delta = int(round(input_diameter - term))
        else:
            delta = input_diameter
            
        # Ensure delta is valid (at least 1)
        delta = max(1, delta)

        # --- Lines 4-8: Inflate degree sequence ---
        while True:
            current_n = len(d)
            exp_sum = sum(math.exp(-x) for x in d)
            
            lhs = current_n - exp_sum
            
            if lhs > eta:
                break

            # Line 6: Randomly select nonzero d from d
            non_zeros = [x for x in d if x > 0]
            if not non_zeros:
                break # Avoid infinite loop if all zeros
            
            chosen_d = random.choice(non_zeros)
            d.append(chosen_d)

        d = np.array(d)
        n_total = len(d)

        # --- Lines 9-11: Randomly distribute non-isolated vertices into boxes ---
        I_O = np.where(d >= 1)[0]
        B = list(range(delta + 1))

        # --- Lines 13-19: Select Box Subset C ---
        max_d = np.max(d) if len(d) > 0 else 0
        C = []

        if (delta + 1) > 0 and max_d > 0 and (eta / (delta + 1)) < max_d:
            c_size = int(math.ceil(eta / max_d)) 
            c_size = max(1, min(c_size, len(B)))
            C = random.sample(B, c_size)
        else:
            C = list(B)

        # Line 20: v = ASSIGNBOXES(I_O, C)
        v = np.full(n_total, -1, dtype=int)
        
        assignments = self.assign_boxes(I_O, C)
        for idx, box_id in assignments.items():
            v[idx] = box_id

        # --- Lines 21-31: Diameter Path Selection ---
        candidates_ge_3 = np.where(d >= 3)[0]
        
        if len(candidates_ge_3) >= (delta + 1):
            I_P = candidates_ge_3
        else:
            I_P = np.where(d >= 2)[0]
            
        # Ensure I_P contains standard Python ints for set operations later
        I_P = [int(x) for x in I_P]

        # Line 27: Randomly choose D subset I_P with |D| = delta + 1
        if len(I_P) < (delta + 1):
            # Fallback: if not enough nodes for diameter, take all available
            D = set(I_P)
        else:
            D = set(random.sample(I_P, delta + 1))

        # Lines 28-31: Assign each i in D to distinct box b in B
        available_boxes_D = list(B)
        random.shuffle(available_boxes_D)
        
        for i in D:
            if available_boxes_D:
                b = available_boxes_D.pop()
                v[i] = b

        # --- Lines 32-40: Subdiameter Path Selection ---
        # Line 33: alpha calculation
        I_P_minus_D = list(set(I_P) - D)
        alpha = min(delta + 1, len(I_P_minus_D)) - 1
        
        if alpha >= 0:
            # Line 34: beta calculation
            term_beta_1 = math.floor((delta + 1) / 2)
            term_beta_2 = math.floor((alpha - 1) / 2)
            beta = term_beta_1 - term_beta_2
            
            # Line 35: B <- {beta, ..., beta + alpha}
            # The formula gives 1-based indices (relative to 1..delta+1)
            # We must shift to 0-based indices for Python (0..delta)
            B_sub_indices = [b - 1 for b in range(beta, beta + alpha + 1)]
            
            # Filter B_sub to ensure it's within valid box range (0 to delta)
            B_sub = [b for b in B_sub_indices if 0 <= b <= delta]

            # Line 36: Randomly choose S subset I_P \ D with |S| = alpha + 1
            S = set()
            
            sample_size = min(len(I_P_minus_D), alpha + 1)
            if sample_size > 0:
                S = set(random.sample(I_P_minus_D, sample_size))

            # Lines 37-40: Assign each i in S to distinct box in B_sub
            available_boxes_S = list(B_sub)
            random.shuffle(available_boxes_S)
            
            for i in S:
                if available_boxes_S:
                    b = available_boxes_S.pop()
                    v[i] = b
        else:
            S = set()

        return d, v, D, S