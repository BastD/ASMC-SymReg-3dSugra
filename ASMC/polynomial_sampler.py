import numpy as np
import numba as nb
import time
from scipy.special import comb
from collections import deque
from functools import reduce
from itertools import product
from typing import Tuple, Callable

from .vectorized_helpers import (
    build_transition_lookup_matrix,
    apply_modify_operation_vectorized,
    apply_add_operation_vectorized,
    apply_remove_operation_vectorized,
    apply_multiply_divide_operation_vectorized
)


@nb.njit
def evaluate_polynomial_batch_fast(term_matrix, coefficients_batch):
    """
    Version JIT compilée de la somme carée de l'évaluation de polynôme. 
    """
    return np.sum(np.square(np.dot(coefficients_batch, term_matrix.T)), axis=1)

@nb.njit
def sum_coeffients_batch_fast(coefficients_batch):
    """
    Version JIT compilée de l'évaluation carrée des coefficients.
    """
    return np.sum(coefficients_batch**2,axis=1)

class SparsePolynomialSampler:

    """
    Class for sampling sparse polynomials using Annealing Importance Sampling,
    with a specialized Metropolis-Hastings kernel to find a polynomial P such that P(x_data) = 0.
    """
    
    def __init__(self, 
                data_x: np.ndarray, 
                max_degree: int, 
                num_vars: int,
                max_num_monomials: int = 5,
                sigma_proposal: Callable[[int, int], float] = lambda i,n_iter: 0.1,
                prob_add: Callable[[int, int], float] = lambda i,n_iter: 0.2,
                prob_remove: Callable[[int, int], float] = lambda i,n_iter: 0.2,
                prob_modify: Callable[[int, int], float] = lambda i,n_iter: 0.2,
                prob_divide: Callable[[int, int], float] = lambda i,n_iter: 0.2,
                prob_multiply: Callable[[int, int], float] = lambda i,n_iter: 0.2,
                regularisation_factor: float = 1e3,
                min_coeff_threshold: float = 0.1,
                random_seed: int = 42
                ):
        """
        Initialize the sparse polynomial sampler with data and parameters.

        Args:
            data_x (np.ndarray): Input data of shape (n_samples, num_vars).
            max_degree (int): Maximum degree of the polynomial.
            num_vars (int): Number of variables.
            max_num_monomials (int): Maximum number of nonzero monomials in a polynomial.
            sigma_proposal (float): Std. dev. for proposed coefficient changes.
            prob_add (float): Probability of adding a monomial.
            prob_remove (float): Probability of removing a monomial.
            prob_modify (float): Probability of modifying a coefficient.
            prob_divide (float): Probability of dividing a monomial by a variable.
            prob_multiply (float): Probability of multiplying a monomial by a variable.
            regularisation_factor (float): Factor for regularizing the sum of coefficients.
            min_coeff_threshold (float): Coefficients below this are set to zero.
            random_seed (int): Random seed for np.random
        """
        self.data_x = data_x
        self.max_degree = max_degree
        self.num_vars = num_vars
        self.max_num_monomials = max_num_monomials
        self.sigma_proposal = sigma_proposal
        self.min_coeff_threshold = min_coeff_threshold          
        self.prob_add = prob_add 
        self.prob_remove = prob_remove
        self.prob_modify = prob_modify 
        self.prob_multiply = prob_multiply 
        self.prob_divide = prob_divide 

        self.regularisation_factor = regularisation_factor

        self.random_seed = random_seed

        self.num_terms = self._compute_num_terms()
        self.term_powers = self._generate_term_powers()
        self.term_matrix = self._precompute_term_matrix()

        # Pré-calcul des transitions multiply/divide pour vectorisation GPU
        self.multiply_transitions, self.divide_transitions = self._precompute_transitions()

        # Construire les matrices de lookup pour version pure vectorisée
        max_trans_mult = max([len(v) for v in self.multiply_transitions.values()]) if self.multiply_transitions else 1
        max_trans_div = max([len(v) for v in self.divide_transitions.values()]) if self.divide_transitions else 1
        self.multiply_lookup_matrix, self.multiply_lookup_valid = build_transition_lookup_matrix(
            self.multiply_transitions, max_trans_mult
        )
        self.divide_lookup_matrix, self.divide_lookup_valid = build_transition_lookup_matrix(
            self.divide_transitions, max_trans_div
        )

        print(f"Number of possible terms in the polynomial: {self.num_terms}")
    
    def apply_coefficient_threshold(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Set coefficients smaller than min_coeff_threshold to zero.

        Args:
            coefficients (np.ndarray): Polynomial coefficients (vector or batch).

        Returns:
            np.ndarray: Filtered coefficients (vector or batch).
        """
        # Copier pour éviter de modifier l'original
        filtered_coeffs = coefficients.copy()
        
        # Mettre à zéro les coefficients dont la valeur absolue est inférieure au seuil
        filtered_coeffs[np.abs(filtered_coeffs) < self.min_coeff_threshold] = 0.0
        
        # Si tous les coefficients sont mis à zéro, restaurer au moins le plus grand
        if len(filtered_coeffs.shape) == 1:
            if np.all(filtered_coeffs == 0):
                idx_max = np.argmax(np.abs(coefficients))
                filtered_coeffs[idx_max] = coefficients[idx_max]
        else:
            zero_rows = np.all(filtered_coeffs == 0, axis=1)
            if np.any(zero_rows):
                # Pour chaque ligne entièrement nulle, restaurer le coefficient de plus grande valeur absolue
                for i in np.where(zero_rows)[0]:
                    idx_max = np.argmax(np.abs(coefficients[i]))
                    filtered_coeffs[i, idx_max] = coefficients[i, idx_max]
    
        return filtered_coeffs
    
    def normalize_polynomial_batch(self, coefficients_batch: np.ndarray) -> np.ndarray:
        """
        Normalize polynomial coefficients to avoid trivial solution P=0.
        """
        # Calculer la norme L2 pour chaque ensemble de coefficients
        norms = np.sqrt(np.sum(np.square(coefficients_batch), axis=1, keepdims=True))
        
        # Éviter la division par zéro en ajoutant un epsilon
        norms = np.maximum(norms, 1e-10)
        
        # Normaliser
        normalized = coefficients_batch / norms
        
        # S'assurer qu'aucun polynôme n'est entièrement nul
        zero_polynomials = np.all(np.abs(normalized) < 1e-10, axis=1)
        
        np.random.seed = self.random_seed
        if np.any(zero_polynomials):
            # Pour chaque polynôme nul, ajouter un coefficient aléatoire
            for i in np.where(zero_polynomials)[0]:
                idx = np.random.randint(0, coefficients_batch.shape[1])
                normalized[i, idx] = 1.0
        
        return normalized
        
    def _compute_num_terms(self) -> int:
        """
        Compute number of terms in a polynomial of degree self.max_degree with self.num_vars variables.
        """
        return int(comb(self.num_vars + self.max_degree, self.max_degree))
    
    def _generate_term_powers(self) -> np.ndarray:
        """
        Generate all possible monomial terms (variable powers) for a polynomial
        of degree max_degree with num_vars variables, excluding the constant term.
        """
        term_powers = []
        
        # Fonction récursive pour générer toutes les combinaisons de puissances
        def generate_powers(current_powers, remaining_degree, var_idx):
            if var_idx == self.num_vars - 1:
                current_powers[var_idx] = remaining_degree
                term_powers.append(current_powers.copy())
                return
            
            for power in range(remaining_degree + 1):
                current_powers[var_idx] = power
                generate_powers(current_powers.copy(), remaining_degree - power, var_idx + 1)
        
        # Générer tous les termes possibles de degré 1 à max_degree
        # Commencer à 1 au lieu de 0 pour exclure le terme constant
        #for degree in range(1, self.max_degree + 1):
        for degree in range(self.max_degree + 1):
            generate_powers(np.zeros(self.num_vars, dtype=int), degree, 0)
        
        return np.array(term_powers)
    
    def _precompute_term_matrix(self) -> np.ndarray:
        """
        Précalcule la matrice de tous les termes évalués sur les données d'entrée.

        Returns:
            Matrice de forme (n_samples, num_terms) où chaque élément [i, j] est
            la valeur du terme j au point data_x[i]
        """
        n_samples = len(self.data_x)
        term_matrix = np.ones((n_samples, self.num_terms))

        for term_idx, powers in enumerate(self.term_powers):
            for var_idx, power in enumerate(powers):
                if power > 0:
                    term_matrix[:, term_idx] *= np.power(self.data_x[:, var_idx], power)

        return term_matrix

    def _precompute_transitions(self) -> Tuple[dict, dict]:
        """
        Precompute all possible transitions for multiply and divide operations.
        This enables vectorized implementation on GPU.

        Returns:
            Tuple[dict, dict]:
                - multiply_transitions[term_idx]: list of (var_idx, target_term_idx)
                - divide_transitions[term_idx]: list of (var_idx, target_term_idx)
        """
        multiply_transitions = {}
        divide_transitions = {}

        print("Pre-compute multiply/divide transitions...")

        for term_idx, term_powers in enumerate(self.term_powers):
            # --- MULTIPLY transitions ---
            multiply_list = []

            # Vérifier si on peut multiplier (degré total < max_degree)
            if np.sum(term_powers) < self.max_degree:
                for var_idx in range(self.num_vars):
                    # Créer les nouvelles puissances après multiplication
                    new_powers = term_powers.copy()
                    new_powers[var_idx] += 1

                    # Trouver l'index du terme correspondant
                    matches = np.where((self.term_powers == new_powers).all(axis=1))[0]
                    if len(matches) > 0:
                        target_term_idx = matches[0]
                        multiply_list.append((var_idx, target_term_idx))

            multiply_transitions[term_idx] = multiply_list

            # --- DIVIDE transitions ---
            divide_list = []

            for var_idx in range(self.num_vars):
                # Vérifier si cette variable a une puissance positive
                if term_powers[var_idx] > 0:
                    # Créer les nouvelles puissances après division
                    new_powers = term_powers.copy()
                    new_powers[var_idx] -= 1

                    # Trouver l'index du terme correspondant
                    matches = np.where((self.term_powers == new_powers).all(axis=1))[0]
                    if len(matches) > 0:
                        target_term_idx = matches[0]
                        divide_list.append((var_idx, target_term_idx))

            divide_transitions[term_idx] = divide_list

        # Statistiques sur les transitions
        avg_multiply = np.mean([len(v) for v in multiply_transitions.values()])
        avg_divide = np.mean([len(v) for v in divide_transitions.values()])
        print(f"  Average of multiply transitions: {avg_multiply:.2f}")
        print(f"  Average of divide transitions: {avg_divide:.2f}")

        return multiply_transitions, divide_transitions

    def generate_monomials(self, k: int) -> list:
        """
        Generate all monomials of degree at most k with num vars variables (constant included)
        
        Args:
            k: integer
            
        Returns:
            all monomials of degree at most k with num vars variables (constant included).
        """
        monomials = []
        for lst in product(range(k + 1), repeat=self.num_vars):
            if sum(lst) <= k:
                monomials.append(list(lst))
        return monomials

    def generate_all_targets(self, targets_terms: list) -> list:
        """
        Generate all polynomials that include the targets, by multiplying the targets with all allowed monomials.
        
        Args:
            target_terms: list of np.ndarray
            
        Returns:
            all polynomials that include the targets.
        """
        all_targets = []
        for tar in targets_terms:
            max_order_target = np.max(np.sum(tar,axis=1))
            diff = self.max_degree - max_order_target
            monomials_to_add = self.generate_monomials(diff)

            # Generate target x monomials
            all_possible_targets = []
            nb_terms = tar.shape[0]
            for monom in monomials_to_add:
                to_add = np.tile(monom, (nb_terms, 1))
                all_possible_targets.append(tar+to_add)

            all_targets.append(np.array(all_possible_targets))
        
        return all_targets

    def test_targets_inclusion(self, pol_terms: list, targets_terms: list) -> bool:
        """
        Test if pol_terms include any of the targets.
        
        Args:
            pol_terms: polynomial to be tested as a list of non-zero power terms (as in term_powers)
            target_terms: list of np.ndarray containing all target polynomials (generated with generate_all_targets)
            
        Returns:
            True if pol_terms include any of the targets, False else.
        """
        pol_tuple = set(map(tuple, pol_terms))
        
        for tar in targets_terms:
            inclusion_temp = np.prod([tuple(monom) in pol_tuple for monom in tar])
            if inclusion_temp:
                return True
            else:
                continue
        return False

    def initialize_sparse_polynomials(self, n_particles: int) -> np.ndarray:
        """
        Initialize a batch of sparse polynomials with a distribution
        favoring polynomials with more monomials.

        Args:
            n_particles (int): Number of particles (polynomials) to generate.

        Returns:
            np.ndarray: Batch of coefficients of shape (n_particles, num_terms).
        """
        particles = np.zeros((n_particles, self.num_terms))

        max_monomials = self.max_num_monomials

        n_mon_to_max_order = comb(self.max_degree+self.num_vars,self.max_degree) # total number of possible monomials 
        n_mon_all_coeff = []
        n_tot_pol = 0
        for i in range(1,self.max_num_monomials+1):
            n_mon_i_coeff = comb(n_mon_to_max_order,i)
            n_tot_pol += n_mon_i_coeff
            n_mon_all_coeff.append(n_mon_i_coeff)
        
        n_mon_all_coeff = np.array(n_mon_all_coeff) 
        proba_ = n_mon_all_coeff/n_tot_pol
        
        np.random.seed = self.random_seed
        for i in range(n_particles):
            # Choisir le nombre de monômes selon la distribution de poids
            # Le +1 garantit qu'il y ait au moins 1 monôme
            num_monomials = np.random.choice(
                range(1, max_monomials + 1), 
                p=proba_
            )
            # Choisir aléatoirement les indices des coefficients non nuls
            nonzero_indices = np.random.choice(
                self.num_terms, 
                size=num_monomials, 
                replace=False
            )
            # Générer des valeurs aléatoires pour ces coefficients
            # Utiliser une distribution qui évite les valeurs proches de zéro
            indices_ = 4*np.random.rand(num_monomials)-2
            particles[i, nonzero_indices] = indices_
        
        # Normaliser les particules initiales

        #return self.normalize_polynomial_batch(particles)
        return particles 
    
    def evaluate_polynomial_batch(self, coefficients_batch: np.ndarray) -> np.ndarray:
        """
        Evaluate a batch of polynomials on the given data
        """
        return evaluate_polynomial_batch_fast(self.term_matrix, coefficients_batch)
    
    def compute_loss_batch(self, coefficients_batch: np.ndarray) -> np.ndarray:
        """
        Compute the loss function for a batch of coefficients.
        Here, the loss is the mean squared error of P(x_data) relative to 0.

        Args:
            coefficients_batch (np.ndarray): Batch of coefficients of shape (n_particles, num_terms).

        Returns:
            np.ndarray: Vector of losses of shape (n_particles,).
        """

        return evaluate_polynomial_batch_fast(self.term_matrix, coefficients_batch) + self.regularisation_factor / sum_coeffients_batch_fast(coefficients_batch)
        
    def compute_prior_log_prob_batch(self, 
                                     coefficients_batch: np.ndarray, 
                                     sparsity_factor: float = 1.0, 
                                     degree_bias: float = 0.5) -> np.ndarray:
        """
        Compute the log prior probability for a batch of coefficients.
        Encourages sparsity with an L1 prior and biases toward higher-degree terms.

        Args:
            coefficients_batch (np.ndarray): Batch of coefficients of shape (n_particles, num_terms).
            sparsity_factor (float): Controls the sparsity strength.
            degree_bias (float): Controls the bias toward higher-degree terms.

        Returns:
            np.ndarray: Vector of log prior probabilities of shape (n_particles,).
        """
        # Calculer le degré de chaque terme
        term_degrees = np.sum(self.term_powers, axis=1)
        
        # Normaliser les degrés pour qu'ils soient entre 0 et 1
        max_possible_degree = self.max_degree
        normalized_degrees = term_degrees / max_possible_degree
        
        # Créer une matrice de poids pour les coefficients en fonction de leur degré
        # Plus le degré est élevé, plus le poids est faible pour favoriser ces termes
        degree_weights = 1.0 - degree_bias * normalized_degrees
        
        # Prior L1 qui favorise la parcimonie mais avec des poids ajustés par degré
        weighted_coeffs = np.abs(coefficients_batch) * degree_weights
        l1_norms = np.sum(weighted_coeffs, axis=1)
        
        return -sparsity_factor * l1_norms
 
    def mh_proposal_batch(self,
                          current_coeffs_batch: np.ndarray,
                          i: int,
                          n_iter: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propose new values for a batch of coefficients using the specialized Metropolis-Hastings kernel.
        Five types of operations are possible:
        1. Add a new monomial (coefficient 0 → nonzero)
        2. Remove an existing monomial (coefficient nonzero → 0)
        3. Modify an existing nonzero coefficient
        4. Multiply a monomial by a variable (increase variable degree)
        5. Divide a monomial by a variable (decrease variable degree)

        Args:
            current_coeffs_batch (np.ndarray): Current batch of coefficients (n_particles, num_terms).

        Returns:
            Tuple:
                - np.ndarray: Batch of proposed new coefficients (n_particles, num_terms)
                - np.ndarray: Proposal log-ratios log(q(z_old|z_new)/q(z_new|z_old)) (n_particles,)
        """
        n_particles, n_terms = current_coeffs_batch.shape
        proposed_coeffs = current_coeffs_batch.copy()
        log_proposal_ratios = np.zeros(n_particles)
        np.random.seed = self.random_seed

        # Calculer les probabilités d'opération
        prob_multiply = self.prob_multiply(i, n_iter)
        prob_divide = self.prob_divide(i, n_iter)
        prob_add = self.prob_add(i, n_iter)
        prob_remove = self.prob_remove(i, n_iter)
        prob_modify = self.prob_modify(i, n_iter)

        prob_tot = prob_multiply + prob_divide + prob_add + prob_remove + prob_modify
        probs = np.array([prob_add, prob_remove, prob_modify, prob_multiply, prob_divide]) / prob_tot

        # Masques pour identifier les coefficients nuls/non-nuls
        nonzero_mask = (current_coeffs_batch != 0)
        zero_mask = ~nonzero_mask
        num_nonzero = np.sum(nonzero_mask, axis=1)
        num_zero = n_terms - num_nonzero

        # Tirer l'opération pour chaque particule
        operations = np.random.choice(5, size=n_particles, p=probs)

        # === OPÉRATION 2: MODIFY ===
        # Pas de if : la fonction gère les masques vides
        mask_modify = (operations == 2) & (num_nonzero > 0)
        perturbations = np.random.normal(0, self.sigma_proposal(i, n_iter), size=n_particles)
        proposed_coeffs = apply_modify_operation_vectorized(
            proposed_coeffs, mask_modify, nonzero_mask, perturbations
        )
        log_proposal_ratios = np.where(mask_modify, 0.0, log_proposal_ratios)

        # === OPÉRATION 0: ADD ===
        mask_add = (operations == 0) & (num_zero > 0) & (num_nonzero < self.max_num_monomials)
        new_values = np.random.normal(0, self.sigma_proposal(i, n_iter), size=n_particles)
        proposed_coeffs = apply_add_operation_vectorized(
            proposed_coeffs, mask_add, zero_mask, new_values
        )
        # Log ratios simplifiés (approximation)
        forward_prob = probs[0] / np.maximum(num_zero, 1)
        reverse_prob = probs[1] / np.maximum(num_nonzero + 1, 1)
        add_log_ratio = np.log(reverse_prob + 1e-10) - np.log(forward_prob + 1e-10)
        log_proposal_ratios = np.where(mask_add, add_log_ratio, log_proposal_ratios)

        # === OPÉRATION 1: REMOVE ===
        mask_remove = (operations == 1) & (num_nonzero > 0)
        proposed_coeffs = apply_remove_operation_vectorized(
            proposed_coeffs, mask_remove, nonzero_mask
        )
        # Log ratios simplifiés
        forward_prob = probs[1] / np.maximum(num_nonzero, 1)
        reverse_prob = probs[0] / np.maximum(num_zero + 1, 1)
        remove_log_ratio = np.log(reverse_prob + 1e-10) - np.log(forward_prob + 1e-10)
        log_proposal_ratios = np.where(mask_remove, remove_log_ratio, log_proposal_ratios)

        # === OPÉRATION 3: MULTIPLY ===
        mask_multiply = (operations == 3) & (num_nonzero > 0)
        # Mettre à jour le nonzero_mask basé sur proposed_coeffs
        current_nonzero_mask = (proposed_coeffs != 0)

        proposed_coeffs, success_mult = apply_multiply_divide_operation_vectorized(
            proposed_coeffs, mask_multiply, current_nonzero_mask,
            self.multiply_lookup_matrix, self.multiply_lookup_valid,
            is_multiply=True
        )

        # Fallback pour les échecs : appliquer MODIFY
        failed_multiply = mask_multiply & ~success_mult
        perturbations_mult = np.random.normal(0, self.sigma_proposal(i, n_iter), size=n_particles)
        proposed_coeffs = apply_modify_operation_vectorized(
            proposed_coeffs, failed_multiply, current_nonzero_mask, perturbations_mult
        )

        log_proposal_ratios = np.where(mask_multiply, 0.0, log_proposal_ratios)

        # === OPÉRATION 4: DIVIDE ===
        mask_divide = (operations == 4) & (num_nonzero > 0)
        # Mettre à jour le nonzero_mask basé sur proposed_coeffs
        current_nonzero_mask = (proposed_coeffs != 0)

        proposed_coeffs, success_div = apply_multiply_divide_operation_vectorized(
            proposed_coeffs, mask_divide, current_nonzero_mask,
            self.divide_lookup_matrix, self.divide_lookup_valid,
            is_multiply=False
        )

        # Fallback pour les échecs : appliquer MODIFY
        failed_divide = mask_divide & ~success_div
        perturbations_div = np.random.normal(0, self.sigma_proposal(i, n_iter), size=n_particles)
        proposed_coeffs = apply_modify_operation_vectorized(
            proposed_coeffs, failed_divide, current_nonzero_mask, perturbations_div
        )

        log_proposal_ratios = np.where(mask_divide, 0.0, log_proposal_ratios)

        return proposed_coeffs, log_proposal_ratios
    
    def compute_log_target_batch(self, coefficients_batch: np.ndarray, delta_beta: float, 
                                sparsity_factor: float = 1.0,
                                degree_bias: float = 0.5) -> np.ndarray:
        """
        Compute the log of the target distribution for a batch of coefficients.

        Args:
            coefficients_batch (np.ndarray): Batch of coefficients (n_particles, num_terms).
            temperature (float): Current temperature.
            sparsity_factor (float): Sparsity factor.

        Returns:
            np.ndarray: Vector of target log-probabilities (n_particles,).
        """
        loss_batch = self.compute_loss_batch(coefficients_batch)
        
        # Réactiver le prior pour avoir une meilleure distribution de poids
        prior_log_prob_batch = self.compute_prior_log_prob_batch(coefficients_batch, sparsity_factor, degree_bias)
        
        # Appliquer une températion progressive à la perte et au prior
        return -loss_batch * delta_beta + prior_log_prob_batch #* temperature
    
    def local_search(self, coefficients, n_steps=100, use_reg=False):
        """Adjust the polynomial with a local search"""
        best_coeffs = coefficients.copy()
        np.random.seed = self.random_seed

        if use_reg:
            best_loss = self.compute_loss_batch(best_coeffs.reshape(1, -1))[0]
        else:
            best_loss = evaluate_polynomial_batch_fast(self.term_matrix, best_coeffs.reshape(1, -1))[0]
        best_loss = evaluate_polynomial_batch_fast(self.term_matrix, best_coeffs.reshape(1, -1))[0]

        for _ in range(n_steps):
            # Modification légère d'un coefficient non nul
            nonzero_indices = np.nonzero(best_coeffs)[0]
            if len(nonzero_indices) > 0:
                idx = np.random.choice(nonzero_indices)
                new_coeffs = best_coeffs.copy()
                new_coeffs[idx] += np.random.normal(0, 0.01)  # Petit pas

                # Évaluer la nouvelle perte
                if use_reg:
                    new_loss = self.compute_loss_batch(new_coeffs.reshape(1, -1))
                else:
                    new_loss = evaluate_polynomial_batch_fast(self.term_matrix, new_coeffs.reshape(1, -1))
                
                # Accepter si amélioration
                if new_loss < best_loss:
                    best_coeffs = new_coeffs
                    best_loss = new_loss

        return best_coeffs, best_loss
    
    def local_search_batch(self, coefficients_batch, n_steps=100, std = 0.01, verbose = True, use_reg=False):
        """
        Refine a batch of polynomials using a local search.

        Args:
            coefficients_batch (np.ndarray): Batch of coefficients (n_particles, num_terms).
            n_steps (int): Number of local search steps.
            use_reg (bool): If True, compute loss with compute_loss_batch; if False, use evaluate_polynomial_batch_fast.

        Returns:
            Tuple:
                - np.ndarray: Batch of improved coefficients (n_particles, num_terms).
                - np.ndarray: Vector of improved losses (n_particles,).
        """
        coefficients_batch_ = coefficients_batch.copy()
        np.random.seed = self.random_seed

        new_losses = []

        for _ in range(n_steps):
            if verbose : 
                print(f"Step {_}/{n_steps} of local search")
            
            if use_reg:
                prev_losses = self.compute_loss_batch(coefficients_batch_)
            else:
                prev_losses = evaluate_polynomial_batch_fast(self.term_matrix,coefficients_batch_)

            non_zero_mask = (coefficients_batch_ != 0)

            non_zero_indices = [np.flatnonzero(row) for row in non_zero_mask]

            chosen_col_indices = np.array([np.random.choice(cols) for cols in non_zero_indices])

            row_indices = np.arange(coefficients_batch_.shape[0])

            coefficients_batch_copy = coefficients_batch_.copy()
            coefficients_batch_copy[row_indices, chosen_col_indices] += np.random.normal(loc=0, scale=std, size=coefficients_batch_copy.shape[0])

            if use_reg:
                new_losses = self.compute_loss_batch(coefficients_batch_copy)
            else:
                new_losses = evaluate_polynomial_batch_fast(self.term_matrix,coefficients_batch_copy)


            mask = new_losses < prev_losses

            coefficients_batch_[mask] = coefficients_batch_copy[mask]
        
        return coefficients_batch_, new_losses
    
    def multinomial_resample(self, weights, n_samples):
        """
        Resampling multinomial
        
        Args:
            weights: normalised particle weights
            n_samples: number of samples to generate
        
        Returns:
            indices: resampled particle indices
        """
        np.random.seed = self.random_seed
        return np.random.choice(len(weights), size=n_samples, p=weights, replace=True)

    def run_annealing_is(
            self, 
            n_iter: int, 
            n_particles: int, 
            n_iter_exploit: int = 0, 
            std_exploit: float = 0.01,
            sparsity_factor: float = 1.0,
            degree_bias: float = 0.5,
            verbose: bool = True,
            target_acc_rate: Callable[[int, int], float] = lambda x,y: 0.3,
            adaptation_strength: Callable[[int,int], float] = lambda x,y: 0.5,
            beta_schedule: Callable[[int, int], float] = lambda i,n_iter: 1e-10 + (i/n_iter)**2,
            adaptative_temp: bool = True,
            analysis: bool = False,
            target_terms: list = [],
        )-> Tuple[np.ndarray, float, np.ndarray, np.ndarray, list]:
        """
        Run the vectorized Annealing Importance Sampling algorithm.

        Args:
            n_iter (int): Number of annealing iterations.
            n_particles (int): Number of particles.
            n_iter_exploit (int): Number of exploitation steps with p_mod = 1.
            std_exploit (float): Std to be used in the local search.
            sparsity_factor (float): Sparsity factor for the prior.
            verbose (bool): If True, print information during execution.
            analysis (bool): If True, retain additional data for analysis.
            target_terms (list): List of nonzero terms for each target polynomial.

        Returns:
            Tuple:
                - Best coefficients found
                - Corresponding loss
                - All final particles
                - Their losses
                - History of the first 10 particles
        """
        start_time = time.time()
        np.random.seed = self.random_seed
        
        # Initialiser les particules avec des polynômes parcimonieux
        particles = self.initialize_sparse_polynomials(n_particles)
        # Sauvegarder les particules de l'étape précédente pour le calcul des poids
        particles_prev = particles.copy()

        # Initialize the count of polynomials containing the target polynomials
        if analysis:
            nb_target = len(target_terms)
            all_targets = self.generate_all_targets(target_terms)

            count_good_pol = np.zeros([n_iter,nb_target])
            for pol_temp in particles:
                # Identify the nonzero coefficients, ie the nonzero monomials
                nonzero_terms_temp = np.nonzero(pol_temp)[0]
                # Select for each nonzero monomial the variables with non-zero power.
                powers_temp = [self.term_powers[term_idx] for term_idx in nonzero_terms_temp]
                for iii in range(nb_target):
                    count_good_pol[0,iii] += self.test_targets_inclusion(powers_temp,all_targets[iii])

        normalized_weights = np.ones(n_particles) / n_particles
        unnormalized_weights = np.ones(n_particles)
        
        best_loss = float('inf')
        best_coeffs = None
        
        # Liste pour sauvegarder les 10 premières particules à chaque étape
        particles_history = []
        
        acceptance_history = deque(maxlen=5)

        beta_history = [beta_schedule(0, n_iter)]
        best_loss_history = []
        ess_history = []
        acceptance_history_all = []

        log_proposal_ratios = 1

        if adaptative_temp == True: 
            beta_current = beta_schedule(1, n_iter)
            beta_prev = beta_schedule(0, n_iter)

        for i in range(1,n_iter):
                
            if adaptative_temp == False: 
                beta_prev = beta_schedule(i-1, n_iter)
                beta_current = beta_schedule(i, n_iter)
            else: 
                if i > 1 and acceptance_history:
                    beta_prev = beta_current
                    avg_acceptance_rate = np.mean(acceptance_history)
                    
                    # Ajuster beta selon l'écart par rapport au taux cible
                    if avg_acceptance_rate < target_acc_rate(i,n_iter) - 0.05:
                        # Taux trop bas -> diminuer beta (augmenter température)
                        beta_current *= (1.0 - adaptation_strength(i,n_iter))
                    elif avg_acceptance_rate > target_acc_rate(i,n_iter) + 0.05:
                        # Taux trop haut -> augmenter beta (diminuer température)
                        beta_current *= (1.0 + adaptation_strength(i,n_iter))
                else:
                    # Au début, suivre un schedule traditionnel pour démarrer
                    beta_current = beta_schedule(i, n_iter)

            beta_history.append(beta_current)
            # Sauvegarder les 10 premières particules à cette étape
            particles_history.append(particles[:10].copy())
            
            # Étape de repondération - vectorisée
            # Calculer γ_n(z_n^k) et γ_{n-1}(z_{n-1}^k) séparément
            log_gamma_n_current = self.compute_log_target_batch(particles, beta_current, sparsity_factor, degree_bias)
            log_gamma_prev_prev = self.compute_log_target_batch(particles_prev, beta_prev, sparsity_factor, degree_bias)
            
            # Calculer le ratio des densités : γ_n(z_n^k) / γ_{n-1}(z_{n-1}^k)
            incremental_log_weights = log_gamma_n_current - log_gamma_prev_prev + log_proposal_ratios

            unnormalized_weights_prev = unnormalized_weights.copy()
            
            # Update unnormalized weights (vectorized, log-space for stability)
            # Avoids log(0) and handles potential -inf in incremental_log_weights
            valid_log_weights = np.isfinite(incremental_log_weights)
            if np.any(valid_log_weights):
                max_log_w_inc = np.max(incremental_log_weights[valid_log_weights])
            else:
                max_log_w_inc = 0 # All weights will become 0

            # Calculate relative weights safely
            relative_incremental_weights = np.zeros(n_particles)
            relative_incremental_weights[valid_log_weights] = np.exp(
                incremental_log_weights[valid_log_weights] - max_log_w_inc
            )

            # Update weights (vectorized)
            unnormalized_weights = unnormalized_weights_prev * relative_incremental_weights
            
            # --- Normalization and ESS Calculation (Already Vectorized) ---
            W_sum = np.sum(unnormalized_weights)

            if W_sum <= 0 or not np.isfinite(W_sum): # Check for <= 0 for robustness
                print("  Warning: Total weight sum is zero, negative or invalid. Assigning equal weights.")
                unnormalized_weights.fill(1.0) # Assign equal non-zero weights
                W_sum = n_particles
                normalized_weights = unnormalized_weights / W_sum
            else:
                normalized_weights = unnormalized_weights / W_sum
                
            ess = 1.0 / np.sum(normalized_weights**2)
            
            # Rééchantillonnage si nécessaire
            if ess < n_particles / 2:
                if verbose:
                    print(f"ESS = {ess:.2f} : resampling needed")
                indices = np.random.choice(n_particles, size=n_particles, p=normalized_weights, replace=True)
                particles = particles[indices]
                # IMPORTANT: Après rééchantillonnage, particles_prev doit aussi être mis à jour
                particles_prev = particles_prev[indices]
                normalized_weights = np.ones(n_particles) / n_particles    
                unnormalized_weights = np.ones(n_particles) 
                ess = 1.0 / np.sum(normalized_weights ** 2)       
            
            # Sauvegarder les particules actuelles avant la mutation
            particles_before_mutation = particles.copy()
            
            # Étape de mutation avec Metropolis-Hastings spécialisé
            proposed_particles, log_proposal_ratios = self.mh_proposal_batch(particles, i, n_iter)
            
            # Calculer les probabilités cibles actuelles et proposées - vectorisé
            current_log_probs = self.compute_log_target_batch(particles, beta_current, sparsity_factor, degree_bias)
            proposed_log_probs = self.compute_log_target_batch(proposed_particles, beta_current, sparsity_factor, degree_bias)
            
            # Calculer les ratios d'acceptation - vectorisé
            log_acceptance_ratios = proposed_log_probs - current_log_probs + log_proposal_ratios

            # Décider lesquelles des propositions accepter - vectorisé
            log_random = np.log(np.random.random(n_particles))
            accept_mask = log_random < log_acceptance_ratios

            # Calculer et enregistrer le taux d'acceptation pour cette itération
            acceptance_rate = np.mean(accept_mask)
            
            # Mettre à jour les particules acceptées
            particles[accept_mask] = proposed_particles[accept_mask]
            
            # Mettre à jour particles_prev pour la prochaine itération
            particles_prev = particles_before_mutation.copy()
            
            # Mettre à jour le meilleur résultat
            losses = self.compute_loss_batch(particles)
            min_loss_idx = np.argmin(losses)
            if losses[min_loss_idx] < best_loss:
                best_loss = losses[min_loss_idx]
                best_coeffs = particles[min_loss_idx].copy()

            best_loss_history.append(best_loss)
            acceptance_history.append(acceptance_rate)
            ess_history.append(ess)

            if verbose:
                elapsed = time.time() - start_time
                print(
                    f"Iteration {i}/{n_iter}, "
                    f"\tBeta: {beta_current:.2g}, "
                    f"\tAcceptance rate: {acceptance_rate:.2f}, "
                    f"\tESS: {ess:.2f}/{n_particles}, "
                    f"\tBest loss: {best_loss:.6f},"
                    f"\tElapsed time: {elapsed:.2f}s, "
                    f"\tProb {self.prob_modify(i,n_iter), self.prob_multiply(i,n_iter), self.prob_divide(i,n_iter), self.prob_add(i,n_iter), self.prob_remove(i,n_iter)}, "
                    f"\tBest pol : {self.polynomial_to_string(particles[min_loss_idx])}"
                )
            # Count the polynomials containing the target polynomials
            if analysis:
                for pol_temp in particles:
                    # Identify the nonzero coefficients, ie the nonzero monomials
                    nonzero_terms_temp = np.nonzero(pol_temp)[0]
                    # Select for each nonzero monomial the variables with non-zero power.
                    powers_temp = [self.term_powers[term_idx] for term_idx in nonzero_terms_temp]
                    for iii in range(nb_target):
                        count_good_pol[i,iii] += self.test_targets_inclusion(powers_temp,all_targets[iii])
                            
        particles_after_exploit = self.local_search_batch(particles,n_steps=n_iter_exploit,std = std_exploit)

        if verbose:
            total_time = time.time() - start_time
            print(f"\nTotal execution time: {total_time:.2f}s")
            print(f"Best loss: {best_loss_history[-1]:.6f}")
            print(f"Number of monomials in best solution: {np.count_nonzero(best_coeffs)}")
        if analysis:
            total_time = time.time() - start_time
            return best_coeffs, best_loss_history, particles, ess_history, acceptance_history_all, particles_after_exploit, losses, particles_history, beta_history, normalized_weights, count_good_pol, total_time
        else:
            return best_coeffs, best_loss_history, particles, ess_history, acceptance_history_all, particles_after_exploit, losses, particles_history, beta_history, normalized_weights
    
    def polynomial_to_string(self, coefficients: np.ndarray, threshold: float = 1e-6) -> str:
        """
        Converts a vector of coefficients into a human-readable representation of the polynomial
        
        Args:
            coefficients: Vector of polynomial coefficients
            threshold: value below which a coefficient is set to zero
            
        Returns:
            Polynomial represented as a str
        """
        terms = []
        
        for i, (coeff, powers) in enumerate(zip(coefficients, self.term_powers)):
            if abs(coeff) < threshold:
                continue
                
            # Créer la représentation du terme
            term = ""
            
            # Ajouter le coefficient (si ce n'est pas 1 ou -1 pour x^n)
            if abs(abs(coeff) - 1.0) > threshold or np.sum(powers) == 0:
                term += f"{coeff:.4f}"
            elif coeff < 0:
                term += "-"
                
            # Ajouter les variables avec leurs puissances
            has_var = False
            var_terms = []  # Stocker chaque variable avec sa puissance
            
            for j, power in enumerate(powers):
                if power > 0:
                    has_var = True
                    var_term = f"x{j+1}"
                    if power > 1:
                        var_term += f"^{power}"
                    var_terms.append(var_term)
            
            # Joindre les variables avec des "*"
            if var_terms:
                vars_str = "*".join(var_terms)
                
                # Si c'est juste un coefficient sans variable, ne pas ajouter de "*"
                if abs(abs(coeff) - 1.0) > threshold:
                    term += f"*{vars_str}"
                else:
                    term += vars_str
                    
            terms.append(term)
        
        if not terms:
            return "0"
            
        return " + ".join(terms).replace("+ -", "- ")