import numpy as np
from scipy.special import comb
from typing import Tuple, Callable
import time
from collections import deque
import cProfile
import pstats
import io
from pstats import SortKey
import numba as nb
from functools import reduce
import itertools
from itertools import combinations
from itertools import product

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
    Classe pour l'échantillonnage de polynômes parcimonieux par Annealing Importance Sampling
    avec un noyau Metropolis-Hastings spécialisé pour trouver un polynôme P tel que P(x_data) = 0.
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
                min_coeff_threshold: float = 0.1):
        """
        Initialise le sampler avec les données et les paramètres.
        
        Args:
            data_x: Données d'entrée de forme (n_samples, num_vars)
            max_degree: Degré maximal du polynôme
            num_vars: Nombre de variables
            max_num_monomials: Nombre maximal de monômes (coefficients non nuls) dans un polynôme
            sigma_proposal: Écart-type pour les propositions de modification des coefficients
            prob_add: Probabilité d'ajouter un nouveau monôme
            prob_remove: Probabilité de supprimer un monôme existant
            prob_modify: Probabilité de modifier un coefficient existant
            prob_multiply: Probabilité de multiplier un monome par une des variables
            prob_divide: Probabilité de diviser un monome par une des variables
            regularisation_factor: Facteur dans le Loss pour régulariser la somme des coefficients (L = P(x_i) + regularisation_factor / sum_coeff)
            min_coeff_threshold: Seuil en dessous duquel un coefficient est supprimé (mis à zéro)
            beta_schedule: Fonction qui définit la température pour chaque itération
        """
        self.data_x = data_x
        self.max_degree = max_degree
        self.num_vars = num_vars
        self.max_num_monomials = max_num_monomials
        self.sigma_proposal = sigma_proposal
        self.min_coeff_threshold = min_coeff_threshold  
        
        # Le reste du code reste inchangé
        self.prob_add = prob_add 
        self.prob_remove = prob_remove
        self.prob_modify = prob_modify 
        self.prob_multiply = prob_multiply 
        self.prob_divide = prob_divide 

        self.regularisation_factor = regularisation_factor

        # if beta_schedule is None:
        #     self.beta_schedule = lambda i, n_iter: 1e-10 + (i / n_iter)**2
        # else:
        #     self.beta_schedule = beta_schedule
            
        self.num_terms = self._compute_num_terms()
        self.term_powers = self._generate_term_powers()
        self.term_matrix = self._precompute_term_matrix()

        # Pré-calcul des transitions multiply/divide pour vectorisation GPU
        self.multiply_transitions, self.divide_transitions = self._precompute_transitions()

        # Construire les matrices de lookup pour version pure vectorisée
        from vectorized_helpers import build_transition_lookup_matrix
        max_trans_mult = max([len(v) for v in self.multiply_transitions.values()]) if self.multiply_transitions else 1
        max_trans_div = max([len(v) for v in self.divide_transitions.values()]) if self.divide_transitions else 1
        self.multiply_lookup_matrix, self.multiply_lookup_valid = build_transition_lookup_matrix(
            self.multiply_transitions, max_trans_mult
        )
        self.divide_lookup_matrix, self.divide_lookup_valid = build_transition_lookup_matrix(
            self.divide_transitions, max_trans_div
        )

        print(f"Nombre de termes possibles dans le polynôme: {self.num_terms}")
    
    def apply_coefficient_threshold(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Supprime les coefficients plus petits que min_coeff_threshold en les mettant à zéro.
        
        Args:
            coefficients: Vecteur des coefficients du polynôme ou batch de coefficients
            
        Returns:
            Vecteur ou batch de coefficients filtrés
        """
        # Copier pour éviter de modifier l'original
        filtered_coeffs = coefficients.copy()
        
        # Mettre à zéro les coefficients dont la valeur absolue est inférieure au seuil
        filtered_coeffs[np.abs(filtered_coeffs) < self.min_coeff_threshold] = 0.0
        
        # Si tous les coefficients sont mis à zéro, restaurer au moins le plus grand
        if len(filtered_coeffs.shape) == 1:  # Cas d'un seul vecteur de coefficients
            if np.all(filtered_coeffs == 0):
                idx_max = np.argmax(np.abs(coefficients))
                filtered_coeffs[idx_max] = coefficients[idx_max]
        else:  # Cas d'un batch de coefficients
            zero_rows = np.all(filtered_coeffs == 0, axis=1)
            if np.any(zero_rows):
                # Pour chaque ligne entièrement nulle, restaurer le coefficient de plus grande valeur absolue
                for i in np.where(zero_rows)[0]:
                    idx_max = np.argmax(np.abs(coefficients[i]))
                    filtered_coeffs[i, idx_max] = coefficients[i, idx_max]
    
        return filtered_coeffs
    
    def normalize_polynomial_batch(self, coefficients_batch: np.ndarray) -> np.ndarray:
        """
        Normalise les coefficients des polynômes pour éviter la solution triviale P=0.
        """
        # Calculer la norme L2 pour chaque ensemble de coefficients
        norms = np.sqrt(np.sum(np.square(coefficients_batch), axis=1, keepdims=True))
        
        # Éviter la division par zéro en ajoutant un epsilon
        norms = np.maximum(norms, 1e-10)
        
        # Normaliser
        normalized = coefficients_batch / norms
        
        # S'assurer qu'aucun polynôme n'est entièrement nul
        zero_polynomials = np.all(np.abs(normalized) < 1e-10, axis=1)
        if np.any(zero_polynomials):
            # Pour chaque polynôme nul, ajouter un coefficient aléatoire
            for i in np.where(zero_polynomials)[0]:
                idx = np.random.randint(0, coefficients_batch.shape[1])
                normalized[i, idx] = 1.0
        
        return normalized
        
    def _compute_num_terms(self) -> int:
        """
        Calcule le nombre de termes dans un polynôme de degré max_degree avec num_vars variables.
        """
        return int(comb(self.num_vars + self.max_degree, self.max_degree)) #- 1
    
    def _generate_term_powers(self) -> np.ndarray:
        """
        Génère tous les termes possibles (puissances des variables) pour un polynôme
        de degré max_degree avec num_vars variables, en excluant le terme constant.
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
        term_matrix = np.ones((n_samples, self.num_terms))#, dtype=np.float32) # pas plus rapide de passer en float32

        for term_idx, powers in enumerate(self.term_powers):
            for var_idx, power in enumerate(powers):
                if power > 0:
                    term_matrix[:, term_idx] *= np.power(self.data_x[:, var_idx], power)

        return term_matrix

    def _precompute_transitions(self) -> Tuple[dict, dict]:
        """
        Pré-calcule toutes les transitions possibles pour les opérations multiply et divide.
        Cela permet de vectoriser ces opérations pour l'implémentation GPU.

        Returns:
            Tuple de deux dictionnaires:
                - multiply_transitions[term_idx] = liste de (var_idx, target_term_idx)
                - divide_transitions[term_idx] = liste de (var_idx, target_term_idx)
        """
        multiply_transitions = {}
        divide_transitions = {}

        print("Pré-calcul des transitions multiply/divide...")

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
        print(f"  Moyenne de transitions multiply par terme: {avg_multiply:.2f}")
        print(f"  Moyenne de transitions divide par terme: {avg_divide:.2f}")

        return multiply_transitions, divide_transitions

    def to_equivalence_class_polynomials(self, pol_in: np.ndarray) -> np.ndarray:
        """
        Renvoie la classe d'équivalence du polynôme pol_in : identifie les variables 
        qui peuvent être factorisées et les élimine.
        
        Args:
            pol_in: coefficients du polynôme d'entrée
            
        Returns:
            Batch de coefficients de forme (num_terms)
        """
        new_pol = np.zeros_like(pol_in)  # Initialiser à zéro au lieu de copier
        
        # Identify the nonzero coefficients, ie the nonzero monomials.
        nonzero_terms = np.nonzero(pol_in)[0]
        
        if len(nonzero_terms) == 0:
            return new_pol
        
        # Select for each nonzero monomial the variables with non-zero power.
        nonzero_powers = [np.nonzero(self.term_powers[term_idx])[0] for term_idx in nonzero_terms]
        
        # Take the intersection of the nonzero_powers, which corresponds to the variables that are common to each monomial.
        common_vars = reduce(np.intersect1d, nonzero_powers)
        
        # Compute the minimal power of each common variable.
        min_power_common_vars = [min([(self.term_powers[term_idx])[var_idx] for term_idx in nonzero_terms]) for var_idx in common_vars]
        
        # Factorization : reduce the power of the common variables by the minimal common power.
        for term_idx in nonzero_terms:
            new_powers = self.term_powers[term_idx].copy()
            
            for idx in range(len(common_vars)):
                new_powers[common_vars[idx]] -= min_power_common_vars[idx]
            
            if (new_powers == np.zeros(self.num_vars)).all(): # CAMILLE : to be removed if we add constant terms
                continue
            else:
                target_term_idx = np.where((self.term_powers == new_powers).all(axis=1))[0][0]
                # Accumuler les coefficients au lieu de les écraser
                new_pol[target_term_idx] += pol_in[term_idx]
        
        return new_pol
    
    def nonzero_terms_targets(self, targets_terms: list) -> list:
        """
        Returns a list containing the position of the nonzero terms of each target polynomials in the term_powers basis
        
        Args:
            target_terms: list of nonzero terms for each target polynomials
            
        Returns:
            list containing the position of the nonzero terms of each target polynomials in the term_powers basis
        """
        return [[np.where((self.term_powers == term).all(axis=1))[0][0] for term in targets_terms[i]] for i in range(len(targets_terms))]
    
    
    # def test_include_targets_partial(self, pol_in: np.ndarray, nonzero_terms_targets: list) -> bool:
    #     """
    #     Returns True if pol_in countains the same monomials of at least one of the target polynomials, False if not.
        
    #     Args:
    #         pol_in: array defining the polynomials
    #         nonzero_terms_targets: list containing the position of the nonzero terms of each target polynomials 
    #         in the term_powers basis, to be computed with self.nonzero_terms_targets
            
    #     Returns:
    #         True if pol_in countains the same monomials of at least one of the target polynomials, False if not.
    #     """
    #     # Non zero terms in the equivalence class of pol_in
    #     nonzero_terms = np.nonzero(self.to_equivalence_class_polynomials(pol_in))[0]

    #     test_targets_inclusion = False

    #     for i in range(len(nonzero_terms_targets)):
    #         try:
    #             test_targets_inclusion |= (np.sort(np.intersect1d(nonzero_terms,nonzero_terms_targets[i])) == np.sort(nonzero_terms_targets[i])).all()
    #         except:
    #             continue
        
    #     return test_targets_inclusion

    # def test_include_targets(self, pol_in: np.ndarray, nonzero_terms_targets: list) -> bool:
    #     """
    #     Returns True if pol_in countains the same monomials of at least one of the target polynomials, False if not.
        
    #     Args:
    #         pol_in: array defining the polynomials
    #         nonzero_terms_targets: list containing the position of the nonzero terms of each target polynomials 
    #         in the term_powers basis, to be computed with self.nonzero_terms_targets
            
    #     Returns:
    #         True if pol_in countains the same monomials of at least one of the target polynomials, False if not.
    #     """
    #     # Non zero terms in the equivalence class of pol_in
    #     nonzero_terms = np.nonzero(self.to_equivalence_class_polynomials(pol_in))[0]

    #     test_targets_inclusion = False

    #     for i in range(len(nonzero_terms_targets)):
    #         try:
    #             test_targets_inclusion |= (np.sort(np.intersect1d(nonzero_terms,nonzero_terms_targets[i])) == np.sort(nonzero_terms_targets[i])).all()
    #         except:
    #             continue
        
    #     return test_targets_inclusion

    
    # def test_include_targets_new(self, pol_in: np.ndarray, nonzero_terms_targets: list) -> bool:
    #     """
    #     Returns True if pol_in contains the same monomials of at least one of the target polynomials 
    #     (possibly with additional terms, or uniformly multiplied by a variable), False if not.
        
    #     Args:
    #         pol_in: array defining the polynomial
    #         nonzero_terms_targets: list containing the position of the nonzero terms of each target polynomial
    #         in the term_powers basis, to be computed with self.nonzero_terms_targets
            
    #     Returns:
    #         True if pol_in matches at least one target polynomial under the specified criteria, False if not.
    #     """
    #     nonzero_terms = np.nonzero(pol_in)[0]
    #     pol_size = len(nonzero_terms)

    #     min_terms_target = min([len(list_temp) for list_temp in nonzero_terms_targets])
    #     diff_len = pol_size - min_terms_target

    #     test_targets_inclusion = False

    #     if diff_len < 0:
    #         return test_targets_inclusion
    #     else :
    #         for nb_to_delete in range(diff_len+1):
    #             for to_be_deleted in list(combinations(range(pol_size),nb_to_delete)):
    #                 nonzero_terms_temp = np.delete(nonzero_terms,to_be_deleted)
    #                 pol_temp = self.to_polynomial(self.term_powers[nonzero_terms_temp],np.ones(len(nonzero_terms_temp)))
    #                 test_targets_inclusion = self.test_include_targets_partial(pol_temp,nonzero_terms_targets)
    #                 if test_targets_inclusion:
    #                     break
    #             if test_targets_inclusion:
    #                 break
    #         return test_targets_inclusion    

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
        # inclusion = []
        # for tar in targets_terms:
        #     inclusion.append(np.prod([tuple(monom) in pol_tuple for monom in tar]))
        # return any(inclusion)
    
        for tar in targets_terms:
            inclusion_temp = np.prod([tuple(monom) in pol_tuple for monom in tar])
            if inclusion_temp:
                return True
            else:
                continue
        return False

    def initialize_sparse_polynomials(self, n_particles: int) -> np.ndarray:
        """
        Initialise un lot de polynômes parcimonieux avec une distribution
        qui favorise les polynômes ayant plus de monômes.
        
        Args:
            n_particles: Nombre de particules (polynômes) à générer
            
        Returns:
            Batch de coefficients de forme (n_particles, num_terms)
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
        Évalue un lot de polynômes sur les données d'entrée.
        """

        return evaluate_polynomial_batch_fast(self.term_matrix, coefficients_batch)
    
    def compute_loss_batch(self, coefficients_batch: np.ndarray) -> np.ndarray:
        """
        Calcule la fonction de perte pour un lot de coefficients.
        Dans ce cas, la perte est l'erreur quadratique moyenne de P(x_data) par rapport à 0.
        
        Args:
            coefficients_batch: Batch de coefficients de forme (n_particles, num_terms)
            
        Returns:
            Vecteur de pertes de forme (n_particles,)
        """

        # Retourner l'erreur moyenne pour chaque particule
        return evaluate_polynomial_batch_fast(self.term_matrix, coefficients_batch) + self.regularisation_factor / sum_coeffients_batch_fast(coefficients_batch)
        #return evaluate_polynomial_batch_fast(self.term_matrix, coefficients_batch) + self.regularisation_factor / sum_coeffients_batch
    
    def compute_prior_log_prob_batch(self, 
                                     coefficients_batch: np.ndarray, 
                                     sparsity_factor: float = 1.0, 
                                     degree_bias: float = 0.5) -> np.ndarray:
        """
        Calcule le logarithme de la probabilité a priori pour un lot de coefficients.
        Favorise la parcimonie avec un prior L1 et introduit un biais en faveur des termes de haut degré.
        
        Args:
            coefficients_batch: Batch de coefficients de forme (n_particles, num_terms)
            sparsity_factor: Facteur qui contrôle la parcimonie
            degree_bias: Facteur qui contrôle le biais en faveur des termes de haut degré
            
        Returns:
            Vecteur de log-probabilités a priori de forme (n_particles,)
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

      
    def mh_proposal_batch_vectorized(self,
                                     current_coeffs_batch: np.ndarray,
                                     i: int,
                                     n_iter: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Version vectorisée de mh_proposal_batch pour GPU.
        Utilise masques et fancy indexing au lieu de boucles for.

        Args:
            current_coeffs_batch: Batch de coefficients actuels (n_particles, num_terms)
            i: Itération courante
            n_iter: Nombre total d'itérations

        Returns:
            Tuple contenant:
                - Batch de nouveaux coefficients proposés (n_particles, num_terms)
                - Log-ratios de proposition (n_particles,)
        """
        n_particles, n_terms = current_coeffs_batch.shape
        proposed_coeffs = current_coeffs_batch.copy()
        log_proposal_ratios = np.zeros(n_particles)

        # Calculer les probabilités d'opération
        prob_multiply = self.prob_multiply(i, n_iter)
        prob_divide = self.prob_divide(i, n_iter)
        prob_add = self.prob_add(i, n_iter)
        prob_remove = self.prob_remove(i, n_iter)
        prob_modify = self.prob_modify(i, n_iter)

        prob_tot = prob_multiply + prob_divide + prob_add + prob_remove + prob_modify
        probs = np.array([prob_add, prob_remove, prob_modify, prob_multiply, prob_divide]) / prob_tot

        # Masques pour identifier les coefficients nuls/non-nuls
        nonzero_mask = (current_coeffs_batch != 0)  # (n_particles, n_terms)
        num_nonzero = np.sum(nonzero_mask, axis=1)  # (n_particles,)
        num_zero = n_terms - num_nonzero

        # Tirer l'opération pour chaque particule
        operations = np.random.choice(5, size=n_particles, p=probs)

        # === OPÉRATION 2: MODIFY (la plus simple) ===
        mask_modify = (operations == 2) & (num_nonzero > 0)
        if np.any(mask_modify):
            # Pour chaque particule qui modifie, choisir un coeff non-nul aléatoirement
            for idx in np.where(mask_modify)[0]:
                nonzero_cols = np.where(nonzero_mask[idx])[0]
                if len(nonzero_cols) > 0:
                    chosen_col = np.random.choice(nonzero_cols)
                    proposed_coeffs[idx, chosen_col] += np.random.normal(0, self.sigma_proposal(i, n_iter))
                    log_proposal_ratios[idx] = 0.0  # Symétrique

        # === OPÉRATION 0: ADD ===
        mask_add = (operations == 0) & (num_zero > 0) & (num_nonzero < self.max_num_monomials)
        if np.any(mask_add):
            for idx in np.where(mask_add)[0]:
                zero_cols = np.where(~nonzero_mask[idx])[0]
                if len(zero_cols) > 0:
                    chosen_col = np.random.choice(zero_cols)
                    proposed_coeffs[idx, chosen_col] = np.random.normal(0, self.sigma_proposal(i, n_iter))

                    # Calcul du log_proposal_ratio (simplifié)
                    forward_prob = probs[0] / num_zero[idx]
                    new_nonzero = num_nonzero[idx] + 1
                    reverse_prob = probs[1] / new_nonzero if new_nonzero > 0 else 0
                    log_proposal_ratios[idx] = np.log(reverse_prob + 1e-10) - np.log(forward_prob + 1e-10)

        # === OPÉRATION 1: REMOVE ===
        mask_remove = (operations == 1) & (num_nonzero > 0)
        if np.any(mask_remove):
            for idx in np.where(mask_remove)[0]:
                nonzero_cols = np.where(nonzero_mask[idx])[0]
                if len(nonzero_cols) > 0:
                    chosen_col = np.random.choice(nonzero_cols)
                    proposed_coeffs[idx, chosen_col] = 0.0

                    # Calcul du log_proposal_ratio (simplifié)
                    forward_prob = probs[1] / num_nonzero[idx]
                    new_zero = num_zero[idx] + 1
                    reverse_prob = probs[0] / new_zero if new_zero > 0 else 0
                    log_proposal_ratios[idx] = np.log(reverse_prob + 1e-10) - np.log(forward_prob + 1e-10)

        # === OPÉRATION 3: MULTIPLY ===
        mask_multiply = (operations == 3) & (num_nonzero > 0)
        if np.any(mask_multiply):
            for idx in np.where(mask_multiply)[0]:
                nonzero_cols = np.where(nonzero_mask[idx])[0]
                if len(nonzero_cols) == 0:
                    # Fallback: modify
                    chosen_col = np.random.randint(n_terms)
                    proposed_coeffs[idx, chosen_col] += np.random.normal(0, self.sigma_proposal(i, n_iter))
                    log_proposal_ratios[idx] = 0.0
                    continue

                # Choisir un terme source aléatoire parmi les non-nuls
                src_col = np.random.choice(nonzero_cols)

                # Récupérer les transitions possibles pour ce terme
                valid_transitions = self.multiply_transitions[src_col]

                # Filtrer pour ne garder que les transitions vers des colonnes avec coeff nul
                valid_transitions_filtered = [(var_idx, tgt) for var_idx, tgt in valid_transitions
                                              if current_coeffs_batch[idx, tgt] == 0]

                if len(valid_transitions_filtered) == 0:
                    # Fallback: modify un coefficient existant
                    chosen_col = np.random.choice(nonzero_cols)
                    proposed_coeffs[idx, chosen_col] += np.random.normal(0, self.sigma_proposal(i, n_iter))
                    log_proposal_ratios[idx] = 0.0
                    continue

                # Choisir une transition aléatoire
                var_idx, tgt_col = valid_transitions_filtered[np.random.randint(len(valid_transitions_filtered))]

                # Déplacer le coefficient
                coeff_value = proposed_coeffs[idx, src_col]
                proposed_coeffs[idx, src_col] = 0.0
                proposed_coeffs[idx, tgt_col] = coeff_value

                # Log ratio (simplifié)
                log_proposal_ratios[idx] = 0.0  # Approximation

        # === OPÉRATION 4: DIVIDE ===
        mask_divide = (operations == 4) & (num_nonzero > 0)
        if np.any(mask_divide):
            for idx in np.where(mask_divide)[0]:
                nonzero_cols = np.where(nonzero_mask[idx])[0]
                if len(nonzero_cols) == 0:
                    # Fallback: modify
                    chosen_col = np.random.randint(n_terms)
                    proposed_coeffs[idx, chosen_col] += np.random.normal(0, self.sigma_proposal(i, n_iter))
                    log_proposal_ratios[idx] = 0.0
                    continue

                # Choisir un terme source aléatoire parmi les non-nuls
                src_col = np.random.choice(nonzero_cols)

                # Récupérer les transitions possibles pour ce terme
                valid_transitions = self.divide_transitions[src_col]

                # Filtrer pour ne garder que les transitions vers des colonnes avec coeff nul
                valid_transitions_filtered = [(var_idx, tgt) for var_idx, tgt in valid_transitions
                                              if current_coeffs_batch[idx, tgt] == 0]

                if len(valid_transitions_filtered) == 0:
                    # Fallback: modify un coefficient existant
                    chosen_col = np.random.choice(nonzero_cols)
                    proposed_coeffs[idx, chosen_col] += np.random.normal(0, self.sigma_proposal(i, n_iter))
                    log_proposal_ratios[idx] = 0.0
                    continue

                # Choisir une transition aléatoire
                var_idx, tgt_col = valid_transitions_filtered[np.random.randint(len(valid_transitions_filtered))]

                # Déplacer le coefficient
                coeff_value = proposed_coeffs[idx, src_col]
                proposed_coeffs[idx, src_col] = 0.0
                proposed_coeffs[idx, tgt_col] = coeff_value

                # Log ratio (simplifié)
                log_proposal_ratios[idx] = 0.0  # Approximation

        return proposed_coeffs, log_proposal_ratios

    def mh_proposal_batch_pure_vectorized(self,
                                           current_coeffs_batch: np.ndarray,
                                           i: int,
                                           n_iter: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Version 100% vectorisée SANS AUCUNE BOUCLE FOR.
        Utilise les fonctions helper de vectorized_helpers.py.
        Prête pour conversion JAX.

        Args:
            current_coeffs_batch: Batch de coefficients actuels (n_particles, num_terms)
            i: Itération courante
            n_iter: Nombre total d'itérations

        Returns:
            Tuple contenant:
                - Batch de nouveaux coefficients proposés (n_particles, num_terms)
                - Log-ratios de proposition (n_particles,)
        """
        from vectorized_helpers import (
            apply_modify_operation_vectorized,
            apply_add_operation_vectorized,
            apply_remove_operation_vectorized,
            apply_multiply_divide_operation_vectorized
        )

        n_particles, n_terms = current_coeffs_batch.shape
        proposed_coeffs = current_coeffs_batch.copy()
        log_proposal_ratios = np.zeros(n_particles)

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

    def mh_proposal_batch(self,
                          current_coeffs_batch: np.ndarray,
                          i: int,
                          n_iter: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propose de nouvelles valeurs pour un lot de coefficients selon le noyau MH spécialisé.
        Cinq types d'opérations sont possibles:
        1. Ajouter un nouveau monôme (coefficient 0 -> non 0)
        2. Supprimer un monôme existant (coefficient non 0 -> 0)
        3. Modifier un coefficient existant non nul
        4. Multiplier un monôme par une variable (augmenter le degré d'une variable)
        5. Diviser un monôme par une variable (diminuer le degré d'une variable)
        
        Args:
            current_coeffs_batch: Batch de coefficients actuels (n_particles, num_terms)
            
        Returns:
            Tuple contenant:
                - Batch de nouveaux coefficients proposés (n_particles, num_terms)
                - Log-ratios de proposition log(q(z_old|z_new)/q(z_new|z_old)) (n_particles,)
        """
        n_particles, n_terms = current_coeffs_batch.shape
        proposed_coeffs = current_coeffs_batch.copy()
        log_proposal_ratios = np.zeros(n_particles)
        
        # Nouvelles probabilités incluant multiplier et diviser
        prob_multiply = self.prob_multiply(i,n_iter)
        prob_divide = self.prob_divide(i,n_iter)
        prob_add = self.prob_add(i,n_iter)
        prob_remove = self.prob_remove(i,n_iter)
        prob_modify = self.prob_modify(i,n_iter)

        prob_tot = prob_multiply + prob_divide + prob_add + prob_remove + prob_modify

        prob_multiply_adj = prob_multiply/prob_tot
        prob_divide_adj = prob_divide/prob_tot
        prob_add_adj = prob_add/prob_tot
        prob_remove_adj = prob_remove/prob_tot
        prob_modify_adj = prob_modify/prob_tot

        for i in range(n_particles):
            # Identifier les indices des coefficients nuls et non nuls
            nonzero_indices = np.nonzero(current_coeffs_batch[i])[0]
            zero_indices = np.where(current_coeffs_batch[i] == 0)[0]
            
            num_nonzero = len(nonzero_indices)
            num_zero = len(zero_indices)
            
            # Déterminer l'opération à effectuer (maintenant 5 opérations)
            operation_probs = np.array([prob_add_adj, prob_remove_adj, prob_modify_adj, prob_multiply_adj, prob_divide_adj])
            
            # Ajuster les probabilités en fonction des contraintes
            if num_nonzero == 0:  # Si aucun coefficient non nul, on doit ajouter
                operation_probs = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
            elif num_nonzero >= self.max_num_monomials:  # Si max atteint, pas d'ajout
                # Normaliser les probs de remove, modify, multiply et divide
                remaining_sum = operation_probs[1:].sum()
                operation_probs = np.array([0.0, 
                                        operation_probs[1] / remaining_sum,
                                        operation_probs[2] / remaining_sum,
                                        operation_probs[3] / remaining_sum,
                                        operation_probs[4] / remaining_sum])
            elif num_zero == 0:  # Si aucun coefficient nul, pas d'ajout
                # Normaliser les probs de remove, modify, multiply et divide
                remaining_sum = operation_probs[1:].sum()
                operation_probs = np.array([0.0, 
                                        operation_probs[1] / remaining_sum,
                                        operation_probs[2] / remaining_sum,
                                        operation_probs[3] / remaining_sum,
                                        operation_probs[4] / remaining_sum])
            
            # Tentatives pour trouver une opération valide
            max_attempts = 10  # Limiter le nombre de tentatives pour éviter les boucles infinies
            for attempt in range(max_attempts):
                operation = np.random.choice(5, p=operation_probs)
                
                # 0: Ajouter un nouveau monôme
                if operation == 0 and num_zero > 0:
                    # Choisir un coefficient nul au hasard
                    idx = np.random.choice(zero_indices)
                    # Lui donner une valeur non nulle
                    proposed_coeffs[i, idx] = np.random.normal(0, self.sigma_proposal(i,n_iter))
                    
                    # Calculer le ratio de proposition
                    forward_prob = operation_probs[0] / num_zero
                    
                    # Pour la reverse move
                    new_nonzero_count = num_nonzero + 1
                    new_zero_count = num_zero - 1
                    
                    # Cas particuliers pour les probabilités de reverse
                    if new_nonzero_count >= self.max_num_monomials:
                        # Pas d'ajout possible pour la reverse
                        remaining_sum = operation_probs[1:].sum()
                        reverse_probs = np.array([0.0, 
                                                operation_probs[1] / remaining_sum,
                                                operation_probs[2] / remaining_sum,
                                                operation_probs[3] / remaining_sum,
                                                operation_probs[4] / remaining_sum])
                    else:
                        reverse_probs = operation_probs
                    
                    reverse_prob = reverse_probs[1] / new_nonzero_count if new_nonzero_count > 0 else 0
                    
                    log_proposal_ratios[i] = np.log(reverse_prob) - np.log(forward_prob) if (forward_prob > 0 and reverse_prob > 0) else 0
                    break  # Opération réussie, sortir de la boucle de tentatives
                
                # 1: Supprimer un monôme existant
                elif operation == 1 and num_nonzero > 0:
                    # Choisir un coefficient non nul au hasard
                    idx = np.random.choice(nonzero_indices)
                    # Le mettre à zéro
                    proposed_coeffs[i, idx] = 0.0
                    
                    # Calculer le ratio de proposition
                    forward_prob = operation_probs[1] / num_nonzero
                    
                    # Pour la reverse move
                    new_nonzero_count = num_nonzero - 1
                    new_zero_count = num_zero + 1
                    
                    # Cas particuliers pour les probabilités de reverse
                    if new_nonzero_count == 0:
                        # Si après suppression il n'y a plus de monômes, seul l'ajout est possible
                        reverse_probs = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
                    else:
                        reverse_probs = operation_probs
                    
                    reverse_prob = reverse_probs[0] / new_zero_count if new_zero_count > 0 else 0
                    
                    log_proposal_ratios[i] = np.log(reverse_prob) - np.log(forward_prob) if (forward_prob > 0 and reverse_prob > 0) else 0
                    break  # Opération réussie, sortir de la boucle de tentatives
                
                # 2: Modifier un coefficient existant
                elif operation == 2 and num_nonzero > 0:
                    # Choisir un coefficient non nul au hasard
                    idx = np.random.choice(nonzero_indices)
                    # Modifier sa valeur
                    proposed_coeffs[i, idx] += np.random.normal(0, self.sigma_proposal(i,n_iter))
                    
                    # Pour une modification gaussienne symétrique, le ratio est 1 (log ratio = 0)
                    log_proposal_ratios[i] = 0.0
                    break  # Opération réussie, sortir de la boucle de tentatives
                
                # 3: Multiplier un monôme par une variable
                elif operation == 3 and num_nonzero > 0:
                    # Choisir un monôme existant au hasard
                    term_idx = np.random.choice(nonzero_indices)
                    # Récupérer les puissances de ce monôme
                    term_powers = self.term_powers[term_idx].copy()
                    
                    # Vérifier si la multiplication est possible (ne dépasse pas le degré max)
                    if np.sum(term_powers) >= self.max_degree:
                        # Degré max atteint, essayer une autre opération
                        continue
                    
                    # Créer une liste de variables utilisables pour la multiplication
                    valid_vars = []
                    for var_idx in range(self.num_vars):
                        # Augmenter la puissance de cette variable pour voir où on atterrirait
                        new_powers = term_powers.copy()
                        new_powers[var_idx] += 1
                        
                        # #Trouver l'index du terme correspondant aux nouvelles puissances
                        # target_term_idx = -1
                        # for idx, powers in enumerate(self.term_powers):
                        #     if np.array_equal(powers, new_powers):
                        #         target_term_idx = idx
                        #         break #old method that used to take more time. Keep it in case problems

                        target_term_idx = np.where((self.term_powers == np.array(new_powers)).all(axis=1))[0][0]
                        
                        # Vérifier si le terme cible existe et a un coefficient nul
                        if target_term_idx != -1 and current_coeffs_batch[i, target_term_idx] == 0:
                            valid_vars.append((var_idx, target_term_idx))

                    # S'il n'y a pas de variables valides pour la multiplication, essayer une autre opération
                    if not valid_vars:
                        continue
                    
                    # Choisir aléatoirement parmi les variables valides
                    var_idx, target_term_idx = valid_vars[np.random.randint(len(valid_vars))]
                    
                    # Déplacer le coefficient vers le nouveau terme
                    coeff_value = proposed_coeffs[i, term_idx]
                    proposed_coeffs[i, term_idx] = 0.0  # Supprimer l'ancien
                    proposed_coeffs[i, target_term_idx] = coeff_value  # Ajouter au nouveau terme
                    
                    # Calculer le ratio de proposition
                    forward_prob = operation_probs[3] / num_nonzero * (1.0 / len(valid_vars))
                    
                    # Pour l'opération inverse (division), il faut trouver la probabilité 
                    # de choisir le terme cible et la variable correspondante
                    new_nonzero_indices = np.nonzero(proposed_coeffs[i])[0] # TODO : pas déjà calculé avec num_zero ? 
                    new_num_nonzero = len(new_nonzero_indices)
                    
                    # Les nouvelles puissances après multiplication
                    new_powers = self.term_powers[target_term_idx]
                    
                    ## Compter le nombre de variables avec puissance > 0 dans le terme cible
                    #divisible_vars = np.sum(new_powers > 0)
                    
                    # Pour chaque variable divisible, vérifier si la division mènerait à un terme non nul
                    valid_reverse_vars = 0
                    for reverse_var_idx in range(self.num_vars):
                        if new_powers[reverse_var_idx] > 0:
                            # Créer les puissances après division
                            reverse_powers = new_powers.copy()
                            reverse_powers[reverse_var_idx] -= 1
                            
                            # # Trouver l'index du terme après division
                            # reverse_term_idx = -1
                            # for idx, powers in enumerate(self.term_powers): 
                            #     if np.array_equal(powers, reverse_powers):
                            #         reverse_term_idx = idx
                            #         break
                            
                            reverse_term_idx = np.where((self.term_powers == np.array(new_powers)).all(axis=1))[0][0]
                            
                            # Vérifier si le terme après division a un coefficient nul
                            if reverse_term_idx != -1 and reverse_term_idx != term_idx and proposed_coeffs[i, reverse_term_idx] == 0:
                                valid_reverse_vars += 1
                    
                    if valid_reverse_vars > 0:
                        prob_choose_target = 1.0 / new_num_nonzero
                        prob_choose_var = 1.0 / valid_reverse_vars
                        reverse_prob = operation_probs[4] * prob_choose_target * prob_choose_var
                    else:
                        reverse_prob = 0.0
                    
                    log_proposal_ratios[i] = np.log(reverse_prob) - np.log(forward_prob) if (forward_prob > 0 and reverse_prob > 0) else 0
                    break  # Opération réussie, sortir de la boucle de tentatives
                
                # 4: Diviser un monôme par une variable
                elif operation == 4 and num_nonzero > 0:
                    # Choisir un monôme existant au hasard
                    term_idx = np.random.choice(nonzero_indices)
                    # Récupérer les puissances de ce monôme
                    term_powers = self.term_powers[term_idx].copy()
                    
                    # Créer une liste de variables utilisables pour la division
                    valid_vars = []
                    for var_idx in range(self.num_vars):
                        # Vérifier si cette variable a une puissance positive
                        if term_powers[var_idx] > 0:
                            # Diminuer la puissance pour voir où on atterrirait
                            new_powers = term_powers.copy()
                            new_powers[var_idx] -= 1
                            
                            # #Trouver l'index du terme correspondant aux nouvelles puissances
                            # target_term_idx = -1
                            # for idx, powers in enumerate(self.term_powers):
                            #     if np.array_equal(powers, new_powers):
                            #         target_term_idx = idx
                            #         break #old method that used to take more time. Keep it in case problems
                            
                            #if new_powers[0] == 6: #by hand fix but this should be fixed better
                            #    target_term_idx = 208

                            # if np.sum(new_powers) != 0:
                            #     target_term_idx = np.where((self.term_powers == np.array(new_powers)).all(axis=1))[0][0]
                            # else: 
                            #     target_term_idx = -1

                            target_term_idx_arr = np.where((self.term_powers == np.array(new_powers)).all(axis=1))[0]
                            if len(target_term_idx_arr) > 0:
                                target_term_idx = target_term_idx_arr[0]
                            else:
                                target_term_idx = -1
                            
                            # Vérifier si le terme cible existe et a un coefficient nul
                            if target_term_idx != -1 and current_coeffs_batch[i, target_term_idx] == 0:
                                valid_vars.append((var_idx, target_term_idx))
                    
                    # S'il n'y a pas de variables valides pour la division, essayer une autre opération
                    if not valid_vars:
                        continue
                        
                    # Choisir aléatoirement parmi les variables valides
                    var_idx, target_term_idx = valid_vars[np.random.randint(len(valid_vars))]
                    
                    # Déplacer le coefficient vers le nouveau terme
                    coeff_value = proposed_coeffs[i, term_idx]
                    proposed_coeffs[i, term_idx] = 0.0  # Supprimer l'ancien
                    proposed_coeffs[i, target_term_idx] = coeff_value  # Ajouter au nouveau terme
                    
                    # Calculer le ratio de proposition
                    forward_prob = operation_probs[4] / num_nonzero * (1.0 / len(valid_vars))
                    
                    # Pour l'opération inverse (multiplication), il faut trouver la probabilité
                    # de choisir le terme cible et la variable correspondante
                    new_nonzero_indices = np.nonzero(proposed_coeffs[i])[0]
                    new_num_nonzero = len(new_nonzero_indices)
                    
                    # Les nouvelles puissances après division
                    new_powers = self.term_powers[target_term_idx]
                    
                    # Vérifier si la multiplication inverse est possible
                    # Pour chaque variable, vérifier si la multiplication mènerait à un terme avec coefficient nul
                    valid_reverse_vars = 0
                    for reverse_var_idx in range(self.num_vars):
                        # Vérifier si la multiplication ne dépasse pas le degré max
                        if np.sum(new_powers) + 1 <= self.max_degree:
                            # Créer les puissances après multiplication
                            reverse_powers = new_powers.copy()
                            reverse_powers[reverse_var_idx] += 1
                            
                            # # Trouver l'index du terme après multiplication
                            # reverse_term_idx = -1
                            # for idx, powers in enumerate(self.term_powers):
                            #     if np.array_equal(powers, reverse_powers):
                            #         reverse_term_idx = idx
                            #         break #old method that used to take more time. Keep it in case problems
                        
                            reverse_term_idx = np.where((self.term_powers == np.array(new_powers)).all(axis=1))[0][0]

                            # Vérifier si le terme après multiplication a un coefficient nul
                            if reverse_term_idx != -1 and reverse_term_idx != term_idx and proposed_coeffs[i, reverse_term_idx] == 0:
                                valid_reverse_vars += 1
                    
                    if valid_reverse_vars > 0:
                        prob_choose_target = 1.0 / new_num_nonzero
                        prob_choose_var = 1.0 / valid_reverse_vars
                        reverse_prob = operation_probs[3] * prob_choose_target * prob_choose_var
                    else:
                        reverse_prob = 0.0
                    
                    log_proposal_ratios[i] = np.log(reverse_prob) - np.log(forward_prob) if (forward_prob > 0 and reverse_prob > 0) else 0
                    break  # Opération réussie, sortir de la boucle de tentatives
                
                # Si toutes les tentatives échouent, faire une modification simple d'un coefficient existant
                if attempt == max_attempts - 1 and num_nonzero > 0:
                    # Choisir un coefficient non nul au hasard
                    idx = np.random.choice(nonzero_indices)
                    # Modifier sa valeur
                    proposed_coeffs[i, idx] += np.random.normal(0, self.sigma_proposal(i,n_iter))
                    
                    # Pour une modification gaussienne symétrique, le ratio est 1 (log ratio = 0)
                    log_proposal_ratios[i] = 0.0
        
        # Appliquer le seuil minimum aux coefficients proposés
        #proposed_coeffs = self.apply_coefficient_threshold(proposed_coeffs)
        
        # Normaliser les particules proposées
        #proposed_coeffs = self.normalize_polynomial_batch(proposed_coeffs)
            
        return proposed_coeffs, log_proposal_ratios
    
    def compute_log_target_batch(self, coefficients_batch: np.ndarray, delta_beta: float, 
                                sparsity_factor: float = 1.0,
                                degree_bias: float = 0.5) -> np.ndarray:
        """
        Calcule le logarithme de la distribution cible pour un lot de coefficients.
        
        Args:
            coefficients_batch: Batch de coefficients (n_particles, num_terms)
            temperature: Température actuelle
            sparsity_factor: Facteur de parcimonie
            
        Returns:
            Vecteur de log-probabilités cibles (n_particles,)
        """
        loss_batch = self.compute_loss_batch(coefficients_batch)
        
        # Réactiver le prior pour avoir une meilleure distribution de poids
        prior_log_prob_batch = self.compute_prior_log_prob_batch(coefficients_batch, sparsity_factor, degree_bias)
        
        # Appliquer une températion progressive à la perte et au prior
        return -loss_batch * delta_beta + prior_log_prob_batch #* temperature
    
    def local_search(self, coefficients, n_steps=100, use_reg=False):
        """Affine le polynôme avec une recherche locale"""
        best_coeffs = coefficients.copy()
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
        Affine un lot de polynômes avec une recherche locale.
        
        Args:
            coefficients_batch: Batch de coefficients de forme (n_particles, num_terms)
            n_steps: Nombre d'étapes de recherche locale
            use_reg: if True, compute loss with compute_loss_batch, if False with evaluate_polynomial_batch_fast
            
        Returns:
            Tuple contenant:
                - Batch de coefficients améliorés de forme (n_particles, num_terms)
                - Vecteur de pertes améliorées de forme (n_particles,)
        """
        coefficients_batch_ = coefficients_batch.copy()

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
    
    def local_search_batch_all_nonzero(self, coefficients_batch, n_steps=100, std = 0.1, verbose=True):
        """
        Affine un lot de polynômes avec une recherche locale 
        perturbant TOUS les coefficients non nuls.
        
        Args:
            coefficients_batch: Batch de coefficients de forme (n_particles, num_terms)
            n_steps: Nombre d'étapes de recherche locale
            verbose: Affichage des informations de progression
                
        Returns:
            Batch de coefficients améliorés de forme (n_particles, num_terms)
        """
        for step in range(n_steps):
            if verbose:
                print(f"Step {step}/{n_steps}")
            
            # Évaluation des pertes initiales
            prev_losses = evaluate_polynomial_batch_fast(self.term_matrix, coefficients_batch)

            # Masque des coefficients non nuls
            non_zero_mask = (coefficients_batch != 0)

            # Création d'une copie pour les perturbations
            coefficients_batch_perturbed = coefficients_batch.copy()
            
            # Génération de perturbations pour tous les coefficients non nuls
            perturbations = np.random.normal(
                loc=0, 
                scale=std, 
                size=coefficients_batch.shape
            )
            
            # Appliquer les perturbations uniquement aux coefficients non nuls
            coefficients_batch_perturbed[non_zero_mask] += perturbations[non_zero_mask]

            # Évaluation des nouvelles pertes
            new_losses = evaluate_polynomial_batch_fast(self.term_matrix, coefficients_batch_perturbed)

            # Mise à jour des coefficients si la perte est améliorée
            improvement_mask = new_losses < prev_losses
            coefficients_batch[improvement_mask] = coefficients_batch_perturbed[improvement_mask]
        
        return coefficients_batch

    def multinomial_resample(self, weights, n_samples):
        """
        Resampling multinomial classique (votre méthode originale)
        
        Args:
            weights: poids normalisés des particules
            n_samples: nombre d'échantillons à générer
        
        Returns:
            indices: indices des particules resamplees
        """
        return np.random.choice(len(weights), size=n_samples, p=weights, replace=True)

    def partial_resampling(self,particles, normalized_weights, unnormalized_weights, 
                        resample_fraction=0.7, verbose=False):
        """
        Resampling partiel : ne resample qu'une fraction des particules
        
        Args:
            particles: array des particules
            normalized_weights: poids normalisés
            unnormalized_weights: poids non normalisés
            resample_fraction: fraction des particules à resampler (0.7 = 70%)
            verbose: affichage debug
        
        Returns:
            particles, normalized_weights, unnormalized_weights: arrays mis à jour
        """
        n_particles = len(particles)
        n_resample = int(n_particles * resample_fraction)
        n_keep = n_particles - n_resample
        
        if verbose:
            print(f"Partial resampling: keeping {n_keep} particles, resampling {n_resample}")
        
        # 1. Garder les n_keep meilleures particules intactes
        best_indices = np.argsort(normalized_weights)[-n_keep:]
        kept_particles = particles[best_indices]
        kept_normalized_weights = normalized_weights[best_indices]
        kept_unnormalized_weights = unnormalized_weights[best_indices]
        
        # 2. Resampler le reste avec la méthode classique
        resample_indices = self.multinomial_resample(normalized_weights, n_resample)
        resampled_particles = particles[resample_indices]
        
        # 3. Combiner les particules gardées et resamplees
        new_particles = np.concatenate([kept_particles, resampled_particles])
        
        # 4. Recalculer les poids
        # Les particules gardées conservent leurs poids originaux
        # Les particules resamplees ont des poids uniformes entre elles
        new_unnormalized_weights = np.concatenate([
            kept_unnormalized_weights,
            np.ones(n_resample)  # Poids uniformes pour les resamplees
        ])
        
        # Renormalisation
        new_normalized_weights = new_unnormalized_weights / np.sum(new_unnormalized_weights)
        
        return new_particles, new_normalized_weights, new_unnormalized_weights
    
    def run_annealing_is(self, n_iter: int, 
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
                        resample_fraction: float = 0.7,
                        )-> Tuple[np.ndarray, float, np.ndarray, np.ndarray, list]:
        """
        Exécute l'algorithme d'Annealing Importance Sampling vectorisé.
        
        Args:
            n_iter: Nombre d'itérations d'annealing
            n_iter_exploit: Number of exploitation steps, with p_mod = 1
            n_particles: Nombre de particules
            sparsity_factor: Facteur de parcimonie pour le prior
            verbose: Si True, affiche des informations pendant l'exécution
            analysis: Si True, garde plus de données en vue de l'analyse
            target_terms: list of nonzero terms for each target polynomials
            
        Returns:
            Tuple contenant les meilleurs coefficients trouvés, leur perte, 
            toutes les particules finales, leurs pertes, et l'historique des 10 premières particules
        """
        start_time = time.time()
        
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
                print(f"Itération {i}/{n_iter}, Beta: {beta_current}, "
                    f"Taux d'acceptation: {acceptance_rate:.3f}, "
                    f"ESS: {ess:.2f}/{n_particles}, "
                    f"Meilleure perte: {best_loss:.6f}, Temps écoulé: {elapsed:.2f}s, "
                    f"probas {self.prob_modify(i,n_iter), self.prob_multiply(i,n_iter), self.prob_divide(i,n_iter), self.prob_add(i,n_iter), self.prob_remove(i,n_iter)}, "
                    f"best pol : {self.polynomial_to_string(particles[min_loss_idx])}"
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
            print(f"\nTemps total d'exécution: {total_time:.2f}s")
            print(f"Meilleure perte: {best_loss_history[-1]:.6f}")
            print(f"Nombre de monômes dans la meilleure solution: {np.count_nonzero(best_coeffs)}")
        if analysis:
            total_time = time.time() - start_time
            return best_coeffs, best_loss_history, particles, ess_history, acceptance_history_all, particles_after_exploit, losses, particles_history, beta_history, normalized_weights, count_good_pol, total_time
        else:
            return best_coeffs, best_loss_history, particles, ess_history, acceptance_history_all, particles_after_exploit, losses, particles_history, beta_history, normalized_weights
    
    def polynomial_to_string(self, coefficients: np.ndarray, threshold: float = 1e-6) -> str:
        """
        Convertit un vecteur de coefficients en une représentation lisible du polynôme.
        
        Args:
            coefficients: Vecteur des coefficients du polynôme
            threshold: Seuil en dessous duquel un coefficient est considéré comme nul
            
        Returns:
            Représentation sous forme de chaîne du polynôme
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
    
    def to_polynomial(self, terms_in: np.ndarray, coeffs_in: np.ndarray) -> np.ndarray:
        """
        Returns the vector representing the polynomial with terms_in terms and coeffs_in coefficients.
        
        Args:
            terms_in: array of terms of the polynomials, each term is of the form (num_vars)
            coeffs_in: array of coefficients of the polynomials
            
        Returns:
            Batch of coefficients of the form (num_terms)
        """
        pol_out = np.zeros(251)
        for i,term in enumerate(terms_in):
            term_idx = np.where((self.term_powers == term).all(axis=1))[0][0]
            pol_out[term_idx] = coeffs_in[i]
        
        return pol_out

