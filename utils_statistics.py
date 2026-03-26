import numpy as np
import pandas as pd
import time
from itertools import product
import multiprocessing as mp
from functools import partial
import os
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
        return int(comb(self.num_vars + self.max_degree, self.max_degree)) - 1

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
        for degree in range(1, self.max_degree + 1):
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

    def to_equivalence_class_polynomials(self, pol_in: np.ndarray) -> np.ndarray:
        """
        Renvoie la classe d'équivalence du polynôme pol_in : identifie les variables qui peuvent être factorisées et les élimine.

        Args:
            pol_in: coefficients du polynôme d'entrée

        Returns:
            Batch de coefficients de forme (num_terms)
        """
        new_pol = pol_in.copy()

        # Identify the nonzero coefficients, ie the nonzero monomials.
        nonzero_terms = np.nonzero(pol_in)[0]

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
                new_pol[term_idx] = 0.0
                new_pol[target_term_idx] = pol_in[term_idx]

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


    def test_include_targets(self, pol_in: np.ndarray, nonzero_terms_targets: list) -> bool:
        """
        Returns True if pol_in countains the same monomials of at least one of the target polynomials, False if not.

        Args:
            pol_in: array defining the polynomials
            nonzero_terms_targets: list containing the position of the nonzero terms of each target polynomials in the term_powers basis, to be computed with self.nonzero_terms_targets

        Returns:
            True if pol_in countains the same monomials of at least one of the target polynomials, False if not.
        """
        # Non zero terms in the equivalence class of pol_in
        nonzero_terms = np.nonzero(self.to_equivalence_class_polynomials(pol_in))[0]

        test_targets_inclusion = False

        for i in range(len(nonzero_terms_targets)):
            try:
                test_targets_inclusion |= (np.sort(np.intersect1d(nonzero_terms,nonzero_terms_targets[i])) == np.sort(nonzero_terms_targets[i])).all()
            except:
                continue

        return test_targets_inclusion


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


    # def evaluate_polynomial_batch(self, coefficients_batch: np.ndarray) -> np.ndarray:
    #     """
    #     Évalue un lot de polynômes sur les données d'entrée.

    #     Args:
    #         coefficients_batch: Batch de coefficients de forme (n_particles, num_terms)

    #     Returns:
    #         Matrice de forme (n_particles, n_samples) où chaque élément [i, j] est
    #         la valeur du polynôme i au point data_x[j]
    #     """

    #     return np.dot(coefficients_batch, self.term_matrix.T)


    def compute_loss_batch(self, coefficients_batch: np.ndarray) -> np.ndarray:
        """
        Calcule la fonction de perte pour un lot de coefficients.
        Dans ce cas, la perte est l'erreur quadratique moyenne de P(x_data) par rapport à 0.

        Args:
            coefficients_batch: Batch de coefficients de forme (n_particles, num_terms)

        Returns:
            Vecteur de pertes de forme (n_particles,)
        """
        # Évaluer P(x_data) pour chaque ensemble de coefficients
        #evaluations_squared = self.evaluate_polynomial_batch(coefficients_batch)

        #sum_coeffients_batch = np.sum(coefficients_batch**2,axis=1)

        # La cible est 0 pour tous les points
        #squared_errors = np.square(evaluations)  # (evaluations - 0)²

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
                            if np.sum(new_powers) != 0:
                                target_term_idx = np.where((self.term_powers == np.array(new_powers)).all(axis=1))[0][0]
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

    def local_search_batch(self, coefficients_batch, n_steps=100, verbose = True):
        """
        Affine un lot de polynômes avec une recherche locale.

        Args:
            coefficients_batch: Batch de coefficients de forme (n_particles, num_terms)
            n_steps: Nombre d'étapes de recherche locale

        Returns:
            Tuple contenant:
                - Batch de coefficients améliorés de forme (n_particles, num_terms)
                - Vecteur de pertes améliorées de forme (n_particles,)
        """
        coefficients_batch_ = coefficients_batch.copy()

        for _ in range(n_steps):
            if verbose :
                print(f"Step {_}/{n_steps}")

            prev_losses = evaluate_polynomial_batch_fast(self.term_matrix,coefficients_batch_)

            non_zero_mask = (coefficients_batch_ != 0)

            non_zero_indices = [np.flatnonzero(row) for row in non_zero_mask]

            chosen_col_indices = np.array([np.random.choice(cols) for cols in non_zero_indices])

            row_indices = np.arange(coefficients_batch_.shape[0])

            coefficients_batch_copy = coefficients_batch_.copy()
            coefficients_batch_copy[row_indices, chosen_col_indices] += np.random.normal(loc=0, scale=0.01, size=coefficients_batch_copy.shape[0])

            new_losses = evaluate_polynomial_batch_fast(self.term_matrix,coefficients_batch_copy)

            mask = new_losses < prev_losses

            coefficients_batch_[mask] = coefficients_batch_copy[mask]

        return coefficients_batch_

    def run_annealing_is(self, n_iter: int, n_particles: int,
                        sparsity_factor: float = 1.0,
                        degree_bias: float = 0.5,
                        verbose: bool = True,
                        target_acc_rate: float = 0.3,
                        adaptation_strength: float = 0.05,
                        beta_schedule: Callable[[int, int], float] = lambda i,n_iter: 1e-10 + (i/n_iter)**2,
                        adapative_temp: bool = True,
                        analysis: bool = False,
                        target_terms: list = [],
                        )-> Tuple[np.ndarray, float, np.ndarray, np.ndarray, list]: # Camille : à changer pour adapter aux output ?
        """
        Exécute l'algorithme d'Annealing Importance Sampling vectorisé.

        Args:
            n_iter: Nombre d'itérations d'annealing
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

        # Initialize the count of polynomials containing the target polynomials
        if analysis:
            nonzero_terms_target = self.nonzero_terms_targets(target_terms)
            count_good_pol = np.zeros(n_iter)
            for pol_temp in particles:
                count_good_pol[0] += self.test_include_targets(pol_temp,nonzero_terms_target)

        # Appliquer le seuil minimum aux coefficients initiaux
        #particles = self.apply_coefficient_threshold(particles)
        # Normaliser après application du seuil
        #particles = self.normalize_polynomial_batch(particles)

        weights = np.ones(n_particles) / n_particles

        best_loss = float('inf')
        best_coeffs = None

        # Liste pour sauvegarder les 10 premières particules à chaque étape
        particles_history = []

        acceptance_history = deque(maxlen=5)
        #beta_current = initial_beta

        beta_history = [0]

        if adapative_temp == True:
            beta_current = beta_schedule(1, n_iter)
            beta_prev = beta_schedule(0, n_iter)

        for i in range(1,n_iter):
            # Calculer la température actuelle
            if adapative_temp == False:
                beta_prev = beta_schedule(i-1, n_iter)
                beta_current = beta_schedule(i, n_iter)
                delta_beta = beta_current - beta_prev
            else:
                if i > 1 and acceptance_history:
                    beta_prev = beta_current
                    avg_acceptance_rate = np.mean(acceptance_history)

                    # Ajuster beta selon l'écart par rapport au taux cible
                    print(avg_acceptance_rate,target_acc_rate,acceptance_history)
                    if avg_acceptance_rate < target_acc_rate - 0.05:
                        print("decrease beta")
                        # Taux trop bas -> diminuer beta (augmenter température)
                        beta_current *= (1.0 - adaptation_strength)
                    elif avg_acceptance_rate > target_acc_rate + 0.05:
                        print("increase beta")

                        # Taux trop haut -> augmenter beta (diminuer température)
                        beta_current *= (1.0 + adaptation_strength)
                else:
                    # Au début, suivre un schedule traditionnel pour démarrer
                    beta_current = beta_schedule(i, n_iter)

                delta_beta = beta_current - beta_prev

            beta_history.append(beta_current)
            # Sauvegarder les 10 premières particules à cette étape
            particles_history.append(particles[:10].copy())

            # Étape de repondération - vectorisée
            log_weights = self.compute_log_target_batch(particles, delta_beta, sparsity_factor, degree_bias)

            # Normaliser les poids avec un contrôle de stabilité numérique
            max_log_weight = np.max(log_weights)

            # Ajouter un facteur d'échelle pour éviter que les poids ne soient trop concentrés
            # Au début de l'annealing, la température est élevée, donc utiliser un facteur d'échelle plus important
            # Cela aide à prévenir la dégénérescence des poids
            #scale_factor = max(0.1, min(1.0, 10.0 * temp))  # Facteur adaptatif basé sur la température

            # Appliquer la mise à l'échelle aux log-poids avant l'exponentiation
            scaled_log_weights = (log_weights - max_log_weight) #* scale_factor

            weights = np.exp(scaled_log_weights)

            weights = weights / np.sum(weights)

            # Calculer la taille effective de l'échantillon
            ess = 1.0 / np.sum(weights ** 2)

            # Rééchantillonnage si nécessaire
            if ess < n_particles / 2:
                if verbose:
                    print(f"ESS = {ess:.2f} : resampling needed")
                indices = np.random.choice(n_particles, size=n_particles, p=weights, replace=True)
                particles = particles[indices]
                weights = np.ones(n_particles) / n_particles
                ess = 1.0 / np.sum(weights ** 2)

            # Étape de mutation avec Metropolis-Hastings spécialisé
            proposed_particles, log_proposal_ratios = self.mh_proposal_batch(particles, i, n_iter)

            ## Appliquer le seuil minimum aux coefficients proposés
            #proposed_particles = self.apply_coefficient_threshold(proposed_particles)

            # Normaliser après application du seuil
            #proposed_particles = self.normalize_polynomial_batch(proposed_particles)

            # Calculer les probabilités cibles actuelles et proposées - vectorisé
            current_log_probs = self.compute_log_target_batch(particles, delta_beta, sparsity_factor, degree_bias)
            proposed_log_probs = self.compute_log_target_batch(proposed_particles, delta_beta, sparsity_factor, degree_bias)

            # Calculer les ratios d'acceptation - vectorisé
            log_acceptance_ratios = proposed_log_probs - current_log_probs + log_proposal_ratios

            # Décider lesquelles des propositions accepter - vectorisé
            log_random = np.log(np.random.random(n_particles))
            accept_mask = log_random < log_acceptance_ratios

            # Calculer et enregistrer le taux d'acceptation pour cette itération
            acceptance_rate = np.mean(accept_mask)
            acceptance_history.append(acceptance_rate)

            # Mettre à jour les particules acceptées
            particles[accept_mask] = proposed_particles[accept_mask]

            # Mettre à jour le meilleur résultat
            losses = self.compute_loss_batch(particles)
            min_loss_idx = np.argmin(losses)
            if losses[min_loss_idx] < best_loss:
                best_loss = losses[min_loss_idx]
                best_coeffs = particles[min_loss_idx].copy()

            if verbose:
                elapsed = time.time() - start_time
                print(f"Itération {i}/{n_iter}, Beta: {beta_current:.6f}, "
                    f"Taux d'acceptation: {acceptance_rate:.3f}, "
                    f"ESS: {ess:.2f}/{n_particles}, "
                    f"Meilleure perte: {best_loss:.6f}, Temps écoulé: {elapsed:.2f}s, "
                )

            # Count the polynomials containing the target polynomials
            if analysis:
                for pol_temp in particles:
                    count_good_pol[i] += self.test_include_targets(pol_temp,nonzero_terms_target)

        # Appliquer le seuil minimum aux meilleurs coefficients trouvés
        if best_coeffs is not None:
            #best_coeffs = self.apply_coefficient_threshold(best_coeffs)
            # Normaliser après application du seuil
            best_coeffs = self.normalize_polynomial_batch(best_coeffs.reshape(1, -1))[0]

        if verbose:
            total_time = time.time() - start_time
            print(f"\nTemps total d'exécution: {total_time:.2f}s")
            print(f"Meilleure perte: {best_loss:.6f}")
            print(f"Nombre de monômes dans la meilleure solution: {np.count_nonzero(best_coeffs)}")

        if analysis:
            total_time = time.time() - start_time
            return best_coeffs, best_loss, particles, losses, particles_history, acceptance_history, beta_history, count_good_pol, total_time
        else:
            return best_coeffs, best_loss, particles, losses, particles_history, beta_history



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

    def local_search(self, coefficients, n_steps=100):
        """Affine le polynôme avec une recherche locale"""
        best_coeffs = coefficients.copy()
        best_loss = self.compute_loss_batch(best_coeffs.reshape(1, -1))[0]

        for _ in range(n_steps):
            # Modification légère d'un coefficient non nul
            nonzero_indices = np.nonzero(best_coeffs)[0]
            if len(nonzero_indices) > 0:
                idx = np.random.choice(nonzero_indices)
                new_coeffs = best_coeffs.copy()
                new_coeffs[idx] += np.random.normal(0, 0.01)  # Petit pas

                # Évaluer la nouvelle perte
                new_loss = self.compute_loss_batch(new_coeffs.reshape(1, -1))[0]

                # Accepter si amélioration
                if new_loss < best_loss:
                    best_coeffs = new_coeffs
                    best_loss = new_loss

        return best_coeffs, best_loss

def init_worker(data_shared_array, data_shape):
    """
    Initialise le processus worker avec les données partagées
    """
    global my_data_raw
    # Reconstruire le tableau numpy à partir de la mémoire partagée sans get_obj()
    # car RawArray n'a pas cette méthode
    my_data_raw = np.frombuffer(data_shared_array, dtype=np.float64).reshape(data_shape)

def run_single_trial_shared(args):
    """
    Version modifiée pour utiliser les données globales en mémoire partagée
    """
    config, trial_idx = args
    
    print(f"Process {os.getpid()} - Starting run {trial_idx} for config {config}")
    
    # Utilisation des données globales partagées
    global my_data_raw
    
    # Échantillonnage des données
    indices = np.random.choice(my_data_raw.shape[0], config.get('n_samples', 10000))
    my_data_x = np.array([my_data_raw[indices,0],
                          my_data_raw[indices,1],
                          my_data_raw[indices,2],
                          np.exp(my_data_raw[indices,3]),
                          my_data_raw[indices,4]]).T
            
    # Configuration du sampler
    sampler = SparsePolynomialSampler(
        data_x=my_data_x,
        max_degree=config.get('max_degree', 4),
        num_vars=config.get('num_vars', 5),
        max_num_monomials=config.get('max_num_monomials', 6),
        sigma_proposal=lambda i, n_iter: config.get('sigma_proposal', 0.1),
        prob_add=lambda i, n_iter: config.get('prob_add', 0.),
        prob_remove=lambda i, n_iter: config.get('prob_remove', 0.),
        prob_modify=lambda i, n_iter: config.get('prob_modify', 0.5),
        prob_divide=lambda i, n_iter: config.get('prob_divide', 0.25),
        prob_multiply=lambda i, n_iter: config.get('prob_multiply', 0.25),
        regularisation_factor=config.get('regularisation_factor', 1e3),
        min_coeff_threshold=config.get('min_coeff_threshold', 0.1)
    )
    
    # Fonction beta_schedule personnalisée
    if 'beta_schedule_power' in config:
        power = config['beta_schedule_power']
        beta_schedule = lambda i, n_iter: 1e-10 + (i / n_iter)**power
    else:
        beta_schedule = lambda i, n_iter: 1e-10 + (i / n_iter)**5

    # Exécution de l'algorithme avec les cibles globales
    target_pols_terms = [np.array([[1,0,0,0,0],[1,0,0,1,0],[0,1,0,1,1],[0,1,1,1,0]]),np.array([[1,0,0,0,1],[0,1,0,0,0],[1,0,1,0,0],[0,1,0,1,0],[0,1,0,1,2]]),
                     np.array([[0,1,0,0,0],[0,1,0,1,0],[1,0,0,1,1],[1,0,1,1,0],[0,1,2,1,0]]),np.array([[0,1,0,0,0],[0,1,0,1,0],[1,0,1,0,0],[1,0,0,1,1],[0,1,1,1,1]])]
    
    start_time = time.time()
    best_coeffs, best_loss, particles, losses, particles_history, acceptance_rate_history, beta_history, count_good_pol, _ = sampler.run_annealing_is(
        n_iter=config.get('n_iter', 1000),
        n_particles=config.get('n_particles', 1000),
        sparsity_factor=config.get('sparsity_factor', 1.0),
        degree_bias=config.get('degree_bias', 0.5),
        verbose=False,
        beta_schedule=beta_schedule,
        adapative_temp=config.get('adapative_temp', False),
        target_acc_rate=config.get('target_acc_rate', 0.0),
        adaptation_strength=config.get('adaptation_strength', 0.00),
        analysis=True,
        target_terms=[target_pols_terms[0], target_pols_terms[1]]
    )
    total_time = time.time() - start_time
    
    # Calcul des métriques
    convergence_threshold = 1.1
    min_losses = [np.min(sampler.compute_loss_batch(part_history)) for part_history in particles_history]
    convergence_iter = next((i for i, loss in enumerate(min_losses) if loss <= best_loss * convergence_threshold), len(min_losses))

    # Ajout de l'identifiant de configuration pour regroupement ultérieur
    config_id = "_".join([f"{k}={v}" for k, v in config.items()])
        
    return {
        'config_id': config_id,
        'config': config,
        'trial': trial_idx,
        'best_loss': best_loss,
        'best_coeffs': best_coeffs,
        'losses': losses,
        'acceptance_rate_history': acceptance_rate_history,
        'convergence_iter': convergence_iter,
        'execution_time': total_time,
        'final_target_recovery': count_good_pol,
        'particles': particles
    }

def run_performance_study_full_parallel(config_variations, n_trials=10, max_workers=None):
    """
    Version améliorée de l'étude de performance utilisant la mémoire partagée pour les données
    et parallélisant toutes les tâches (configurations et essais) en une seule fois.
    """
    # Chargement des données
    my_data_raw = np.load("Points_02_04_25_m2p2init.npy")
    
    # Création d'un tableau partagé
    data_shape = my_data_raw.shape
    shared_array = mp.RawArray('d', my_data_raw.size)
    
    # Copie des données dans le tableau partagé
    shared_np_array = np.frombuffer(shared_array, dtype=np.float64).reshape(data_shape)
    np.copyto(shared_np_array, my_data_raw)
    
    # Détermination du nombre de travailleurs
    if max_workers is None:
        max_workers = mp.cpu_count()
    
    # Génération des combinaisons de paramètres
    param_names = list(config_variations.keys())
    param_values = list(config_variations.values())
    all_combinations = list(product(*param_values))
    all_configs = [dict(zip(param_names, combo)) for combo in all_combinations]
    
    # Création de toutes les tâches (combinaison de config + essai)
    all_tasks = []
    for config_idx, config in enumerate(all_configs):
        for trial_idx in range(n_trials):
            all_tasks.append((config, trial_idx))
    
    print(f"Starting performance study with {len(all_configs)} configurations, "
          f"{n_trials} trials each ({len(all_tasks)} total tasks), using {max_workers} workers")
    
    # Initialisation du pool avec les données partagées
    with mp.Pool(processes=max_workers, 
                 initializer=init_worker, 
                 initargs=(shared_array, data_shape)) as pool:
        
        # Exécution de toutes les tâches en parallèle
        all_results = pool.map(run_single_trial_shared, all_tasks)
    
    # Organisation des résultats par configuration
    results = []
    
    # Regroupement des résultats par configuration
    grouped_results = {}
    for result in all_results:
        config_id = result['config_id']
        if config_id not in grouped_results:
            grouped_results[config_id] = []
        grouped_results[config_id].append(result)
    
    # Calcul des statistiques pour chaque configuration
    for config_id, trials in grouped_results.items():
        config_result = trials[0]['config'].copy()
        
        for metric in ['best_loss', 'best_coeffs', 'losses','acceptance_rate_history', 'convergence_iter', 
                      'execution_time', 'final_target_recovery', 'particles']:
            values = [r[metric] for r in trials]
            config_result[f'{metric}'] = values
   #         if metric not in ['best_coeffs', 'particles']:
   #             config_result[f'{metric}_mean'] = np.mean(values)
   #             config_result[f'{metric}_std'] = np.std(values)
        
        results.append(config_result)
    
    return pd.DataFrame(results)

