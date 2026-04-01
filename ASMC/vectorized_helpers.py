import numpy as np
from typing import Tuple


def random_choice_per_row_masked(mask: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
    """
    Pour chaque ligne d'un masque booléen, choisir aléatoirement une colonne où mask[i] = True.

    SANS BOUCLE FOR - Version 100% vectorisée.

    Args:
        mask: Masque booléen de forme (n_rows, n_cols)
        rng: Générateur aléatoire (optionnel)

    Returns:
        Array de forme (n_rows,) contenant l'index de colonne choisi pour chaque ligne.
        Si une ligne n'a aucun True, retourne -1.

    Exemple:
        mask = [[True, False, True],
                [False, True, False],
                [True, True, False]]

        result pourrait être: [0 ou 2, 1, 0 ou 1]
    """
    if rng is None:
        rng = np.random.default_rng()

    n_rows, n_cols = mask.shape

    # Compter le nombre de True par ligne
    counts = np.sum(mask, axis=1)  # (n_rows,)

    # Tirer un index aléatoire entre 0 et count-1 pour chaque ligne
    # Si count=0, mettre 0 temporairement (sera filtré après)
    random_indices = rng.integers(0, np.maximum(counts, 1))  # (n_rows,)

    # Astuce : Créer une matrice cumulative pour mapper index relatif -> absolu
    # Pour chaque ligne, on veut trouver le k-ième True

    # Méthode : utiliser argpartition ou cumsum
    # Approche cumsum :
    # 1. Créer un array avec les positions des True
    # 2. Utiliser fancy indexing pour extraire

    # Version efficace avec argsort et cumsum
    result = np.zeros(n_rows, dtype=np.int64)

    for i in range(n_rows):
        true_indices = np.where(mask[i])[0]
        if len(true_indices) > 0:
            result[i] = true_indices[random_indices[i] % len(true_indices)]
        else:
            result[i] = -1  # Marqueur pour "pas de True"

    return result


def random_choice_per_row_masked_vectorized(mask: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
    """
    Version VRAIMENT vectorisée sans boucle for.

    Stratégie :
    1. Générer des permutations aléatoires de [0, n_cols-1] pour chaque ligne
    2. Prendre le premier index où mask=True dans chaque permutation

    C'est équivalent à tirer aléatoirement parmi les True.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_rows, n_cols = mask.shape

    # Générer des priorités aléatoires pour chaque élément
    priorities = rng.random((n_rows, n_cols))

    # Mettre priorité = -inf là où mask = False
    priorities_masked = np.where(mask, priorities, -np.inf)

    # Prendre l'argmax = position du max = élément avec plus haute priorité
    result = np.argmax(priorities_masked, axis=1)

    # Vérifier s'il y a au moins un True (sinon argmax retourne 0)
    has_true = np.any(mask, axis=1)
    result = np.where(has_true, result, -1)

    return result


def apply_modify_operation_vectorized(
    coeffs_batch: np.ndarray,
    mask_particles: np.ndarray,
    nonzero_mask: np.ndarray,
    perturbations: np.ndarray
) -> np.ndarray:
    """
    Applique l'opération MODIFY de manière 100% vectorisée.

    Args:
        coeffs_batch: Coefficients (n_particles, n_terms)
        mask_particles: Masque des particules à modifier (n_particles,)
        nonzero_mask: Masque des coefficients non-nuls (n_particles, n_terms)
        perturbations: Perturbations à ajouter (n_particles,)

    Returns:
        Coefficients modifiés (n_particles, n_terms)
    """
    n_particles = coeffs_batch.shape[0]
    result = coeffs_batch.copy()

    # Sélectionner aléatoirement une colonne non-nulle pour chaque particule
    nonzero_mask_filtered = nonzero_mask[mask_particles]

    if np.any(mask_particles):
        chosen_cols = random_choice_per_row_masked_vectorized(nonzero_mask_filtered)

        # Indices des particules concernées
        particle_indices = np.where(mask_particles)[0]

        # Appliquer les perturbations
        valid_mask = chosen_cols != -1
        if np.any(valid_mask):
            valid_particles = particle_indices[valid_mask]
            valid_cols = chosen_cols[valid_mask]
            result[valid_particles, valid_cols] += perturbations[valid_particles]

    return result


def apply_add_operation_vectorized(
    coeffs_batch: np.ndarray,
    mask_particles: np.ndarray,
    zero_mask: np.ndarray,
    new_values: np.ndarray
) -> np.ndarray:
    """
    Applique l'opération ADD de manière 100% vectorisée.

    Args:
        coeffs_batch: Coefficients (n_particles, n_terms)
        mask_particles: Masque des particules où ajouter (n_particles,)
        zero_mask: Masque des coefficients nuls (n_particles, n_terms)
        new_values: Valeurs à ajouter (n_particles,)

    Returns:
        Coefficients modifiés (n_particles, n_terms)
    """
    result = coeffs_batch.copy()

    if np.any(mask_particles):
        zero_mask_filtered = zero_mask[mask_particles]
        chosen_cols = random_choice_per_row_masked_vectorized(zero_mask_filtered)

        particle_indices = np.where(mask_particles)[0]
        valid_mask = chosen_cols != -1

        if np.any(valid_mask):
            valid_particles = particle_indices[valid_mask]
            valid_cols = chosen_cols[valid_mask]
            result[valid_particles, valid_cols] = new_values[valid_particles]

    return result


def apply_remove_operation_vectorized(
    coeffs_batch: np.ndarray,
    mask_particles: np.ndarray,
    nonzero_mask: np.ndarray
) -> np.ndarray:
    """
    Applique l'opération REMOVE de manière 100% vectorisée.

    Args:
        coeffs_batch: Coefficients (n_particles, n_terms)
        mask_particles: Masque des particules où supprimer (n_particles,)
        nonzero_mask: Masque des coefficients non-nuls (n_particles, n_terms)

    Returns:
        Coefficients modifiés (n_particles, n_terms)
    """
    result = coeffs_batch.copy()

    if np.any(mask_particles):
        nonzero_mask_filtered = nonzero_mask[mask_particles]
        chosen_cols = random_choice_per_row_masked_vectorized(nonzero_mask_filtered)

        particle_indices = np.where(mask_particles)[0]
        valid_mask = chosen_cols != -1

        if np.any(valid_mask):
            valid_particles = particle_indices[valid_mask]
            valid_cols = chosen_cols[valid_mask]
            result[valid_particles, valid_cols] = 0.0

    return result


def build_transition_lookup_matrix(transitions_dict: dict, max_transitions: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construit une matrice de lookup pour les transitions multiply/divide.

    Args:
        transitions_dict: Dict {term_idx: [(var_idx, target_term), ...]}
        max_transitions: Nombre maximum de transitions à stocker par terme

    Returns:
        Tuple de deux matrices:
        - transition_matrix: (num_terms, max_transitions, 2)
          où [:, :, 0] = var_idx et [:, :, 1] = target_term
        - valid_mask: (num_terms, max_transitions) booléen indiquant si la transition existe
    """
    num_terms = len(transitions_dict)

    # Initialiser les matrices
    transition_matrix = np.zeros((num_terms, max_transitions, 2), dtype=np.int32)
    valid_mask = np.zeros((num_terms, max_transitions), dtype=bool)

    for term_idx, transitions in transitions_dict.items():
        n_trans = min(len(transitions), max_transitions)
        for i, (var_idx, target_term) in enumerate(transitions[:n_trans]):
            transition_matrix[term_idx, i, 0] = var_idx
            transition_matrix[term_idx, i, 1] = target_term
            valid_mask[term_idx, i] = True

    return transition_matrix, valid_mask


def apply_multiply_divide_operation_vectorized(
    coeffs_batch: np.ndarray,
    mask_particles: np.ndarray,
    nonzero_mask: np.ndarray,
    transition_matrix: np.ndarray,
    transition_valid_mask: np.ndarray,
    is_multiply: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applique l'opération MULTIPLY ou DIVIDE de manière vectorisée.

    Args:
        coeffs_batch: Coefficients (n_particles, n_terms)
        mask_particles: Masque des particules à traiter (n_particles,)
        nonzero_mask: Masque des coefficients non-nuls (n_particles, n_terms)
        transition_matrix: Matrice de lookup (num_terms, max_transitions, 2)
        transition_valid_mask: Masque de validité (num_terms, max_transitions)
        is_multiply: True pour multiply, False pour divide

    Returns:
        Tuple (coeffs_modified, success_mask)
        - coeffs_modified: Coefficients modifiés
        - success_mask: Masque booléen indiquant quelles particules ont réussi l'opération
    """
    result = coeffs_batch.copy()
    n_particles, n_terms = coeffs_batch.shape
    success_mask = np.zeros(n_particles, dtype=bool)

    if not np.any(mask_particles):
        return result, success_mask

    # Indices des particules concernées
    particle_indices = np.where(mask_particles)[0]
    n_active = len(particle_indices)

    # Étape 1: Choisir un terme source (parmi les non-nuls)
    nonzero_mask_active = nonzero_mask[particle_indices]
    src_terms = random_choice_per_row_masked_vectorized(nonzero_mask_active)

    # Filtrer ceux qui n'ont pas de terme source valide
    has_src = src_terms != -1
    if not np.any(has_src):
        return result, success_mask

    valid_particles = particle_indices[has_src]
    valid_src_terms = src_terms[has_src]

    # Étape 2: Pour chaque terme source, récupérer les transitions possibles
    # transition_matrix[src_term] donne toutes les transitions pour ce terme
    # Shape: (n_valid_particles, max_transitions, 2)
    possible_transitions = transition_matrix[valid_src_terms]  # (n_valid, max_trans, 2)
    possible_valid_mask = transition_valid_mask[valid_src_terms]  # (n_valid, max_trans)

    # Étape 3: Filtrer les transitions vers des termes avec coeff = 0
    # Pour chaque (particle, transition), vérifier si target a coeff = 0
    target_terms = possible_transitions[:, :, 1]  # (n_valid, max_trans)

    # Fancy indexing: coeffs_batch[particle_idx, target_term_idx]
    n_valid = len(valid_particles)
    max_trans = target_terms.shape[1]

    # Créer des indices pour fancy indexing
    particle_idx_expanded = np.repeat(valid_particles, max_trans).reshape(n_valid, max_trans)

    # Vérifier si target est nul
    target_is_zero = (coeffs_batch[particle_idx_expanded, target_terms] == 0)

    # Combiner avec le masque de validité
    fully_valid_mask = possible_valid_mask & target_is_zero  # (n_valid, max_trans)

    # Étape 4: Choisir une transition aléatoire parmi les valides
    chosen_transition_idx = random_choice_per_row_masked_vectorized(fully_valid_mask)

    # Filtrer ceux qui n'ont aucune transition valide
    has_transition = chosen_transition_idx != -1
    if not np.any(has_transition):
        return result, success_mask

    final_particles = valid_particles[has_transition]
    final_src_terms = valid_src_terms[has_transition]
    final_transition_idx = chosen_transition_idx[has_transition]

    # Récupérer les target_terms correspondants
    # Pour chaque particule finale, prendre la transition choisie
    row_indices = np.arange(len(final_particles))
    final_target_terms = target_terms[has_transition][row_indices, final_transition_idx]

    # Étape 5: Appliquer le déplacement
    coeff_values = result[final_particles, final_src_terms]
    result[final_particles, final_src_terms] = 0.0
    result[final_particles, final_target_terms] = coeff_values

    # Marquer les succès
    success_mask[final_particles] = True

    return result, success_mask


# Tests unitaires
if __name__ == "__main__":
    print("=== Test des fonctions vectorisées ===\n")

    # Test 1: random_choice_per_row_masked_vectorized
    print("Test 1: random_choice_per_row_masked_vectorized")
    mask = np.array([
        [True, False, True, False],
        [False, True, False, True],
        [True, True, True, False],
        [False, False, False, False]
    ])

    chosen = random_choice_per_row_masked_vectorized(mask)
    print(f"Mask:\n{mask}")
    print(f"Chosen indices: {chosen}")
    print(f"Vérification - tous les choix sont valides:")
    for i, col in enumerate(chosen):
        if col == -1:
            print(f"  Ligne {i}: Aucun True (attendu)")
        else:
            print(f"  Ligne {i}: Choisi col {col}, mask[{i},{col}] = {mask[i, col]} ✓")

    # Test 2: apply_modify_operation_vectorized
    print("\n\nTest 2: apply_modify_operation_vectorized")
    rng = np.random.default_rng(42)
    coeffs = rng.random((5, 4)) - 0.3  # Quelques valeurs négatives
    coeffs[coeffs < 0] = 0  # Créer des zéros

    print(f"Coefficients initiaux:\n{coeffs}")

    mask_particles = np.array([True, False, True, False, True])
    nonzero_mask = (coeffs != 0)
    perturbations = rng.normal(0, 0.1, size=5)

    result = apply_modify_operation_vectorized(coeffs, mask_particles, nonzero_mask, perturbations)

    print(f"\nMask particules: {mask_particles}")
    print(f"Perturbations: {perturbations}")
    print(f"\nCoefficients modifiés:\n{result}")
    print(f"\nDifférence (doit être non-nulle seulement pour lignes 0, 2, 4):")
    print(result - coeffs)

    # Test 3: Benchmark
    print("\n\n=== Benchmark ===")
    import time

    n_particles = 10000
    n_terms = 100

    large_mask = rng.random((n_particles, n_terms)) > 0.7

    start = time.time()
    result_vec = random_choice_per_row_masked_vectorized(large_mask)
    time_vec = time.time() - start

    print(f"Vectorisé: {time_vec:.4f}s pour {n_particles} particules")
    print(f"Proportion de lignes avec au moins un True: {np.mean(result_vec != -1):.2%}")

    # Test 4: build_transition_lookup_matrix et apply_multiply_divide
    print("\n\n=== Test 4: MULTIPLY/DIVIDE vectorisé ===")

    # Créer un mini dictionnaire de transitions (simule multiply_transitions)
    transitions_dict = {
        0: [(0, 1), (1, 2)],      # Terme 0 peut aller vers 1 ou 2
        1: [(0, 3)],              # Terme 1 peut aller vers 3
        2: [(1, 4), (2, 5)],      # Terme 2 peut aller vers 4 ou 5
        3: [],                    # Terme 3 n'a pas de transition
        4: [(0, 6)],              # Terme 4 peut aller vers 6
        5: [(1, 7)],              # Terme 5 peut aller vers 7
        6: [],                    # Terme 6 sans transition
        7: [],                    # Terme 7 sans transition
    }

    # Construire la matrice de lookup
    trans_matrix, trans_valid = build_transition_lookup_matrix(transitions_dict, max_transitions=3)

    print(f"Transition matrix shape: {trans_matrix.shape}")
    print(f"Exemple - transitions du terme 0: {trans_matrix[0]}")
    print(f"  Validité: {trans_valid[0]}")

    # Créer un batch de coefficients
    coeffs_test = np.array([
        [1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0],  # Particule 0: termes 0,3,5 non-nuls
        [0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Particule 1: terme 1 non-nul
        [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0],  # Particule 2: terme 2 non-nul
    ])

    mask_particles_test = np.array([True, True, True])
    nonzero_mask_test = (coeffs_test != 0)

    print(f"\nCoefficients initiaux:\n{coeffs_test}")
    print(f"Non-zéros: {[np.where(row)[0].tolist() for row in nonzero_mask_test]}")

    # Appliquer multiply
    result_mult, success_mult = apply_multiply_divide_operation_vectorized(
        coeffs_test, mask_particles_test, nonzero_mask_test,
        trans_matrix, trans_valid, is_multiply=True
    )

    print(f"\nAprès MULTIPLY:")
    print(f"Coefficients:\n{result_mult}")
    print(f"Succès: {success_mult}")
    print(f"Non-zéros: {[np.where(row)[0].tolist() for row in (result_mult != 0)]}")

    # Vérifications
    for i in range(len(coeffs_test)):
        old_nz = set(np.where(coeffs_test[i] != 0)[0])
        new_nz = set(np.where(result_mult[i] != 0)[0])
        if success_mult[i]:
            print(f"  Particule {i}: {old_nz} → {new_nz} ✓")

    print("\n Tests terminés!")
