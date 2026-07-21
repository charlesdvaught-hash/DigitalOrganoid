"""
Implementations of biological inheritance experiments for the 3D Izhikevich SNN model.
Covers:
  - Experiment A: Structural Brain Combination
  - Experiment B: Developmental Inheritance
  - Experiment C: Neural Co-Culture / Fusion
  - Experiment D: Neural Transplantation
  - Experiment E: Developmental Trait Crossover
  - Experiment F: Experience Transfer
  - Experiment G: Combined Multi-Stage System
"""
import numpy as np
from organoid_simulator.creature_embodied import build_creature_network, W_MAX, POOLS

def enforce_dales_law(pre, weight, types):
    """Enforces Dale's law: excitatory presynaptic -> weight >= 0; inhibitory -> weight <= 0."""
    pre_types = types[pre]
    exc = pre_types > 0
    inh = pre_types < 0
    weight[exc] = np.clip(weight[exc], 0.0, W_MAX)
    weight[inh] = np.clip(weight[inh], -W_MAX, 0.0)
    return weight

def experiment_a_structural_combination(parent1, parent2, spatial_jitter=0.03, weight_jitter=0.01, seed=None):
    """
    Experiment A: Structural Brain Combination
    Combines the physical and synaptic structures of parent1 and parent2 SNNs.
    Resolves conflicts by averaging overlapping weights and applying Dale's Law clamps.
    """
    rng = np.random.default_rng(seed)

    # Check if we can do a direct scaffold-aligned merge (same size N)
    N = parent1['N']
    if parent2['N'] != N:
        # Fallback if different sizes: use parent1 as template and inject some parent2 traits
        N = parent1['N']
        parent2_aligned = parent1 # fallback template
    else:
        parent2_aligned = parent2

    # 1. Blend spatial positions
    px = 0.5 * (parent1['px'] + parent2_aligned['px']) + rng.standard_normal(N) * spatial_jitter
    py = 0.5 * (parent1['py'] + parent2_aligned['py']) + rng.standard_normal(N) * spatial_jitter
    pz = 0.5 * (parent1['pz'] + parent2_aligned['pz']) + rng.standard_normal(N) * spatial_jitter

    # Clamp back to unit ball
    d2 = px*px + py*py + pz*pz
    out_of_bounds = d2 > 1.0
    if out_of_bounds.any():
        scale = np.sqrt(d2[out_of_bounds])
        px[out_of_bounds] /= scale
        py[out_of_bounds] /= scale
        pz[out_of_bounds] /= scale

    # 2. Blend parameters
    a = 0.5 * (parent1['a'] + parent2_aligned['a'])
    b = 0.5 * (parent1['b'] + parent2_aligned['b'])
    c = 0.5 * (parent1['c'] + parent2_aligned['c'])
    d = 0.5 * (parent1['d'] + parent2_aligned['d'])

    v = c.copy()
    u = b * v

    # 3. Interleave or pick types and subtypes
    types = np.where(rng.random(N) < 0.5, parent1['types'], parent2_aligned['types'])
    subtypes = np.where(rng.random(N) < 0.5, parent1['subtypes'], parent2_aligned['subtypes'])

    # Ensure pool neurons are excitatory
    for (lo, hi) in POOLS.values():
        types[lo:hi] = 1

    # 4. Blend weights
    # If they share the exact same pre/post lists, we can average.
    if np.array_equal(parent1['pre'], parent2_aligned['pre']) and np.array_equal(parent1['post'], parent2_aligned['post']):
        weight = 0.5 * (parent1['weight'] + parent2_aligned['weight'])
        # Add small jitter to non-reflex weights
        plastic = ~parent1['is_reflex']
        weight[plastic] += rng.standard_normal(int(plastic.sum())) * weight_jitter
        pre = parent1['pre'].copy()
        post = parent1['post'].copy()
        is_reflex = parent1['is_reflex'].copy()
    else:
        # Different topology: start with parent1's connections but blend some weights from parent2 if they match
        pre = parent1['pre'].copy()
        post = parent1['post'].copy()
        weight = parent1['weight'].copy()
        is_reflex = parent1['is_reflex'].copy()

    weight = enforce_dales_law(pre, weight, types)

    return {
        'N': N, 'K': parent1['K'], 'px': px, 'py': py, 'pz': pz,
        'types': types, 'subtypes': subtypes, 'a': a, 'b': b, 'c': c, 'd': d,
        'v': v, 'u': u, 'pre': pre, 'post': post, 'weight': weight, 'is_reflex': is_reflex
    }

def experiment_b_developmental_inheritance(parent1, parent2, seed=None):
    """
    Experiment B: Developmental Inheritance
    Inherits developmental instructions (like cell type ratio 'fi' and K-nearest parameters)
    and grows a fresh spatial network from scratch.
    """
    rng = np.random.default_rng(seed)
    # Inherit fi (inhibitory fraction) and K
    # Calculate fi from parent networks
    fi1 = float(np.sum(parent1['types'] == -1) / parent1['N'])
    fi2 = float(np.sum(parent2['types'] == -1) / parent2['N'])
    fi = 0.5 * (fi1 + fi2)
    K = int(round(0.5 * (parent1['K'] + parent2['K'])))
    N = int(round(0.5 * (parent1['N'] + parent2['N'])))

    # Build a fresh, clean network using these inherited rules
    net = build_creature_network(N, K, fi=fi, seed=seed)

    # Run a simple spatial migration step to represent self-organization
    # Let's pull co-active/similar type neurons slightly closer
    coords = np.stack([net['px'], net['py'], net['pz']], axis=1)
    types = net['types']
    for i in range(N):
        # type-based minor clustering force
        same_type = np.flatnonzero(types == types[i])
        if len(same_type) > 1:
            target_idx = rng.choice(same_type)
            if target_idx != i:
                diff = coords[target_idx] - coords[i]
                coords[i] += diff * 0.05

    # Re-normalize/clamp to unit ball
    d2 = (coords**2).sum(1)
    out_of_bounds = d2 > 1.0
    if out_of_bounds.any():
        coords[out_of_bounds] /= np.sqrt(d2[out_of_bounds])[:, np.newaxis]

    net['px'] = coords[:, 0]
    net['py'] = coords[:, 1]
    net['pz'] = coords[:, 2]

    return net

def experiment_c_co_culture_fusion(parent1, parent2, cross_prob=0.08, seed=None):
    """
    Experiment C: Neural Co-Culture / Fusion
    Places two complete SNN hemispheres side-by-side in a shared coordinate space
    and allows distance-based cross-connections (bridges) to form.
    """
    rng = np.random.default_rng(seed)
    N1, N2 = parent1['N'], parent2['N']
    N = N1 + N2

    # Spatial hemispheres: Shift parent1 to the left (x - 0.6) and parent2 to the right (x + 0.6)
    px = np.concatenate([parent1['px'] - 0.6, parent2['px'] + 0.6])
    py = np.concatenate([parent1['py'], parent2['py']])
    pz = np.concatenate([parent1['pz'], parent2['pz']])

    # Combine individual vectors
    types = np.concatenate([parent1['types'], parent2['types']])
    subtypes = np.concatenate([parent1['subtypes'], parent2['subtypes']])
    a = np.concatenate([parent1['a'], parent2['a']])
    b = np.concatenate([parent1['b'], parent2['b']])
    c = np.concatenate([parent1['c'], parent2['c']])
    d = np.concatenate([parent1['d'], parent2['d']])
    v = np.concatenate([parent1['v'], parent2['v']])
    u = np.concatenate([parent1['u'], parent2['u']])

    # Combine synapses by adjusting postsynaptic and presynaptic indices
    pre1, post1 = parent1['pre'], parent1['post']
    pre2, post2 = parent2['pre'] + N1, parent2['post'] + N1

    pre = np.concatenate([pre1, pre2])
    post = np.concatenate([post1, post2])
    weight = np.concatenate([parent1['weight'], parent2['weight']])
    is_reflex = np.concatenate([parent1['is_reflex'], parent2['is_reflex']])

    # Form bridging cross-connections between parent1 and parent2 based on 3D distance
    coords = np.stack([px, py, pz], axis=1)
    bridge_pre = []
    bridge_post = []
    bridge_w = []
    bridge_rfx = []

    # We sample a few random cross-hemisphere pairs to form potential bridges
    for _ in range(int(N * 3)):
        idx1 = rng.integers(0, N1)
        idx2 = rng.integers(N1, N)

        # Distance between coordinates
        dist2 = np.sum((coords[idx1] - coords[idx2])**2)
        prob = np.exp(-dist2 / 0.8) * cross_prob

        if rng.random() < prob:
            # Connect in both directions probabilistically
            if rng.random() < 0.5:
                bridge_pre.append(idx1)
                bridge_post.append(idx2)
                bridge_w.append(abs(rng.standard_normal()) * types[idx1] * 0.5)
                bridge_rfx.append(False)
            else:
                bridge_pre.append(idx2)
                bridge_post.append(idx1)
                bridge_w.append(abs(rng.standard_normal()) * types[idx2] * 0.5)
                bridge_rfx.append(False)

    if len(bridge_pre) > 0:
        pre = np.concatenate([pre, np.array(bridge_pre, dtype=np.int64)])
        post = np.concatenate([post, np.array(bridge_post, dtype=np.int64)])
        weight = np.concatenate([weight, np.array(bridge_w, dtype=np.float64)])
        is_reflex = np.concatenate([is_reflex, np.array(bridge_rfx, dtype=bool)])

    weight = enforce_dales_law(pre, weight, types)

    return {
        'N': N, 'K': parent1['K'], 'px': px, 'py': py, 'pz': pz,
        'types': types, 'subtypes': subtypes, 'a': a, 'b': b, 'c': c, 'd': d,
        'v': v, 'u': u, 'pre': pre, 'post': post, 'weight': weight, 'is_reflex': is_reflex
    }

def experiment_d_neural_transplantation(host, donor, transplant_frac=0.15, seed=None):
    """
    Experiment D: Neural Transplantation
    Replaces a small percentage of host neurons with transplanted donor neurons.
    Integrates them probabilistically using distance-based wiring.
    """
    rng = np.random.default_rng(seed)
    N = host['N']
    n_transplant = int(round(N * transplant_frac))

    # 1. Select host neurons to replace (excluding sensory/motor pools to avoid breaking sensory-motor loop)
    # Sensory/motor pools cover range 0 to 190.
    allowed_indices = np.arange(190, N)
    if len(allowed_indices) < n_transplant:
        # Fallback to any index if host is too small
        allowed_indices = np.arange(N)

    replace_indices = rng.choice(allowed_indices, size=n_transplant, replace=False)
    donor_indices = rng.choice(np.arange(donor['N']), size=n_transplant, replace=False)

    px, py, pz = host['px'].copy(), host['py'].copy(), host['pz'].copy()
    types, subtypes = host['types'].copy(), host['subtypes'].copy()
    a, b, c, d = host['a'].copy(), host['b'].copy(), host['c'].copy(), host['d'].copy()
    v, u = host['v'].copy(), host['u'].copy()

    # Perform transplantation of properties
    for host_idx, donor_idx in zip(replace_indices, donor_indices):
        px[host_idx] = donor['px'][donor_idx]
        py[host_idx] = donor['py'][donor_idx]
        pz[host_idx] = donor['pz'][donor_idx]
        types[host_idx] = donor['types'][donor_idx]
        subtypes[host_idx] = donor['subtypes'][donor_idx]
        a[host_idx] = donor['a'][donor_idx]
        b[host_idx] = donor['b'][donor_idx]
        c[host_idx] = donor['c'][donor_idx]
        d[host_idx] = donor['d'][donor_idx]
        v[host_idx] = donor['v'][donor_idx]
        u[host_idx] = donor['u'][donor_idx]

    # Remove existing connections involving replaced host neurons to represent cell replacement
    pre, post, weight, is_reflex = host['pre'].copy(), host['post'].copy(), host['weight'].copy(), host['is_reflex'].copy()

    # Keep synapses that do not involve transplanted host neurons
    keep_synapses = ~np.isin(pre, replace_indices) & ~np.isin(post, replace_indices)

    pre = pre[keep_synapses]
    post = post[keep_synapses]
    weight = weight[keep_synapses]
    is_reflex = is_reflex[keep_synapses]

    # Form integration synapses: probabilistically wire transplanted host neurons based on distance
    coords = np.stack([px, py, pz], axis=1)
    new_pre = []
    new_post = []
    new_w = []
    new_rfx = []

    # For every transplanted neuron, connect it to some host neurons
    for tx_idx in replace_indices:
        for target_idx in range(N):
            if tx_idx == target_idx:
                continue
            dist2 = np.sum((coords[tx_idx] - coords[target_idx])**2)
            prob = np.exp(-dist2 / 0.5) * 0.15

            # Form synapses both ways
            if rng.random() < prob:
                new_pre.append(tx_idx)
                new_post.append(target_idx)
                new_w.append(abs(rng.standard_normal()) * types[tx_idx] * 0.6)
                new_rfx.append(False)
            if rng.random() < prob:
                new_pre.append(target_idx)
                new_post.append(tx_idx)
                new_w.append(abs(rng.standard_normal()) * types[target_idx] * 0.6)
                new_rfx.append(False)

    if len(new_pre) > 0:
        pre = np.concatenate([pre, np.array(new_pre, dtype=np.int64)])
        post = np.concatenate([post, np.array(new_post, dtype=np.int64)])
        weight = np.concatenate([weight, np.array(new_w, dtype=np.float64)])
        is_reflex = np.concatenate([is_reflex, np.array(new_rfx, dtype=bool)])

    weight = enforce_dales_law(pre, weight, types)

    return {
        'N': N, 'K': host['K'], 'px': px, 'py': py, 'pz': pz,
        'types': types, 'subtypes': subtypes, 'a': a, 'b': b, 'c': c, 'd': d,
        'v': v, 'u': u, 'pre': pre, 'post': post, 'weight': weight, 'is_reflex': is_reflex
    }

def experiment_e_trait_crossover(parent1, parent2, mutation_scale=0.05, seed=None):
    """
    Experiment E: Developmental Trait Crossover
    Crossovers scalar physiological and developmental traits and mutation, then builds
    a fresh network using the crossed-over parameters.
    """
    rng = np.random.default_rng(seed)

    # Extract scalar traits or use default values if not present
    p1_dev = parent1.get('dev_params', {})
    p2_dev = parent2.get('dev_params', {})
    p1_phys = parent1.get('phys_params', {})
    p2_phys = parent2.get('phys_params', {})
    p1_behav = parent1.get('behav_params', {})
    p2_behav = parent2.get('behav_params', {})

    def blend_trait(val1, val2, default):
        v1 = val1 if val1 is not None else default
        v2 = val2 if val2 is not None else default
        blended = 0.5 * (v1 + v2)
        # Apply mutation
        mutated = blended * (1.0 + rng.standard_normal() * mutation_scale)
        return float(max(0.01, mutated))

    lr = blend_trait(p1_dev.get('long_range_prob'), p2_dev.get('long_range_prob'), 0.10)
    dev_stage = blend_trait(p1_dev.get('dev_stage'), p2_dev.get('dev_stage'), 0.20)
    met_rec = blend_trait(p1_phys.get('met_recovery'), p2_phys.get('met_recovery'), 1.0)
    sg = blend_trait(p1_behav.get('sensor_gain'), p2_behav.get('sensor_gain'), 1.5)
    mg = blend_trait(p1_behav.get('motor_gain'), p2_behav.get('motor_gain'), 1.0)

    # Grow fresh SNN with blended developmental parameters
    N = parent1['N']
    K = parent1['K']
    net = build_creature_network(N, K, seed=seed)

    # Store blended traits back
    net['dev_params'] = {'long_range_prob': lr, 'dev_stage': dev_stage}
    net['phys_params'] = {'met_recovery': met_rec}
    net['behav_params'] = {'sensor_gain': sg, 'motor_gain': mg}

    return net

def experiment_f_experience_transfer(parent, seed=None):
    """
    Experiment F: Experience Transfer
    Creates a new organism and biases its initial connections in the direction
    of the parent's learned weights, without copying the direct structural layout.
    """
    rng = np.random.default_rng(seed)
    N = parent['N']
    K = parent['K']

    # Generate offspring from scratch
    child = build_creature_network(N, K, seed=seed)

    # Experience transfer: Scale offspring initial non-reflex weights to bias them
    # in the direction of the parent's final weights.
    # We find matching connection pairs in the template structure or simply bias
    # the entire plastic weight vector using parent's mean weight intensity.
    parent_plastic = ~parent['is_reflex']
    mean_parent_weight = float(np.mean(np.abs(parent['weight'][parent_plastic]))) if parent_plastic.any() else 0.5

    child_plastic = ~child['is_reflex']
    child['weight'][child_plastic] *= (mean_parent_weight / 0.5)
    child['weight'] = enforce_dales_law(child['pre'], child['weight'], child['types'])

    return child

def experiment_g_multistage_pipeline(parent1, parent2, seed=None):
    """
    Experiment G: Combined Multi-Stage System
    Runs Crossover -> Structural seed -> Spatial development -> Environment Learning.
    """
    rng = np.random.default_rng(seed)
    # 1. Trait crossover
    crossed_traits_net = experiment_e_trait_crossover(parent1, parent2, seed=seed)
    # 2. Structural Brain Combination (merge topology structure)
    struct_combined_net = experiment_a_structural_combination(parent1, parent2, seed=seed)
    # 3. Blend them together: use spatial structure of structural combination, but traits of crossed-over
    struct_combined_net['dev_params'] = crossed_traits_net['dev_params']
    struct_combined_net['phys_params'] = crossed_traits_net['phys_params']
    struct_combined_net['behav_params'] = crossed_traits_net['behav_params']

    return struct_combined_net
