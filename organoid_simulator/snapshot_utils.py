"""
Snapshot Utilities for the Digital Organoid SNN Model.
Handles JSON serialization/deserialization of full organism state (neural, developmental, physiological, behavioral).
"""
import uuid
from datetime import datetime, timezone
import numpy as np

def capture_snapshot(net, label='organism', dev_params=None, phys_params=None, behav_params=None):
    """
    Serializes a complete digital organism state into a JSON-compatible Python dictionary.

    Args:
        net (dict): The neural network dictionary (containing px, py, pz, types, subtypes, a, b, c, d, v, u, pre, post, weight, is_reflex).
        label (str): Name or label of the snapshot.
        dev_params (dict, optional): Developmental rules and parameters.
        phys_params (dict, optional): Physiological and metabolic parameters.
        behav_params (dict, optional): Behavioral and motor/sensor profiles.

    Returns:
        dict: A fully serializable Python dictionary.
    """
    # 1. Neural State
    positions = []
    px, py, pz = net['px'], net['py'], net['pz']
    for x_val, y_val, z_val in zip(px, py, pz):
        positions.append({'x': float(x_val), 'y': float(y_val), 'z': float(z_val)})

    synapses = []
    pre, post, weight, is_reflex = net['pre'], net['post'], net['weight'], net['is_reflex']
    for p, q, w, r in zip(pre, post, weight, is_reflex):
        synapses.append({
            'pre': int(p),
            'post': int(q),
            'weight': float(w),
            'is_reflex': bool(r)
        })

    params = []
    a, b, c, d = net['a'], net['b'], net['c'], net['d']
    for av, bv, cv, dv in zip(a, b, c, d):
        params.append({'a': float(av), 'b': float(bv), 'c': float(cv), 'd': float(dv)})

    state = []
    v, u = net['v'], net['u']
    for vv, uv in zip(v, u):
        state.append({'v': float(vv), 'u': float(uv)})

    neural = {
        'positions': positions,
        'types': [int(t) for t in net['types']],
        'subtypes': [int(st) for st in net['subtypes']],
        'params': params,
        'synapses': synapses,
        'state': state
    }

    # 2. Developmental, physiological, and behavioral defaults if not provided
    developmental = dev_params or {
        'growth_rules': {'self_org_strength': 1.0, 'rewire_rate': 1.0},
        'long_range_prob': 0.10,
        'dev_stage': 0.20
    }

    physiological = phys_params or {
        'metabolic_rate': 1.0,
        'met_recovery': 1.0
    }

    behavioral = behav_params or {
        'sensor_gain': 1.5,
        'motor_gain': 1.0
    }

    return {
        'id': f"{int(datetime.now(timezone.utc).timestamp() * 1000)}-{str(uuid.uuid4())[:8]}",
        'label': label,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'neural': neural,
        'developmental': developmental,
        'physiological': physiological,
        'behavioral': behavioral
    }

def reconstruct_organism(snapshot):
    """
    Reconstructs the active SNN network dictionary from a serialized snapshot.

    Args:
        snapshot (dict): The serialized state.

    Returns:
        dict: Reconstructed SNN net dictionary.
    """
    neural = snapshot['neural']
    N = len(neural['positions'])

    px = np.array([p['x'] for p in neural['positions']], dtype=np.float32)
    py = np.array([p['y'] for p in neural['positions']], dtype=np.float32)
    pz = np.array([p['z'] for p in neural['positions']], dtype=np.float32)

    types = np.array(neural['types'], dtype=np.int8)
    subtypes = np.array(neural['subtypes'], dtype=np.int8)

    a = np.array([p['a'] for p in neural['params']], dtype=np.float64)
    b = np.array([p['b'] for p in neural['params']], dtype=np.float64)
    c = np.array([p['c'] for p in neural['params']], dtype=np.float64)
    d = np.array([p['d'] for p in neural['params']], dtype=np.float64)

    v = np.array([s['v'] for s in neural['state']], dtype=np.float32)
    u = np.array([s['u'] for s in neural['state']], dtype=np.float32)

    synapses = neural['synapses']
    pre = np.array([s['pre'] for s in synapses], dtype=np.int64)
    post = np.array([s['post'] for s in synapses], dtype=np.int64)
    weight = np.array([s['weight'] for s in synapses], dtype=np.float64)
    is_reflex = np.array([s['is_reflex'] for s in synapses], dtype=bool)

    net = {
        'N': N,
        'px': px,
        'py': py,
        'pz': pz,
        'types': types,
        'subtypes': subtypes,
        'a': a,
        'b': b,
        'c': c,
        'd': d,
        'v': v,
        'u': u,
        'pre': pre,
        'post': post,
        'weight': weight,
        'is_reflex': is_reflex
    }

    # Store other non-neural params inside the net dict for convenience
    net['dev_params'] = snapshot.get('developmental', {})
    net['phys_params'] = snapshot.get('physiological', {})
    net['behav_params'] = snapshot.get('behavioral', {})

    return net
