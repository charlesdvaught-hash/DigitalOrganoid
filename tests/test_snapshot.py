import unittest
import numpy as np
from organoid_simulator.creature_embodied import build_creature_network
from organoid_simulator.snapshot_utils import capture_snapshot, reconstruct_organism

class TestSnapshot(unittest.TestCase):
    def test_capture_and_reconstruct(self):
        net = build_creature_network(200, 4, fi=0.2, seed=123)
        snapshot = capture_snapshot(net, label='test-snapshot')

        self.assertEqual(snapshot['label'], 'test-snapshot')
        self.assertIn('neural', snapshot)
        self.assertIn('positions', snapshot['neural'])
        self.assertEqual(len(snapshot['neural']['positions']), 200)

        reconstructed = reconstruct_organism(snapshot)
        self.assertEqual(reconstructed['N'], 200)
        np.testing.assert_allclose(reconstructed['px'], net['px'], rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(reconstructed['types'], net['types'])
        np.testing.assert_allclose(reconstructed['weight'], net['weight'], rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(reconstructed['is_reflex'], net['is_reflex'])

if __name__ == '__main__':
    unittest.main()
