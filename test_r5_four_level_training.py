"""R5 test: train the four-level, target-guarded pruning model."""
import unittest
from r5_terrain_dataset import make_structured_terrain_sample, prepare_sparse_dataset
from r5_training import train_four_level_sparse_model

class R5FourLevelTrainingTests(unittest.TestCase):
    def test_four_level_training_reduces_combined_reconstruction_and_multiscale_loss(self):
        data=prepare_sparse_dataset([make_structured_terrain_sample(h,c,0.6,1.1) for h,c in [(0.55,0.6),(0.65,1.1)]])
        run=train_four_level_sparse_model(data,steps=20,seed=0)
        self.assertLess(run.final_loss,run.initial_loss)

if __name__=='__main__': unittest.main()
