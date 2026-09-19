"""Small sparse model used only to test the MinkowskiEngine contracts."""

import MinkowskiEngine as ME
from torch import nn

from models.r5_sparse_model import SparseCompletionPredictions


class Minimal4DCompletionModel(ME.MinkowskiNetwork):
    """One downsample/upsample paper-style slice for contract tests."""

    def __init__(self, hidden_channels: int) -> None:
        super().__init__(D=4)
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")

        spatial_stride = (2, 2, 2, 1)
        self.encoder = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=3,
                out_channels=hidden_channels,
                kernel_size=spatial_stride,
                stride=spatial_stride,
                dimension=4,
            ),
            ME.MinkowskiReLU(),
        )
        self.decoder = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=spatial_stride,
                stride=spatial_stride,
                dimension=4,
            ),
            ME.MinkowskiReLU(),
        )
        self.occupancy_head = ME.MinkowskiConvolution(
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=1,
            dimension=4,
        )
        self.position_head = ME.MinkowskiConvolution(
            in_channels=hidden_channels,
            out_channels=3,
            kernel_size=1,
            dimension=4,
        )

    def forward(self, sparse_input: ME.SparseTensor) -> SparseCompletionPredictions:
        latent = self.encoder(sparse_input)
        candidates = self.decoder(latent)
        return SparseCompletionPredictions(
            candidates=candidates,
            occupancy_logits=self.occupancy_head(candidates),
            position_offsets=self.position_head(candidates),
        )
