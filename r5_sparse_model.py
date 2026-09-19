"""R5: a smallest real 4D sparse encoder-decoder slice of the paper method.

Real call path:
    SparseTensor [batch, x, y, z, k] + voxel-offset features
    -> spatial downsample (time k is kept)
    -> generative spatial upsample (creates candidate voxels)
    -> occupancy logit and 3D offset prediction for every candidate

This is intentionally one encoder/decoder level, not yet the paper's four
levels or its skip connections. Its job is to validate the exact 4D sparse
operations before we scale the architecture and add the paper losses.
"""

from dataclasses import dataclass

import torch
from torch import nn
import MinkowskiEngine as ME

from paper_config import IMPLEMENTATION_CHANNELS, IMPLEMENTATION_GENERATIVE_KERNEL, IMPLEMENTATION_SPATIAL_STRIDE


class EmptyPruningError(RuntimeError):
    """Raised instead of asking MinkowskiEngine to construct an unsafe empty tensor."""


@dataclass(frozen=True)
class SparseCompletionPredictions:
    """Two paper outputs defined on the same candidate 4D voxel coordinates.

    candidates: generated candidate locations, [batch, x, y, z, k].
    occupancy_logits: one unnormalized occupied/not-occupied score per location.
    position_offsets: three predicted within-voxel coordinates per location.
    """

    candidates: ME.SparseTensor
    occupancy_logits: ME.SparseTensor
    position_offsets: ME.SparseTensor
    decoder_likelihoods: tuple[ME.SparseTensor, ...] = ()


def align_skip_to_candidates(
    skip: ME.SparseTensor,
    candidates: ME.SparseTensor,
) -> ME.SparseTensor:
    """Express encoder skip features on generated candidate coordinates.

    Existing coordinates retrieve their encoder feature. New candidate locations
    receive zeros, allowing a real skip concat without pretending they were seen.
    """
    features = skip.features_at_coordinates(candidates.C.float())
    return ME.SparseTensor(
        features=features,
        coordinate_map_key=candidates.coordinate_map_key,
        coordinate_manager=candidates.coordinate_manager,
    )


class Minimal4DCompletionModel(ME.MinkowskiNetwork):
    """One downsample/upsample paper-style slice for an input with 3 features."""

    def __init__(self, hidden_channels: int) -> None:
        super().__init__(D=4)
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")

        # Stride [2, 2, 2, 1]: halve x/y/z resolution, preserve time k.
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
        # Generative transpose convolution can create unobserved spatial candidates.
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
        """Run the minimal 4D network and keep both prediction heads aligned."""
        latent = self.encoder(sparse_input)
        candidates = self.decoder(latent)
        return SparseCompletionPredictions(
            candidates=candidates,
            occupancy_logits=self.occupancy_head(candidates),
            position_offsets=self.position_head(candidates),
        )


class FourLevel4DCompletionModel(ME.MinkowskiNetwork):
    """Paper-shaped 4D U-Net: four spatial scales and aligned skip connections.

    The paper does not publish the channel widths or exact kernel sizes.  Those
    values are therefore named implementation defaults in ``paper_config.py``
    and must not be presented as recovered author settings.
    """

    def __init__(
        self,
        channels: tuple[int, int, int, int] = IMPLEMENTATION_CHANNELS,
        generative_kernels: tuple[tuple[int, int, int, int], ...] = (IMPLEMENTATION_GENERATIVE_KERNEL,) * 4,
    ) -> None:
        super().__init__(D=4)
        if len(channels) != 4 or any(value <= 0 for value in channels):
            raise ValueError("channels must contain four positive values")
        if len(generative_kernels) != 4:
            raise ValueError("generative_kernels must contain four 4D kernels")
        c1, c2, c3, c4 = channels
        stride = IMPLEMENTATION_SPATIAL_STRIDE
        self.enc1 = _down_block(3, c1, stride)
        self.enc2 = _down_block(c1, c2, stride)
        self.enc3 = _down_block(c2, c3, stride)
        self.enc4 = _down_block(c3, c4, stride)
        self.up3, self.dec3 = _up_and_fuse_blocks(c4, c3, c3, stride, generative_kernels[0])
        self.up2, self.dec2 = _up_and_fuse_blocks(c3, c2, c2, stride, generative_kernels[1])
        self.up1, self.dec1 = _up_and_fuse_blocks(c2, c1, c1, stride, generative_kernels[2])
        self.up0, self.dec0 = _up_and_fuse_blocks(c1, 3, c1, stride, generative_kernels[3])
        self.decoder_likelihood_heads = nn.ModuleList([
            # Bias lets each occupancy head first represent the highly sparse
            # occupied prior before feature-dependent separation is learned.
            # Without it, BCE gradients averaged over hundreds of thousands of
            # generated candidates leave logits pinned near zero.
            ME.MinkowskiConvolution(c3, 1, kernel_size=1, dimension=4, bias=True),
            ME.MinkowskiConvolution(c2, 1, kernel_size=1, dimension=4, bias=True),
            ME.MinkowskiConvolution(c1, 1, kernel_size=1, dimension=4, bias=True),
            ME.MinkowskiConvolution(c1, 1, kernel_size=1, dimension=4, bias=True),
        ])
        self.position_head = ME.MinkowskiConvolution(c1, 3, kernel_size=1, dimension=4, bias=True)

    def forward(
        self,
        sparse_input: ME.SparseTensor,
        alpha: float | None = None,
        training_target_coordinates: tuple[torch.Tensor, ...] | None = None,
    ) -> SparseCompletionPredictions:
        """Run four decoder scales, optionally pruning between successive scales.

        `training_target_coordinates` is ordered for strides 8/4/2/1. It is
        used only as a target guard during supervised training, never inference.
        """
        if training_target_coordinates is not None and len(training_target_coordinates) != 4:
            raise ValueError("training_target_coordinates must contain four scales")
        e1 = self.enc1(sparse_input)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        latent = self.enc4(e3)
        u3 = self.up3(latent)
        d3 = self.dec3(ME.cat(u3, align_skip_to_candidates(e3, u3)))
        l3 = self.decoder_likelihood_heads[0](d3)
        d3_next = _maybe_prune(d3, l3, alpha, _target_at(training_target_coordinates, 0))
        u2 = self.up2(d3_next)
        d2 = self.dec2(ME.cat(u2, align_skip_to_candidates(e2, u2)))
        l2 = self.decoder_likelihood_heads[1](d2)
        d2_next = _maybe_prune(d2, l2, alpha, _target_at(training_target_coordinates, 1))
        u1 = self.up1(d2_next)
        d1 = self.dec1(ME.cat(u1, align_skip_to_candidates(e1, u1)))
        l1 = self.decoder_likelihood_heads[2](d1)
        d1_next = _maybe_prune(d1, l1, alpha, _target_at(training_target_coordinates, 2))
        u0 = self.up0(d1_next)
        candidates = self.dec0(ME.cat(u0, align_skip_to_candidates(sparse_input, u0)))
        l0 = self.decoder_likelihood_heads[3](candidates)
        # The final prediction is the current-frame (k=0) completion only. The
        # temporal encoder propagates k=1 voxels through the decoder, but they are
        # not part of the output (completion_loss matches k=0 targets only).
        k0_mask = candidates.C[:, 4] == 0
        if not bool(k0_mask.any()):
            raise EmptyPruningError("decoder produced no k=0 candidates")
        candidates = ME.MinkowskiPruning()(candidates, k0_mask)
        l0 = ME.MinkowskiPruning()(l0, k0_mask)
        candidates, final_likelihood = _prune_final_likelihood(
            candidates, l0, alpha, _target_at(training_target_coordinates, 3)
        )
        # NOTE: l0 is the final occupancy head (supervised once via completion_loss);
        # only the intermediate decoder likelihoods l3/l2/l1 receive the auxiliary
        # per-layer BCE supervision (otherwise l0 would be pulled by two conflicting
        # objectives with different class weights).
        likelihoods = (l3, l2, l1)
        return SparseCompletionPredictions(candidates, final_likelihood, self.position_head(candidates), likelihoods)


def _down_block(
    in_channels: int, out_channels: int, stride: tuple[int, int, int, int]
) -> nn.Sequential:
    # kernel spans the temporal axis (length 2 over k) so a k=0 output reads its
    # k=1 neighbours -- the paper's "across space and time" 4D convolution.
    # stride keeps the temporal dimension at 1 (time resolution preserved).
    return nn.Sequential(
        ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=(2, 2, 2, 2), stride=stride, dimension=4),
        ME.MinkowskiReLU(),
    )


def _up_and_fuse_blocks(
    in_channels: int,
    skip_channels: int,
    out_channels: int,
    stride: tuple[int, int, int, int],
    generative_kernel: tuple[int, int, int, int] = (2, 2, 2, 1),
) -> tuple[nn.Sequential, nn.Sequential]:
    # The generative transpose upsamples only in space (temporal kernel = 1):
    # the decoder must emit k=0 candidates only. A temporal kernel > 1 here
    # produced k=1..5 candidates which completion_loss (matching k=0 targets)
    # then labelled as negatives, collapsing the positive rate from ~6% to ~1%.
    up = nn.Sequential(
        ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels, out_channels, kernel_size=generative_kernel, stride=stride, dimension=4
        ),
        ME.MinkowskiReLU(),
    )
    fuse = nn.Sequential(ME.MinkowskiConvolution(out_channels + skip_channels, out_channels, kernel_size=1, dimension=4), ME.MinkowskiReLU())
    return up, fuse


def _target_at(targets: tuple[torch.Tensor, ...] | None, index: int) -> torch.Tensor | None:
    return None if targets is None else targets[index]


def _maybe_prune(
    candidates: ME.SparseTensor,
    likelihood: ME.SparseTensor,
    alpha: float | None,
    target_coordinates: torch.Tensor | None,
) -> ME.SparseTensor:
    if alpha is None:
        return candidates
    return prune_candidates(candidates, likelihood.F, alpha, target_coordinates)


def _prune_final_likelihood(
    candidates: ME.SparseTensor,
    likelihood: ME.SparseTensor,
    alpha: float | None,
    target_coordinates: torch.Tensor | None,
) -> tuple[ME.SparseTensor, ME.SparseTensor]:
    """Apply one final likelihood to both pruning and final occupancy output."""
    if alpha is None:
        return candidates, likelihood
    keep_mask = _pruning_mask(candidates, likelihood.F, alpha, target_coordinates)
    pruning = ME.MinkowskiPruning()
    return pruning(candidates, keep_mask), pruning(likelihood, keep_mask)


def prune_candidates(
    candidates: ME.SparseTensor,
    occupancy_logits: torch.Tensor,
    alpha: float,
    target_coordinates: torch.Tensor | None = None,
) -> ME.SparseTensor:
    """Keep candidates whose sigmoid occupancy likelihood is at least alpha.

    `occupancy_logits` is passed separately so tests and training can inspect the
    score-to-mask rule without losing the candidate tensor itself.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if occupancy_logits.shape != (len(candidates.C), 1):
        raise ValueError("occupancy_logits must have one value per candidate")

    keep_mask = _pruning_mask(candidates, occupancy_logits, alpha, target_coordinates)
    return ME.MinkowskiPruning()(candidates, keep_mask)


def _pruning_mask(
    candidates: ME.SparseTensor,
    occupancy_logits: torch.Tensor,
    alpha: float,
    target_coordinates: torch.Tensor | None,
) -> torch.Tensor:
    """Create the paper's likelihood mask, plus the supervised target guard."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if occupancy_logits.shape != (len(candidates.C), 1):
        raise ValueError("occupancy_logits must have one value per candidate")
    keep_mask = torch.sigmoid(occupancy_logits[:, 0]) >= alpha
    if target_coordinates is not None:
        target_keys = {tuple(int(value) for value in row) for row in target_coordinates.cpu().numpy()}
        protected = torch.tensor(
            [tuple(int(value) for value in row) in target_keys for row in candidates.C.cpu().numpy()],
            dtype=torch.bool,
            device=keep_mask.device,
        )
        keep_mask |= protected
    if not bool(keep_mask.any()):
        raise EmptyPruningError("pruning removed every candidate")
    return keep_mask
