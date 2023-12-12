from nflows.transforms.splines.cubic import unconstrained_cubic_spline
from nflows.transforms import made as made_module
from nflows.transforms.autoregressive import AutoregressiveTransform
from torch.nn import functional as F
import numpy as np
from nflows.utils import torchutils


class MaskedUnconstrainedPiecewiseCubicAutoregressiveTransform(AutoregressiveTransform):
    def __init__(
        self,
        num_bins,
        features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        self.num_bins = num_bins
        self.features = features
        made = made_module.MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
        super(MaskedUnconstrainedPiecewiseCubicAutoregressiveTransform, self).__init__(made)

    def _output_dim_multiplier(self):
        return self.num_bins * 2 + 2

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size = inputs.shape[0]

        transform_params = autoregressive_params.view(
            batch_size, self.features, self.num_bins * 2 + 2
        )

        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        derivatives = transform_params[..., 2 * self.num_bins :]
        unnorm_derivatives_left = derivatives[..., 0][..., None]
        unnorm_derivatives_right = derivatives[..., 1][..., None]

        if hasattr(self.autoregressive_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.autoregressive_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.autoregressive_net.hidden_features)

        outputs, logabsdet = unconstrained_cubic_spline(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnorm_derivatives_left=unnorm_derivatives_left,
            unnorm_derivatives_right=unnorm_derivatives_right,
            inverse=inverse,
            tail_bound=1.0,
            tails="linear",
        )
        return outputs, torchutils.sum_except_batch(logabsdet)

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)