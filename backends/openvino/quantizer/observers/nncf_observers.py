# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from torch.ao.quantization.observer import MappingType, PerGroup, PerAxis, PerChannelMinMaxObserver, get_block_size
from torch.ao.quantization.pt2e._affine_quantization import (
    _get_reduction_params,
    AffineQuantizedMinMaxObserver,
)
from nncf.torch.quantization.layers import INT4AsymmetricWeightsDecompressor, INT4SymmetricWeightsDecompressor, INT8AsymmetricWeightsDecompressor, INT8SymmetricWeightsDecompressor
from nncf.experimental.torch.fx.transformations import constant_update_fn, module_insertion_transformation_builder
from nncf.experimental.torch.fx.node_utils import get_tensor_constant_from_node
from nncf.torch.graph.transformations.commands import PTTargetPoint, TargetType

from nncf.quantization.algorithms.weight_compression.weight_lowering import do_integer_quantization
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.parameters import CompressWeightsMode
from nncf.tensor.tensor import Tensor

class PTWeightCompressionObserverBase:
    def __init__(self):
        self.wc_config = None

    def calculate_qparams(self, weight):
        assert hasattr(self, "min_val") and hasattr(
            self, "max_val"
        ), "Observer must be run before calculate_qparams"

        self.block_size = get_block_size(weight.shape, self.granularity)
        _, reduction_dims = _get_reduction_params(self.block_size, weight.size())
        reduction_dims = reduction_dims[0] - 1 if isinstance(self.granularity, PerGroup) else reduction_dims

        q_weight, scale, zp = do_integer_quantization(
            Tensor(weight), self.wc_config, reduction_axes=reduction_dims
        )
        zp = zp.data if zp is not None else None
        return q_weight.data, scale.data, zp

    def convert(self, model: torch.fx.GraphModule, observer_node: torch.fx.Node):
        print("calling convert")
        weight_node = observer_node.args[0]
        original_weight = get_tensor_constant_from_node(weight_node, model)
        q_weight, scale, zero_point = self.calculate_qparams(original_weight)

        with model.graph.inserting_before(observer_node):
            decompressor = self._create_decompressor(
                scale, zero_point, q_weight, original_weight
            )
            packed_q_weight = decompressor.pack_weight(q_weight)

            constant_update_fn(model, observer_node, packed_q_weight, input_port_id=0)

            compressed_weight_name = observer_node.all_input_nodes[0].name
            decompressor_suffix = "_".join(
                compressed_weight_name.replace(".", "_").split("_")[:-2]
            )
            decompressor_name = f"{decompressor.quantization_mode}_weights_decompressor_{decompressor_suffix}"

            module_insertion_transformation_builder(
                decompressor,
                [
                    PTTargetPoint(
                        TargetType.OPERATOR_POST_HOOK,
                        target_node_name=compressed_weight_name,
                    )
                ],
                decompressor_name,
            )(model)

        decomp_node = observer_node.args[0]
        observer_node.replace_all_uses_with(decomp_node)
        model.graph.erase_node(observer_node)

    def _create_decompressor(self, scale, zero_point, q_weight, original_weight):
        raise NotImplementedError("Must be implemented by subclasses")


class PTPerBlockParamObserver(PTWeightCompressionObserverBase, AffineQuantizedMinMaxObserver):
    def __init__(self, *args, **kwargs):
        PTWeightCompressionObserverBase.__init__(self)
        AffineQuantizedMinMaxObserver.__init__(self, *args, **kwargs)

        assert isinstance(self.granularity, PerGroup), "Only PerGroup granularity is supported"
        qmode = (
            CompressWeightsMode.INT4_ASYM
            if self.mapping_type == MappingType.ASYMMETRIC
            else CompressWeightsMode.INT4_SYM
        )
        self.wc_config = WeightCompressionConfig(mode=qmode, group_size=self.granularity.group_size)

    def _create_decompressor(self, scale, zero_point, q_weight, original_weight):
        if zero_point is not None:
            return INT4AsymmetricWeightsDecompressor(
                scale, zero_point, q_weight.shape, original_weight.shape, original_weight.dtype
            )
        else:
            return INT4SymmetricWeightsDecompressor(
                scale, q_weight.shape, original_weight.shape, original_weight.dtype
            )


class NNCFInt8observer(PTWeightCompressionObserverBase, PerChannelMinMaxObserver):
    def __init__(self, *args, **kwargs):
        PTWeightCompressionObserverBase.__init__(self)
        PerChannelMinMaxObserver.__init__(self, *args, **kwargs)

        qmode = (
            CompressWeightsMode.INT8_SYM
            if self.qscheme == torch.per_channel_symmetric
            else CompressWeightsMode.INT8_ASYM
        )
        self.wc_config = WeightCompressionConfig(mode=qmode)
        self.granularity = PerAxis(axis=self.ch_axis)

    def _create_decompressor(self, scale, zero_point, q_weight, original_weight):
        if zero_point is not None:
            return INT8AsymmetricWeightsDecompressor(scale, zero_point, original_weight.dtype)
        else:
            return INT8SymmetricWeightsDecompressor(scale, original_weight.dtype)
