# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only GraniteDoclingHybrid model compatible with HuggingFace weights.

Thin wrapper around Idefics3 that adds:
- Registry entry for the GraniteDoclingHybridForConditionalGeneration architecture
- HF processor loading (Idefics3Processor with the model's GotOcr2ImageProcessor)
- logits_scaling from the GraniteMoeHybrid text config
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

from .idefics3 import (
    Idefics3DummyInputsBuilder as GraniteDoclingHybridDummyInputsBuilder,
    Idefics3ForConditionalGeneration,
    Idefics3MultiModalProcessor as GraniteDoclingHybridMultiModalProcessor,
    Idefics3ProcessingInfo,
)

if TYPE_CHECKING:
    from transformers import ProcessorMixin


class GraniteDoclingHybridProcessingInfo(Idefics3ProcessingInfo):
    """Load an Idefics3Processor wired to the model's GotOcr2ImageProcessor.

    GraniteDoclingHybridProcessor has training-time optimizations (uint8
    pixel transfer, skipped CPU normalize) that are incompatible with
    vLLM's multimodal pipeline.  Instead we construct a standard
    Idefics3Processor whose __call__ produces the float32 normalised
    tensors and 4-D layout that the existing Idefics3 code path expects.
    """

    def get_hf_processor(self, **kwargs: object) -> ProcessorMixin:
        from transformers import (
            AutoImageProcessor,
            AutoTokenizer,
            Idefics3Processor,
        )

        model_id = self.ctx.model_config.model

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        image_processor = AutoImageProcessor.from_pretrained(model_id)

        processor_cfg = self.ctx.model_config.hf_config
        image_seq_len = getattr(processor_cfg, "image_seq_len", None)
        if image_seq_len is None:
            scale_factor = getattr(processor_cfg, "scale_factor", 2)
            vis = processor_cfg.vision_config
            image_seq_len = int(
                ((vis.image_size // vis.patch_size) ** 2)
                / (scale_factor ** 2)
            )

        processor = Idefics3Processor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            image_seq_len=image_seq_len,
        )
        return processor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}


@MULTIMODAL_REGISTRY.register_processor(
    GraniteDoclingHybridMultiModalProcessor,
    info=GraniteDoclingHybridProcessingInfo,
    dummy_inputs=GraniteDoclingHybridDummyInputsBuilder,
)
class GraniteDoclingHybridForConditionalGeneration(
    Idefics3ForConditionalGeneration,
):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        logits_scaling = getattr(
            self.config.text_config, "logits_scaling", 1
        )
        if logits_scaling != 1:
            self.logits_processor.scale = 1.0 / logits_scaling