# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.lora.utils import is_in_target_modules


class TestIsInTargetModules:
    """Tests for is_in_target_modules (deployment-time filter)."""

    def test_none_allows_all(self):
        assert is_in_target_modules("model.layers.0.self_attn.o_proj", None)

    def test_suffix_in_target(self):
        assert is_in_target_modules(
            "model.layers.0.self_attn.o_proj", ["o_proj", "q_proj"]
        )

    def test_suffix_not_in_target(self):
        assert not is_in_target_modules(
            "model.layers.0.self_attn.o_proj", ["q_proj", "k_proj"]
        )

    def test_empty_target_modules(self):
        assert not is_in_target_modules("model.layers.0.self_attn.o_proj", [])

    def test_exact_name_match(self):
        assert is_in_target_modules("dense1", ["dense1", "dense2"])

    def test_exact_name_no_match(self):
        assert not is_in_target_modules("dense3", ["dense1", "dense2"])

    def test_packed_parent_matches_child_target_modules(self):
        assert is_in_target_modules(
            "model.layers.0.mlp.gate_up_proj",
            ["gate_proj", "up_proj"],
            {"gate_up_proj": ["gate_proj", "up_proj"]},
        )

    def test_packed_child_matches_parent_target_modules(self):
        assert is_in_target_modules(
            "model.layers.0.mlp.gate_proj",
            ["gate_up_proj"],
            {"gate_up_proj": ["gate_proj", "up_proj"]},
        )

    def test_fused_parent_matches_child_target_modules(self):
        assert is_in_target_modules(
            "model.layers.0.self_attn.fused_qkv_a_proj",
            ["q_a_proj", "kv_a_proj_with_mqa"],
            {"fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"]},
        )


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_lfm3_vl_lora_uses_language_backbone_projection_names():
    from vllm.lora.utils import parse_fine_tuned_lora_name
    from vllm.model_executor.models.interfaces import supports_lora
    from vllm.model_executor.models.lfm3_vl import Lfm3VlForConditionalGeneration

    cls = Lfm3VlForConditionalGeneration
    assert supports_lora(cls)
    mapper = cls.hf_to_vllm_mapper.get_rename_mapper()
    for projection in ("in_proj", "out_proj"):
        actual, lora_a = parse_fine_tuned_lora_name(
            "base_model.model.model.language_model.layers.0"
            f".conv.{projection}.lora_A.weight",
            mapper,
        )
        assert actual == f"language_model.model.layers.0.short_conv.{projection}"
        assert lora_a

    actual, _ = parse_fine_tuned_lora_name(
        "base_model.model.model.language_model.layers.2.self_attn.q_proj.lora_B.weight",
        mapper,
    )
    assert actual == "language_model.model.layers.2.self_attn.q_proj"
    assert "q_proj" in cls.packed_modules_mapping["qkv_proj"]
    assert cls.hf_to_vllm_mapper.apply_list(
        ["model.vision_tower.layers.0.conv.in_proj.weight"]
    ) == ["vision_tower.layers.0.conv.in_proj.weight"]
