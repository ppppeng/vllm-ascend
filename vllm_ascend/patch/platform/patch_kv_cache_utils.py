from collections import defaultdict

import vllm
from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_groups_uniform_page_size, _get_kv_cache_groups_uniform_spec, _auto_fit_max_model_len,
    _get_kv_cache_groups_uniform_type, create_kv_cache_group_specs, _check_enough_kv_cache_memory,
    get_num_blocks, get_uniform_page_size, is_kv_cache_spec_uniform, _max_memory_usage_bytes_from_groups,
    is_kv_cache_type_attention_free, may_override_num_blocks, _estimate_max_model_len_from_groups,
    unify_hybrid_kv_cache_specs, unify_kv_cache_spec_page_size, _report_kv_cache_config)
from vllm.v1.kv_cache_interface import (KVCacheConfig, KVCacheGroupSpec,
                                        KVCacheSpec, KVCacheTensor,
                                        UniformTypeKVCacheSpecs)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

from vllm_ascend.patch.platform.patch_kv_cache_coordinator import \
    USE_MULTI_BLOCK_POOL


def _get_kv_cache_groups_uniform_block_size(
    kv_cache_spec: dict[str, KVCacheSpec], ) -> list[KVCacheGroupSpec]:
    '''
    Generates the KV cache groups with same block size,
    and there maybe multiple groups with different spec,
    each group has their own block_pool and each layer
    of each group has their own kv_cache_tensor.

    :param kv_cache_spec: The KVCacheSpecs of all the layers
    :type kv_cache_spec: dict[str, KVCacheSpec]
    :return: a list of KVCacheGroupSpecs, there is one type of KVCacheSpec in each group 
    :rtype: list[KVCacheGroupSpec]
    '''
    same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
    _, first_kv_cache_config = next(iter(kv_cache_spec.items()))
    block_size = first_kv_cache_config.block_size
    for layer_name, layer_spec in kv_cache_spec.items():
        # assert block_size == layer_spec.block_size, "Layer block size is not equal."
        same_type_layers[layer_spec].append(layer_name)
    grouped_layers = list(same_type_layers.values())
    return create_kv_cache_group_specs(kv_cache_spec, grouped_layers)


def check_uniform_page_size(kv_cache_groups: list[KVCacheGroupSpec]) -> bool:
    kv_cache_specs = [group.kv_cache_spec for group in kv_cache_groups]
    page_sizes = {layer.page_size_bytes for layer in kv_cache_specs}
    return len(page_sizes) == 1


def get_kv_cache_groups(
        vllm_config: VllmConfig,
        kv_cache_spec: dict[str, KVCacheSpec]) -> list[KVCacheGroupSpec]:
    """
    Split the layers in the model into groups with the same KV cache spec.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroups
    """

    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        unify_hybrid_kv_cache_specs(kv_cache_spec)

    if is_kv_cache_type_attention_free(kv_cache_spec):
        # This returns an empty list to allow for the KVCacheManager to handle
        # attention free models.
        return []

    if is_kv_cache_spec_uniform(kv_cache_spec):
        # KV cache of all layers are the same, which is true for
        # most models. Allocate the same amount of memory for
        # each layer.
        return _get_kv_cache_groups_uniform_spec(kv_cache_spec)
    elif USE_MULTI_BLOCK_POOL:
        # kv cache group spec with multi groups and same block size without share hybrid blocks
        return _get_kv_cache_groups_uniform_block_size(kv_cache_spec)
    elif uniform_spec := UniformTypeKVCacheSpecs.from_specs(kv_cache_spec):
        # All layers need the same number of token slots (e.g., all layers are
        # full attention, or all layers are sliding window attention with the
        # same window size). Put all layers into one group.
        return _get_kv_cache_groups_uniform_type(uniform_spec)

    # As KVCacheManager can only allocate memory of one size, we need to unify
    # the page size of the layers. For cases cannot be unified, this function
    # will raise an error.
    kv_cache_spec = unify_kv_cache_spec_page_size(kv_cache_spec)
    # Model contains multiple attention types, but KV cache of all layers
    # have the same physical memory per block per layer. Split the layers
    # into groups with the same number of layers, and thus same total page
    # size.
    return _get_kv_cache_groups_uniform_page_size(kv_cache_spec)


def get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """
    Generate the KV cache configuration from the KV cache groups and spec
    of each layer.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_groups: The KV cache groups
        available_memory: Memory available for KV cache in bytes
    Returns:
        The generated KVCacheConfig
    """
    if len(kv_cache_groups) == 0:
        # Attention free models do not have KV cache.
        # Return num_blocks=1 as BlockPool always needs a null_block.
        return KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=kv_cache_groups,
        )

    # Determine how model runners should initialize the KV cache tensors.
    if len(kv_cache_groups) == 1 and isinstance(
            kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs):
        # Special case: all layers have the same type of KV cache but with
        # different hidden size. Allocate different amount of memory for each
        # layer based on its hidden size.
        num_blocks = (available_memory //
                      kv_cache_groups[0].kv_cache_spec.page_size_bytes)
        num_blocks = may_override_num_blocks(vllm_config, num_blocks)
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
        kv_cache_tensors = [
            KVCacheTensor(
                size=per_layer_specs[layer_name].page_size_bytes * num_blocks,
                shared_by=[layer_name],
            ) for layer_name in kv_cache_groups[0].layer_names
        ]
    elif USE_MULTI_BLOCK_POOL:
        # USE_MULTI_BLOCKPOOL
        # Special case: there are multiple groups of KV cache, and the block
        # size of them is different. We will still have `num_kv_cache_groups` memory
        # pools, this means the memory pools won't be shared across the groups.
        # For linear layers such as MambaLayer, we provide `max_num_seqs + 1` blocks for them in order
        # to save more memory blocks for full_attn layers.
        attn_page_size_bytes = 0
        linear_memory_size = 0
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        num_spec_token = vllm_config.speculative_config.num_speculative_tokens if vllm_config.speculative_config is not None else 0
        num_linear_blocks = (max_num_seqs + 1) * 2 * (num_spec_token + 1)
        for kv_cache_group in kv_cache_groups:
            num_layers = len(kv_cache_group.layer_names)
            page_size = kv_cache_group.kv_cache_spec.page_size_bytes
            if isinstance(kv_cache_group.kv_cache_spec, AttentionSpec):
                attn_page_size_bytes += page_size * num_layers
            elif isinstance(kv_cache_group.kv_cache_spec, MambaSpec):
                linear_memory_size += page_size * num_layers
            else:
                continue
        num_blocks = num_attn_blocks = (available_memory - linear_memory_size * num_linear_blocks) // attn_page_size_bytes
        assert num_attn_blocks > 0
        kv_cache_tensors = []
        for i in range(len(kv_cache_groups)):
            if isinstance(kv_cache_groups[i].kv_cache_spec, AttentionSpec):
                tensor_size = kv_cache_groups[i].kv_cache_spec.page_size_bytes * num_attn_blocks
            elif isinstance(kv_cache_groups[i].kv_cache_spec, MambaSpec):
                tensor_size = kv_cache_groups[i].kv_cache_spec.page_size_bytes * num_linear_blocks
            else:
                continue
            for layer_name in kv_cache_groups[i].layer_names:
                # NOTE(zxr): each layer has its own kv_cache tensor
                shared_by = [layer_name]
                kv_cache_tensors.append(
                    KVCacheTensor(
                        size=tensor_size,
                        shared_by=shared_by))
    else:
        # General case:
        # We will have group_size memory pools, each is shared by one layer from
        # each group. As layers of different groups have different block table,
        # they will use different parts of the shared Tensor.
        # The memory layout for 3 groups (full.0, full.1), (sw.0, sw.2),
        # (sw.1, padding) will be: (group_size = 2)
        # full.0, sw.0, sw.1: share a Tensor with size=available_memory//2
        # full.1, sw.2: share another Tensor with size=available_memory//2
        group_size = max(len(group.layer_names) for group in kv_cache_groups)

        page_size = get_uniform_page_size(
            [group.kv_cache_spec for group in kv_cache_groups])
        assert group_size > 0, "group_size must be greater than 0"
        num_blocks = get_num_blocks(vllm_config, group_size, available_memory,
                                    page_size)
        kv_cache_tensors = []
        for i in range(group_size):
            shared_by = []
            for j in range(len(kv_cache_groups)):
                if i < len(kv_cache_groups[j].layer_names):
                    shared_by.append(kv_cache_groups[j].layer_names[i])
            kv_cache_tensors.append(
                KVCacheTensor(size=page_size * num_blocks,
                              shared_by=shared_by))

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )


def get_kv_cache_configs(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> list[KVCacheConfig]:
    """
    Generates the KV cache configurations for a model.
    Since we use a shared centralized controller for all workers, we need the
    `kv_cache_config` to be consistent across all workers to make sure
    the KV cache allocation can be applied to all workers. However, different
    workers may have different memory available, and different type of layers
    (when pipeline parallel is enabled). To handle the difference between
    workers, the current implementation is:
    1. Merge the KV cache specs of all workers to get the KVCacheSpecs for
       the whole model.
    2. Generate the KV cache groups based on the layer ratio of the whole model.
       This also handles spec unification for hybrid models.
    3. Handle auto-fit max_model_len and memory checks using the unified specs.
    4. Generate the KV cache configs for each worker based on the KV cache
       grouping strategy. (This is reasonable because the layer ratio of
       different PP stages are similar.)
    5. Change the num_blocks of each worker to the smallest among all workers
       and shrink tensor sizes proportionally to avoid allocating unused memory.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_specs: List of dict[layer_name, KVCacheSpec] for each worker.
        available_memory: Memory available for KV cache in bytes for each
            worker.

    Returns:
        The generated KVCacheConfigs for each worker.
    """

    # Merge the KV cache specs of all workers. Different PP stages may have
    # different layer names, and different TP ranks of the same PP stage should
    # have the same KV cache spec.
    merged_kv_cache_specs: dict[str, KVCacheSpec] = {}
    for kv_cache_spec_one_worker in kv_cache_specs:
        for layer_name, layer_spec in kv_cache_spec_one_worker.items():
            if layer_name not in merged_kv_cache_specs:
                merged_kv_cache_specs[layer_name] = layer_spec
            else:
                assert merged_kv_cache_specs[layer_name] == layer_spec, (
                    "The KV cache specs for the same layer are different "
                    "across workers. This is not supported yet."
                )

    # Get global KV cache groups. This also handles spec unification for
    # hybrid models when disable_hybrid_kv_cache_manager is enabled.
    # After this call, merged_kv_cache_specs may be modified in-place.
    global_kv_cache_groups = get_kv_cache_groups(vllm_config, merged_kv_cache_specs)

    # If original_max_model_len was -1, automatically
    # determine the maximum model length that fits in available GPU memory.
    # We use the global groups here to correctly account for padding.
    if vllm_config.model_config.original_max_model_len == -1:
        _auto_fit_max_model_len(vllm_config, global_kv_cache_groups, available_memory)

    # Check if the available memory is enough (using min across all workers).
    # We use the global groups to correctly account for padding.
    if global_kv_cache_groups:
        _check_enough_kv_cache_memory(
            min(available_memory),
            lambda: _max_memory_usage_bytes_from_groups(
                vllm_config, global_kv_cache_groups
            ),
            vllm_config.model_config.max_model_len,
            lambda am: _estimate_max_model_len_from_groups(
                vllm_config, global_kv_cache_groups, am
            ),
        )

    kv_cache_configs: list[KVCacheConfig] = []
    for kv_cache_spec_one_worker, available_memory_one_worker in zip(
        kv_cache_specs, available_memory
    ):
        kv_cache_groups_one_worker: list[KVCacheGroupSpec] = []
        for group in global_kv_cache_groups:
            group_layer_names_one_worker = [
                layer_name
                for layer_name in group.layer_names
                if layer_name in kv_cache_spec_one_worker
            ]
            kv_cache_groups_one_worker.append(
                KVCacheGroupSpec(group_layer_names_one_worker, group.kv_cache_spec)
            )
        assert sum(
            len(group.layer_names) for group in kv_cache_groups_one_worker
        ) == len(kv_cache_spec_one_worker), "Some layers are not assigned to any group."
        kv_cache_configs.append(
            get_kv_cache_config_from_groups(
                vllm_config, kv_cache_groups_one_worker, available_memory_one_worker
            )
        )

    # Change the num_blocks of each rank to the smallest among all ranks.
    # We also need to shrink the tensor size proportionally to avoid
    # allocating unused memory.
    min_num_blocks = min(
        kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
    )
    for kv_cache_config in kv_cache_configs:
        num_blocks_old = kv_cache_config.num_blocks
        kv_cache_config.num_blocks = min_num_blocks

        linear_layers = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            if isinstance(kv_cache_group.kv_cache_spec, MambaSpec):
                linear_layers.extend(kv_cache_group.layer_names)

        # Shrink tensor size proportionally
        for tensor in kv_cache_config.kv_cache_tensors:
            linear_tensor = False
            for layer in tensor.shared_by:
                if layer in linear_layers:
                    linear_tensor = True
                    break
            if linear_tensor:
                continue
            assert tensor.size % num_blocks_old == 0
            tensor.size = tensor.size // num_blocks_old * min_num_blocks

        if len(kv_cache_config.kv_cache_groups) > 0:
            _report_kv_cache_config(vllm_config, kv_cache_config)

    return kv_cache_configs


vllm.v1.core.kv_cache_utils.get_kv_cache_groups = get_kv_cache_groups
vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups = get_kv_cache_config_from_groups
vllm.v1.core.kv_cache_utils.get_kv_cache_configs = get_kv_cache_configs
