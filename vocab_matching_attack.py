import torch

from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM, Gemma2Config, LlamaConfig
from transformers.models.gemma2.modeling_gemma2 import apply_rotary_pos_emb, repeat_kv

# Model Import

device_map = "cuda"
# device_map = "cpu" # Uncomment if no gpu is available
model_dtype = torch.bfloat16
model_name_or_path = "google/gemma-2-2b-it"
# model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct" # Uncomment to use llama instead of gemma

TOKENIZER = AutoTokenizer.from_pretrained(model_name_or_path)
MODEL = AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=model_dtype, attn_implementation="eager"
).to(device_map)
MODEL.eval()

# Generate Hidden States and Next-Token Proposal

def gen_hidden_states(sentence, layers=[1]):
    """
    Helper function to compute and return the hidden states from the specified layers
    of the loaded model for a given input sentence.

    Args:
        sentence (str): The input sentence to be encoded and passed through the model.
        layers (list of int): A list of layer indices from which to extract hidden states.

    Returns:
        hidden_states (list of torch.Tensor): The hidden states from the specified layers.
    """
    token_ids = TOKENIZER.encode(sentence, return_tensors="pt").to(device_map)
    output = MODEL.forward(token_ids, output_hidden_states=True)
    hidden_states = [output.hidden_states[l] for l in layers]
    return hidden_states

def gen_next_proposal(token_ids):
    """
    Proposes an optimized ordering of vocabulary tokens for use in the vocabulary
    matching attack.

    Instead of searching through the vocabulary in a fixed or random order, this
    function returns token indices sorted by descending likelihood, as predicted
    by the model given the input `token_ids`. This significantly improves the
    efficiency of the vocabulary matching process.

    Note:
        The tokenizer used to generate `token_ids` and the tokenizer of the model
        being attacked must be the same for correct alignment.

    For further details, see Appendix A of our paper.

    Args:
        token_ids (torch.Tensor): Input token IDs (batch size 1) to condition the model.

    Returns:
        torch.Tensor: Token indices sorted by descending likelihood.
    """
    output = MODEL.forward(token_ids)
    logits = output.logits[0, -1]
    return torch.argsort(logits, descending=True).long()

# Forward Pass with Caching Across Batches

def self_attn_cache(
    config,
    rot_emb,
    q_proj,
    k_proj,
    v_proj,
    o_proj,
    hidden_states: torch.Tensor,
    past_k_values: torch.Tensor,
    past_v_values: torch.Tensor,
    device_map=device_map,
    model_dtype=torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replacement for Attention.forward() when we wish to find the possible outputs
    of a batch of tokens, given the previous tokens.

    This implements caching which is *different* from standard KV-caching, as it caches
    repeated computation across *batches*, not just within a batch. For example, given
    inputs like:
        [[2, 3, 4, 5],
         [2, 3, 4, 7],
         [2, 3, 4, 9]],
    standard KV-caching would recompute attention for the prefix [2, 3, 4] three times.
    This function avoids that by caching the shared prefix computation once.

    Note that this implementation is specific to Gemma2Attention and LlamaAttention. This function will
    need to be manually tailored if the attack was used on other models to match their
    attention mechanism forward pass. If you do not wish to implement this function for
    every model, set `use_cache` to False in `vocab_matching_attack`, at the expense of a slower attack.

    Input:
        - config: Model configuration.
        - rot_emb: Rotary embedding function.
        - q_proj, k_proj, v_proj, o_proj: Linear projection layers used in attention.
        - hidden_states (torch.Tensor): Hidden states for the batch of possible (N+1)th tokens, shape (B, D_init).
        - past_k_values (torch.Tensor): Key matrix from previous tokens, shape (H, N, D).
        - past_v_values (torch.Tensor): Value matrix from previous tokens, shape (H, N, D).

    Returns:
        - O (torch.Tensor): Output of shape (B, D_init).
        - K (torch.Tensor): Key vectors for the (N+1)st tokens, shape (B, H, D).
        - V (torch.Tensor): Value vectors for the (N+1)st tokens, shape (B, H, D).
    """
    B, _ = hidden_states.size()
    num_heads, N, head_dim = past_k_values.size()

    # Make sure data types of past values are correct.
    if past_k_values.dtype != model_dtype:
        past_k_values = past_k_values.to(dtype=model_dtype)
    if past_v_values.dtype != model_dtype:
        past_v_values = past_v_values.to(dtype=model_dtype)

    # Linear projection, rotary embedding, and KV head repetition. No speedups can be done here, as this is for the batch of possible (N+1)st tokens.
    Q = q_proj(hidden_states).view(B, num_heads, 1, head_dim)  # (B, H, 1, D)
    K = k_proj(hidden_states).view(
        B, config.num_key_value_heads, 1, head_dim
    )  # (B, H_KV, 1, D)
    V = v_proj(hidden_states).view(
        B, config.num_key_value_heads, 1, head_dim
    )  # (B, H_KV, 1, D)
    token_pos = (
        torch.arange(N, N + 1).unsqueeze(0).to(device_map)
    )  # Don't cast this, this should always be int64.
    cos, sin = rot_emb(
        V, token_pos
    )  # Note: we only need a position_id for the (N+1)st token here.
    Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
    K = repeat_kv(K, num_heads // config.num_key_value_heads)  # (B, H, 1, D)
    V = repeat_kv(V, num_heads // config.num_key_value_heads)  # (B, H, 1, D)

    # We use broadcasted matmult when applicable to speedup attention score computation, and only compute the last row.
    A_past = torch.matmul(
        Q, past_k_values.transpose(1, 2)
    )  # (B, H, 1, D) @ (H, D, N) = (B, H, 1, N)
    A_self = (Q * K).sum(dim=-1, keepdim=True)  # (B, H, 1, 1)
    A = torch.cat([A_past, A_self], dim=-1)  # (B, H, 1, N+1)

    # Attention score processing. We ignore the attention mask since the new token depends on all tokens so far.
    if isinstance(config, Gemma2Config):
        A = A * (config.query_pre_attn_scalar ** -0.5)
        if config.attn_logit_softcapping is not None:
            A = (
                torch.tanh(A / config.attn_logit_softcapping)
                * config.attn_logit_softcapping
            )
    elif isinstance(config, LlamaConfig):
        A = A * (head_dim ** -0.5)
    else:
        raise Exception("Caching for this model architecture is not yet supported. Please set use_cache to False.")
    
    A = softmax(A, dim=-1, dtype=torch.float32).to(
        Q.dtype
    )  # Always upcast to float32 before back to original dtype, regardless of model_dtype.

    # We use broadcasted matmult when applicable to speedup attention-value multiplication, and only compute the last row.
    O_past = torch.matmul(
        A[..., :-1], past_v_values
    )  # (B, H, 1, N) @ (H, N, D) = (B, H, 1, D)
    O_self = A[..., -1:] * V  # (B, H, 1, 1) @ (B, H, 1, D) = (B, H, 1, D)
    O = O_past + O_self  # (B, H, 1, D)

    # Final linear projection (remove heads).
    O = O.transpose(1, 2).contiguous()  # (B, 1, H, D)
    O = O.reshape(B, -1).contiguous()  # (B, 1, H * D)
    O = o_proj(O)  # (B, 1, D_init)
    return O, K[..., 0, :], V[..., 0, :]

def forward_pass_cache(
    token_ids: torch.Tensor,
    all_past_k_values: list[torch.Tensor],
    all_past_v_values: list[torch.Tensor],
    num_layers: int,
    device_map=device_map,
    model_dtype: torch.dtype = model_dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replacement for AutoModelForCausalLM.forward() when we wish to find the possible outputs
    of a batch of tokens, given the previous tokens.

    This implements caching which is *different* from standard KV-caching, as it caches
    repeated computation *across batches*, not just within a batch.

    Note that, like `self_attn_cache`, this function needs to be tailored to the attacked model.
    Otherwise, set `use_cache` to False in `vocab_matching_attack`.

    Input:
        - all_past_k_values (list[torch.Tensor]): List of length `num_layers`, each of shape (H, N, D),
          containing key matrices from previous tokens.
        - all_past_v_values (list[torch.Tensor]): List of length `num_layers`, each of shape (H, N, D),
          containing value matrices from previous tokens.
        - token_ids (torch.Tensor): Tensor of token IDs for the (N+1)st token, of shape (B,).
        - num_layers (int): Number of decoder layers to forward through. If equal to total number of layers,
          final normalization is applied.

    Returns:
        - hidden_states (torch.Tensor): Final hidden state outputs at specified layer, of shape (B, D_init).
        - all_K (torch.Tensor): Cached key matrices at each layer, shape (num_layers, B, H, D).
        - all_V (torch.Tensor): Cached value matrices at each layer, shape (num_layers, B, H, D).
    """
    config = MODEL.config
    model = MODEL.model
    assert num_layers <= config.num_hidden_layers

    # Embedding and pre-scaling.
    hidden_states = model.embed_tokens(token_ids)
    if isinstance(config, Gemma2Config):
        hidden_states = hidden_states * torch.tensor(
            config.hidden_size**0.5, dtype=hidden_states.dtype
        )

    # These will store cached key and value matrices of sizes (B, H, D) for each layer. B is the length of token_ids.
    all_K, all_V = [], []

    # Decoder blocks with caching.
    for layer_idx in range(num_layers):
        layer = model.layers[layer_idx]

        # Extract public weights needed for cached attention.
        q_proj = layer.self_attn.q_proj
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj
        o_proj = layer.self_attn.o_proj
        rot_emb = layer.self_attn.rotary_emb

        # Past key and value matrices of shape (num_heads, num_tokens, head_dim) that the adversary may have deciphered thus far.
        past_k_values = all_past_k_values[layer_idx]
        past_v_values = all_past_v_values[layer_idx]

        # Attention block. We remember to cache the K and V to return later.
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        hidden_states, K, V = self_attn_cache(
            config,
            rot_emb,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            hidden_states,
            past_k_values,
            past_v_values,
            device_map=device_map,
            model_dtype=model_dtype,
        )
        if isinstance(config, Gemma2Config):
            hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        all_K.append(K)
        all_V.append(V)

        # MLP block.
        residual = hidden_states
        if isinstance(config, Gemma2Config):
            hidden_states = layer.pre_feedforward_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)    
            hidden_states = layer.post_feedforward_layernorm(hidden_states)
        elif isinstance(config, LlamaConfig):
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

    # Final normalization for the last layer.
    if num_layers >= config.num_hidden_layers:
        hidden_states = model.norm(hidden_states)

    all_K, all_V = torch.stack(all_K), torch.stack(all_V)
    return hidden_states, all_K, all_V

# Vocab Matching Attack

def permute_states(hidden_states: torch.Tensor, perm_type: str) -> torch.Tensor:
    """
    Applies a permutation to the hidden states tensor based on the specified permutation type.

    Possible permutation types are:
        * 'None' -- No permutation applied.
        * 'S'    -- Sequence-level permutation (shuffle rows).
        * 'D'    -- Per-token hidden dimension permutation (shuffle columns).
        * 'SD'   -- Combined sequence and hidden dimension permutation (apply 'D' then 'S').

    Args:
        hidden_states (torch.Tensor): Input tensor of shape (N, d), where N is the number of tokens
                                      and d is the hidden dimension.
        perm_type (str): Type of permutation to apply.

    Returns:
        torch.Tensor: Permuted tensor of shape (N, d).

    Raises:
        Exception: If the permutation type is not supported.
    """
    N, d = hidden_states.size()
    device = hidden_states.device
    if perm_type == "None":
        return hidden_states
    elif perm_type == "S":
        return hidden_states[torch.randperm(N, device=device)]
    elif perm_type == "D":
        return hidden_states[:, torch.randperm(d, device=device)]
    elif perm_type == "SD":
        return permute_states(permute_states(hidden_states, "D"), "S")
    else:
        raise Exception(f"Unsupported permutation pattern {perm_type}")

# Different Matching Functions

def argmin_except(tensor: torch.Tensor, ignore_indices: list[int]) -> tuple[int, int]:
    """
    Returns the 2D index of the minimum value in a tensor, excluding specified column indices.

    This function finds the global minimum across all elements in a 2D tensor,
    except for positions in columns listed in `ignore_indices`, which are masked out.

    Args:
        tensor (torch.Tensor): A 2D tensor of shape (B, N) to search for the minimum value.
        ignore_indices (list[int]): List of column indices to ignore in the search.

    Returns:
        tuple[int, int]: The (batch_index, sequence_index) corresponding to the minimum value.
    """

    # Create a mask of valid positions (False for ignored indices)
    mask = torch.ones(tensor.shape[1], dtype=torch.bool, device=tensor.device)
    mask[ignore_indices] = False

    # Expand mask to match tensor dimensions
    mask = mask.expand(tensor.shape)

    # Set ignored positions to maximum value
    masked_tensor = torch.where(mask, tensor, torch.finfo(tensor.dtype).max)

    # Get global argmin across both dimensions
    flat_idx = masked_tensor.view(-1).argmin()

    # Convert flat index back to 2D indices
    batch_idx = flat_idx // tensor.shape[1]
    seq_idx = flat_idx % tensor.shape[1]

    return batch_idx.item(), seq_idx.item()

def l1_dist_matching(
    perm_hidden_states: torch.Tensor,
    vocab_hidden_states: torch.Tensor,
    index: int,
    ignore_perm_idx: list[int] = [],
) -> tuple[tuple[int, int], float]:
    """
    Matches by closest L1 distance between a row vector from `perm_hidden_states` and all row vectors in `vocab_hidden_states`.

    This function works for `None` and `S` permutations in `perm_hidden_states`. You can specify indices to ignore from
    matching using `ignore_perm_idx`.

    Args:
        perm_hidden_states (torch.Tensor): Tensor of permuted hidden states of shape (N, D).
        vocab_hidden_states (torch.Tensor): Tensor of vocab hidden states of shape (V, 1, D).
        index (int): Index of the row in perm_hidden_states to match.
        ignore_perm_idx (list[int], optional): Indices in perm_hidden_states to ignore. Defaults to [].

    Returns:
        tuple[tuple[int, int], float]: Tuple of best-matching indices (perm_index, vocab_index) and the L1 distance.
    """
    perm_hidden_states = perm_hidden_states[index, :]
    intermediate = torch.sum(
        torch.abs(perm_hidden_states - vocab_hidden_states[:, -1:, :]), dim=-1
    )
    vocab_argmin = torch.argmin(intermediate)
    min_l1_dist = intermediate[vocab_argmin].item()
    return (index, vocab_argmin), min_l1_dist


def l1_dist_matching_permuted(
    perm_hidden_states: torch.Tensor,
    vocab_hidden_states: torch.Tensor,
    index: int | None = None,
    ignore_perm_idx: list[int] = [],
) -> tuple[tuple[int, int], float]:
    """
    Matches by closest L1 distance between all pairs of row vectors between `perm_hidden_states` and `vocab_hidden_states`.

    This function handles arbitrary permutations and returns the best match among all combinations,
    excluding any permuted indices listed in `ignore_perm_idx`.

    Args:
        perm_hidden_states (torch.Tensor): Tensor of permuted hidden states of shape (N, D).
        vocab_hidden_states (torch.Tensor): Tensor of vocab hidden states of shape (V, 1, D).
        index (int | None, optional): Unused placeholder for compatibility. Defaults to None.
        ignore_perm_idx (list[int], optional): Indices in perm_hidden_states to ignore. Defaults to [].

    Returns:
        tuple[tuple[int, int], float]: Tuple of best-matching indices (perm_index, vocab_index) and the L1 distance.
    """
    intermediate = torch.sum(
        torch.abs(perm_hidden_states.unsqueeze(0) - vocab_hidden_states[:, -1:, :]),
        dim=-1,
    )
    vocab_argmin, seq_argmin = argmin_except(
        intermediate, ignore_indices=ignore_perm_idx
    )
    min_l1_dist = intermediate[vocab_argmin, seq_argmin].item()
    return (seq_argmin, vocab_argmin), min_l1_dist


def l1_sort_dist_matching(
    perm_hidden_states: torch.Tensor,
    vocab_hidden_states: torch.Tensor,
    index: int | None = None,
    ignore_perm_idx: list[int] = [],
) -> tuple[tuple[int, int], float]:
    """
    Matches by closest L1 distance between sorted row vectors of 2D tensors.

    This function supports all permutation types (`None`, `S`, `D`, `SD`) by sorting the vectors before matching.
    It calls `l1_dist_matching` on the sorted tensors.

    Args:
        perm_hidden_states (torch.Tensor): Tensor of permuted hidden states of shape (N, D).
        vocab_hidden_states (torch.Tensor): Tensor of vocab hidden states of shape (V, 1, D).
        index (int | None, optional): Row index to match. Defaults to None.
        ignore_perm_idx (list[int], optional): Indices in perm_hidden_states to ignore. Defaults to [].

    Returns:
        tuple[tuple[int, int], float]: Tuple of best-matching indices (perm_index, vocab_index) and the L1 distance.
    """
    sorted_phs, _ = torch.sort(perm_hidden_states, dim=-1)
    sorted_vhs, _ = torch.sort(vocab_hidden_states, dim=-1)
    return l1_dist_matching(
        sorted_phs, sorted_vhs, index, ignore_perm_idx=ignore_perm_idx
    )


def l1_sort_dist_matching_permuted(
    perm_hidden_states: torch.Tensor,
    vocab_hidden_states: torch.Tensor,
    index: int | None = None,
    ignore_perm_idx: list[int] = [],
) -> tuple[tuple[int, int], float]:
    """
    Matches by closest L1 distance between all pairs of sorted row vectors.

    This handles all permutation types by sorting the vectors in both tensors first, then matching
    all pairs using `l1_dist_matching_permuted`.

    Args:
        perm_hidden_states (torch.Tensor): Tensor of permuted hidden states of shape (N, D).
        vocab_hidden_states (torch.Tensor): Tensor of vocab hidden states of shape (V, 1, D).
        index (int | None, optional): Unused placeholder for compatibility. Defaults to None.
        ignore_perm_idx (list[int], optional): Indices in perm_hidden_states to ignore. Defaults to [].

    Returns:
        tuple[tuple[int, int], float]: Tuple of best-matching indices (perm_index, vocab_index) and the L1 distance.
    """
    sorted_phs, _ = torch.sort(perm_hidden_states, dim=-1)
    sorted_vhs, _ = torch.sort(vocab_hidden_states, dim=-1)
    return l1_dist_matching_permuted(
        sorted_phs, sorted_vhs, ignore_perm_idx=ignore_perm_idx
    )

# Full Vocab Attack

def vocab_matching_attack(
    perm_hidden_states: torch.Tensor,
    num_layers: int,
    matching_ftn: callable,
    perm_type: str,
    batch_sz: int = 1000,
    matching_eps: float = 1e-3,
    next_token_proposal: bool = False,
    use_cache: bool = False,
) -> list[int]:
    """
    Performs a full vocabulary matching attack to reconstruct tokens from hidden states using distance-based matching.

    This generalized attack supports various permutation types and corresponding matching functions (`l1_dist_matching`,
    `l1_dist_matching_permuted`, `l1_sort_dist_matching`, `l1_sort_dist_matching_permuted`). It optionally supports
    next-token proposals using the model's logits and can use key/value caching for efficient forward passes.

    Args:
        perm_hidden_states (torch.Tensor): Tensor of permuted hidden states of shape (N, D).
        num_layers (int): Number of layers to extract hidden states from.
        matching_ftn (callable): Matching function to use, must be compatible with the given permutation type.
        perm_type (str): The permutation type, one of 'None', 'S', 'D', 'SD'.
        batch_sz (int, optional): Batch size for forward passes. Defaults to 1000.
        matching_eps (float, optional): Early stopping threshold for matching error. Defaults to 1e-3.
        next_token_proposal (bool, optional): Whether to use LLM to suggest next token candidates. Defaults to False.
        use_cache (bool, optional): Whether to use key/value caching to speed up matching. Defaults to False.

    Returns:
        list[int]: A list of decoded token IDs corresponding to the best matches found.
    """

    if use_cache:
        MODEL.config.use_cache = True
    else:
        MODEL.config.use_cache = False

    # Check matching attack/function is valid with permutation type.
    if matching_ftn == l1_dist_matching:
        assert perm_type in ["None"]
    elif matching_ftn == l1_dist_matching_permuted:
        assert perm_type in ["None", "S"]
    elif matching_ftn == l1_sort_dist_matching:
        assert perm_type in ["None", "D"]
    elif matching_ftn == l1_sort_dist_matching_permuted:
        assert perm_type in ["None", "S", "D", "SD"]
    else:
        raise Exception("Unsupported matching type currently")

    # Get embedding weights.
    config = MODEL.config
    vocab_sz = config.vocab_size

    # Initialize cached key and value matrices. See `forward_pass_cache` for explanation of the shape.
    if use_cache:
        all_past_k_values = torch.empty(
            num_layers, config.num_attention_heads, 0, config.head_dim
        ).to(device_map)
        all_past_v_values = torch.empty(
            num_layers, config.num_attention_heads, 0, config.head_dim
        ).to(device_map)

    # Vocab attack deciphers token one at a time.
    ignore_perm_idx = []
    input_tokens = []
    num_tokens = perm_hidden_states.numel() // config.hidden_size
    for i in range(num_tokens):
        global_best_error = 100000
        global_best_token = None
        # Based on our token proposal method, we either iterate through tokens in order of ID, or use the LLM to predict high-prob next tokens.
        if not next_token_proposal or i == 0:
            token_ids = torch.arange(0, vocab_sz, device=device_map).long()
        else:
            token_ids = gen_next_proposal(
                torch.LongTensor(input_tokens).unsqueeze(0).to(device_map)
            )

        # We batch tokens due to memory and time constraints.
        for batch_start in range(0, vocab_sz, batch_sz):
            # If caching, use `forward_pass_cache` to get the batched output for the next token.
            if use_cache:
                batch_ids = token_ids[
                    batch_start : min(batch_start + batch_sz, vocab_sz)
                ]
                batch_hidden_states, all_K, all_V = forward_pass_cache(
                    batch_ids, all_past_k_values, all_past_v_values, num_layers
                )
                batch_hidden_states = batch_hidden_states.reshape(batch_sz, 1, -1)

            # Otherwise, directly run the model's forward pass on all batches for the next token, concatenating with previous known tokens.
            else:
                batch_ids = token_ids[
                    batch_start : min(batch_start + batch_sz, vocab_sz)
                ].reshape(-1, 1)
                batch_input_tokens = (
                    torch.tensor(input_tokens)
                    .to(device_map)
                    .reshape(1, -1)
                    .repeat(batch_sz, 1)
                )
                batch_ids = torch.cat([batch_input_tokens, batch_ids], dim=-1).long()
                outputs = MODEL.forward(batch_ids, output_hidden_states=True)
                batch_hidden_states = outputs.hidden_states[num_layers]

            best_pair, best_err = matching_ftn(
                perm_hidden_states,
                batch_hidden_states,
                index=i,
                ignore_perm_idx=ignore_perm_idx,
            )

            if global_best_error > best_err:
                global_best_error = best_err
                global_best_token = token_ids[batch_start + best_pair[1]].item()
                global_ignored_idx = best_pair[0]

                if use_cache:
                    global_best_K = all_K[:, best_pair[1], :, None]
                    global_best_V = all_V[:, best_pair[1], :, None]

            if batch_start + batch_sz >= vocab_sz and global_best_error > matching_eps:
                print(f"No match for token {i} under eps")
                print(f"Best error: {global_best_error} for token {global_best_token}")

            # If we get below the epsilon matching error, we choose this token and move on to the next token reversal.
            # If the full vocabulary is exhausted, choose the token with lowest global error
            if (i < num_tokens and best_err < matching_eps) or batch_start + batch_sz >= vocab_sz:
                chosen_token = global_best_token
                ignore_perm_idx.append(global_ignored_idx)

                # Add the token to our list of known tokens before stopping batched run.
                input_tokens.append(chosen_token)

                # If caching, we must update the known key and value matrices.
                if use_cache:
                    all_past_k_values = torch.cat(
                        [all_past_k_values, global_best_K], dim=-2
                    )
                    all_past_v_values = torch.cat(
                        [all_past_v_values, global_best_V], dim=-2
                    )
                break

    return input_tokens

def run_vocab_matching_attack(
    sentence: str,
    layer: int,
    dist_funct: str = "l1",
    perm_type: str = "None",
    batch_sz: int = 100,
    matching_eps: float = 1e-3,
    next_token_proposal: bool = False,
    use_cache: bool = False,
) -> tuple[list[int], list[int]]:
    """
    Utility function that performs the full workflow of the vocabulary matching attack.

    This includes tokenizing the input sentence, generating hidden states, applying a specified permutation type,
    selecting a matching function, and invoking the main `vocab_matching_attack`. It optionally supports next-token
    proposal and key/value caching to speed up matching.

    Args:
    sentence (str): Input prompt to be attacked.
    layer (int): Transformer layer from which to extract hidden states.
    dist_funct (str, optional): Matching function to use: one of 'l1', 'l1_permuted', 'l1_sort', or 'l1_sort_permuted'. Defaults to 'l1'.
    perm_type (str, optional): Type of permutation to apply on the hidden states. One of 'None', 'S', 'D', or 'SD'. Defaults to 'None'.
    batch_sz (int, optional): Number of candidate tokens evaluated per batch during matching. Defaults to 100.
    matching_eps (float, optional): Threshold for early stopping based on matching error. Defaults to 1.0.
    next_token_proposal (bool, optional): Whether to use model logits to prioritize candidate tokens. Defaults to False.
    use_cache (bool, optional): Whether to use key/value caching to accelerate forward passes. Defaults to False.

    Returns:
    tuple[list[int], list[int]]: A tuple containing the original token IDs and the decoded token IDs predicted by the attack.
    """

    hidden_states = gen_hidden_states(sentence, layers=[layer])[0][0]
    perm_hidden_states = permute_states(hidden_states, perm_type)

    if dist_funct == "l1":
        dist_funct = l1_dist_matching
    elif dist_funct == "l1_permuted":
        dist_funct = l1_dist_matching_permuted
    elif dist_funct == "l1_sort":
        dist_funct = l1_sort_dist_matching
    elif dist_funct == "l1_sort_permuted":
        dist_funct = l1_sort_dist_matching_permuted
    else:
        raise ValueError(f"Unknown dist_funct {dist_funct}")

    decoded = vocab_matching_attack(
        perm_hidden_states,
        layer,
        dist_funct,
        perm_type,
        batch_sz=batch_sz,
        matching_eps=matching_eps,
        next_token_proposal=next_token_proposal,
        use_cache=use_cache,
    )
    return decoded

# Testing

def truncate_prompt(text: str, num_tokens: int) -> str:
    """
    Helper function that encodes a prompt, truncates it to a specified number of tokens,
    and decodes it back into text without adding special tokens.

    This is useful for creating a prompt with a maximum token length constraint.

    Args:
        text (str): The input prompt string.
        num_tokens (int): The maximum number of tokens to retain.

    Returns:
        str: The truncated prompt string, decoded from the first `num_tokens` tokens.
    """
    token_ids = TOKENIZER.encode(text, add_special_tokens=False)[:num_tokens]
    return TOKENIZER.decode(token_ids, skip_special_tokens=True)


if __name__ == "__main__":
    # Example prompts to test the vocabulary matching attack on.
    PROMPTS = [
        "Write a short story in which a robot discovers an ancient human diary. The robot interprets the diary as a set of instructions to rebuild humanity but encounters unexpected challenges. Describe its thought process and how it resolves its dilemmas.",
        "If the sequence \(a_n\) is defined recursively by \(a_0 = 2\) and \(a_{n+1} = 3a_n + 4\), find a closed-form expression for \(a_n\). Explain your reasoning and solve for \(a_5\).",
        "Explain the key factors that led to the fall of the Western Roman Empire. Focus on economic, military, and political causes, and discuss whether any single cause was decisive or if it was a combination of factors.",
    ]

    # Attack parameters
    layer = 1
    dist_funct = "l1_sort_permuted"
    perm_type = "SD"
    batch_sz = 256
    matching_eps = 22
    next_token_proposal = True
    use_cache = True

    prompts = [truncate_prompt(prompt, 50) for prompt in PROMPTS]
    for prompt in prompts:
        decoded_tokens = run_vocab_matching_attack(
            prompt,
            layer,
            dist_funct,
            perm_type,
            batch_sz,
            matching_eps,
            next_token_proposal,
            use_cache,
        )
        obtained_prediction = TOKENIZER.decode(decoded_tokens, skip_special_tokens=True)
        print(obtained_prediction)
        print("Results match!" if prompt == obtained_prediction else "Results don't match.")
        print()

