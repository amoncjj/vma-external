import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import argparse

# Set random seeds for reproducibility
RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    """设置所有随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Model configurations
MODEL_CONFIGS = {
    "llama3.2-1B": "/home/junjie_chen/models/llama3.2-1B",
    "llama3-8B": "/home/junjie_chen/models/llama3-8B",
    # "llama-7B": "/home/junjie_chen/models/llama-7B",  # 暂时移除：torch版本兼容性问题
    "qwen3-8B": "/home/junjie_chen/models/qwen3-8B",
}

def gen_kv_states(model, tokenizer, sentence, layers=[1], device_map="cuda"):
    """
    生成指定层的K和V states
    
    Returns:
        tuple: (k_states_list, v_states_list)
    """
    token_ids = tokenizer.encode(sentence, return_tensors="pt", add_special_tokens=False).to(device_map)
    
    with torch.no_grad():
        outputs = model(token_ids, use_cache=True, output_hidden_states=True)
    
    k_states_list = []
    v_states_list = []
    
    for layer_idx in layers:
        k_cache = outputs.past_key_values[layer_idx][0]
        v_cache = outputs.past_key_values[layer_idx][1]
        
        batch_size, num_heads, seq_len, head_dim = k_cache.shape
        k_states = k_cache.squeeze(0).transpose(0, 1).reshape(seq_len, num_heads * head_dim)
        v_states = v_cache.squeeze(0).transpose(0, 1).reshape(seq_len, num_heads * head_dim)
        
        k_states_list.append(k_states)
        v_states_list.append(v_states)
    
    return k_states_list, v_states_list

def gen_next_proposal(model, token_ids):
    """使用模型预测下一个token的概率排序"""
    with torch.no_grad():
        output = model(token_ids)
    logits = output.logits[0, -1]
    return torch.argsort(logits, descending=True).long()

def permute_states(states: torch.Tensor, perm_type: str) -> torch.Tensor:
    """对states进行置换"""
    N, d = states.size()
    device = states.device
    if perm_type == "None":
        return states
    elif perm_type == "S":
        return states[torch.randperm(N, device=device)]
    elif perm_type == "D":
        return states[:, torch.randperm(d, device=device)]
    elif perm_type == "SD":
        return permute_states(permute_states(states, "D"), "S")
    else:
        raise Exception(f"Unsupported permutation pattern {perm_type}")

def kv_matching_attack(
    model,
    tokenizer,
    perm_k_states: torch.Tensor,
    perm_v_states: torch.Tensor,
    layer: int,
    batch_sz: int = 128,
    matching_eps: float = 1.0,
    next_token_proposal: bool = True,
    max_proposal_candidates: int = 5000,
    device_map: str = "cuda",
    ground_truth_tokens: list[int] = None,
) -> list[int]:
    """
    使用KV cache执行vocabulary matching attack
    
    Args:
        model: 模型
        tokenizer: tokenizer
        perm_k_states: Permuted K states (num_tokens, k_dim)
        perm_v_states: Permuted V states (num_tokens, v_dim)
        layer: 要攻击的层
        batch_sz: 批次大小
        matching_eps: 匹配阈值
        next_token_proposal: 是否使用next token proposal
        max_proposal_candidates: 最大候选数
        device_map: 设备
    
    Returns:
        解码出的token列表
    """
    vocab_sz = model.config.vocab_size
    num_tokens = perm_k_states.shape[0]
    
    input_tokens = []
    
    for i in range(num_tokens):
        global_best_error = float('inf')
        global_best_token = None
        
        # 根据是否使用next token proposal决定搜索策略
        if not next_token_proposal or i == 0:
            token_ids = torch.arange(0, vocab_sz, device=device_map).long()
            max_search_tokens = vocab_sz
        else:
            token_ids = gen_next_proposal(
                model,
                torch.LongTensor(input_tokens).unsqueeze(0).to(device_map)
            )
            max_search_tokens = min(max_proposal_candidates, vocab_sz) if max_proposal_candidates > 0 else vocab_sz
        
        # 批量处理tokens
        for batch_start in range(0, max_search_tokens, batch_sz):
            batch_end = min(batch_start + batch_sz, max_search_tokens)
            actual_batch_sz = batch_end - batch_start
            
            # 构建batch输入
            batch_ids = token_ids[batch_start:batch_end].reshape(-1, 1)
            
            if i > 0:
                batch_input_tokens = (
                    torch.tensor(input_tokens)
                    .to(device_map)
                    .reshape(1, -1)
                    .repeat(actual_batch_sz, 1)
                )
                batch_ids = torch.cat([batch_input_tokens, batch_ids], dim=-1).long()
            
            # Forward pass获取KV cache
            with torch.no_grad():
                outputs = model(batch_ids, use_cache=True, output_hidden_states=True)
            
            # 提取K和V
            k_cache = outputs.past_key_values[layer][0]  # (batch, num_heads, seq_len, head_dim)
            v_cache = outputs.past_key_values[layer][1]
            
            # 只取最后一个token
            batch_size, num_heads, seq_len, head_dim = k_cache.shape
            batch_k = k_cache[:, :, -1, :].reshape(batch_size, num_heads * head_dim)
            batch_v = v_cache[:, :, -1, :].reshape(batch_size, num_heads * head_dim)
            
            # 计算K和V的L1距离（使用排序）
            perm_k_row = perm_k_states[i, :]
            perm_v_row = perm_v_states[i, :]
            
            sorted_perm_k, _ = torch.sort(perm_k_row)
            sorted_perm_v, _ = torch.sort(perm_v_row)
            
            # 计算每个候选的K和V error
            batch_best_error = float('inf')
            batch_best_token = None
            
            for j in range(actual_batch_sz):
                sorted_k, _ = torch.sort(batch_k[j])
                sorted_v, _ = torch.sort(batch_v[j])
                
                k_error = torch.sum(torch.abs(sorted_perm_k - sorted_k)).item()
                v_error = torch.sum(torch.abs(sorted_perm_v - sorted_v)).item()
                total_error = k_error + v_error
                
                if total_error < global_best_error:
                    global_best_error = total_error
                    global_best_token = token_ids[batch_start + j].item()
                
                if total_error < batch_best_error:
                    batch_best_error = total_error
                    batch_best_token = token_ids[batch_start + j].item()
            
            # 清理GPU内存
            del outputs
            torch.cuda.empty_cache()
            
            # 如果找到低于matching_eps的token，立即停止（与原版逻辑一致）
            if batch_best_error < matching_eps:
                global_best_error = batch_best_error
                global_best_token = batch_best_token
                break
            
            # 如果这是最后一个batch且还没找到低于eps的，打印警告
            if batch_end >= max_search_tokens and global_best_error > matching_eps:
                print(f"⚠ No match for token {i} under eps={matching_eps:.4f}")
                print(f"   Best error: {global_best_error:.4f} for token {global_best_token} ('{tokenizer.decode([global_best_token])}')")
        
        # 记录找到的最佳token
        input_tokens.append(global_best_token)
        
        # 打印进度
        status = "✓" if global_best_error < matching_eps else "⚠"
        print(f"{status} Token {i}: {global_best_token} ('{tokenizer.decode([global_best_token])}'), "
              f"error={global_best_error:.4f}, eps={matching_eps:.4f}")
    
    return input_tokens

def run_kv_attack(
    model,
    tokenizer,
    sentence: str,
    layer: int,
    perm_type: str = "D",
    batch_sz: int = 128,
    matching_eps: float = 1.0,
    next_token_proposal: bool = True,
    max_proposal_candidates: int = 5000,
    device_map: str = "cuda",
) -> tuple[list[int], list[int]]:
    """
    执行完整的KV cache攻击流程
    
    Returns:
        (ground_truth_tokens, decoded_tokens)
    """
    # 获取ground truth
    ground_truth_tokens = tokenizer.encode(sentence, add_special_tokens=False)
    
    # 生成KV states
    print(f"🔑 Attacking KV Cache at layer {layer}")
    k_states_list, v_states_list = gen_kv_states(model, tokenizer, sentence, layers=[layer], device_map=device_map)
    k_states = k_states_list[0]
    v_states = v_states_list[0]
    
    # 应用置换
    perm_k_states = permute_states(k_states, perm_type)
    perm_v_states = permute_states(v_states, perm_type)
    
    # 执行攻击
    decoded_tokens = kv_matching_attack(
        model,
        tokenizer,
        perm_k_states,
        perm_v_states,
        layer,
        batch_sz=batch_sz,
        matching_eps=matching_eps,
        next_token_proposal=next_token_proposal,
        max_proposal_candidates=max_proposal_candidates,
        device_map=device_map,
    )
    
    return ground_truth_tokens, decoded_tokens

def truncate_prompt(tokenizer, text: str, num_tokens: int) -> str:
    """截断prompt到指定的token数量"""
    token_ids = tokenizer.encode(text, add_special_tokens=False)[:num_tokens]
    return tokenizer.decode(token_ids, skip_special_tokens=True)

def load_lmsys_samples(tokenizer, num_samples=100, max_tokens=50, seed=RANDOM_SEED):
    """从lmsys数据集加载样本"""
    data_file = "/home/junjie_chen/datasets/lmsys-chat-1m-data-1000/data.json"
    print(f"Loading samples from {data_file} (seed={seed})...")
    
    with open(data_file, 'r') as f:
        all_data = json.load(f)
    
    all_valid_samples = []
    for text in all_data:
        text = text.strip()
        if len(text) > 20:
            truncated = truncate_prompt(tokenizer, text, max_tokens)
            if len(truncated) > 10:
                all_valid_samples.append(truncated)
    
    print(f"Found {len(all_valid_samples)} valid samples")
    
    if len(all_valid_samples) > num_samples:
        random.seed(seed)
        samples = random.sample(all_valid_samples, num_samples)
        print(f"Randomly sampled {num_samples} samples (seed={seed})")
    else:
        samples = all_valid_samples[:num_samples]
        print(f"Using first {len(samples)} samples")
    
    return samples

def load_config(config_file: str, model_name: str, layer: int) -> float:
    """从配置文件加载matching_eps"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        if model_name in config:
            layer_str = str(layer)
            if layer_str in config[model_name]['layers']:
                return config[model_name]['layers'][layer_str]['matching_eps']
        
        print(f"⚠️  Warning: No config found for {model_name} layer {layer}, using default eps=1.0")
        return 1.0
    
    except FileNotFoundError:
        print(f"⚠️  Warning: Config file {config_file} not found, using default eps=1.0")
        return 1.0

def main():
    parser = argparse.ArgumentParser(description="KV Cache 攻击")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="要攻击的模型")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="要攻击的层列表（默认：第一层、中间层、最后一层）")
    parser.add_argument("--perm_type", type=str, default="None",
                        choices=["None", "D"],
                        help="置换类型: None=无置换, D=维度置换")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径（默认根据perm_type自动选择）")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="测试样本数")
    parser.add_argument("--max_tokens", type=int, default=50,
                        help="每个样本的最大token数")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="批次大小")
    parser.add_argument("--max_proposal_candidates", type=int, default=5000,
                        help="最大候选数")
    parser.add_argument("--output", type=str, default=None,
                        help="输出结果文件")
    
    args = parser.parse_args()
    
    # 根据perm_type自动选择配置文件
    if args.config is None:
        if args.perm_type == "None":
            args.config = "kv_attack_config_no_perm.json"
        elif args.perm_type == "D":
            args.config = "kv_attack_config_with_perm.json"
        else:
            args.config = "kv_attack_config.json"
    
    # 设置随机种子
    set_seed(RANDOM_SEED)
    print(f"Random seed set to: {RANDOM_SEED}")
    
    # 加载模型
    model_path = MODEL_CONFIGS[args.model]
    device_map = "cuda"
    model_dtype = torch.bfloat16
    
    print(f"\n{'='*80}")
    print(f"Loading model: {args.model}")
    print(f"Model path: {model_path}")
    print(f"{'='*80}\n")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=model_dtype, attn_implementation="eager"
    ).to(device_map)
    model.eval()
    
    num_hidden_layers = model.config.num_hidden_layers
    print(f"Model loaded. Number of layers: {num_hidden_layers}")
    
    # 确定要攻击的层
    # 注意：层索引从0开始，所以最后一层是 num_hidden_layers - 1
    if args.layers is None:
        first_layer = 0  # 使用第0层（第一层）
        middle_layer = num_hidden_layers // 2
        last_layer = num_hidden_layers - 1  # 最后一层的索引
        attack_layers = [first_layer, middle_layer, last_layer]
    else:
        attack_layers = args.layers
    
    print(f"Attack layers: {attack_layers}")
    print(f"  - First layer: {attack_layers[0] if len(attack_layers) > 0 else 'N/A'} (layer 0)")
    print(f"  - Middle layer: {attack_layers[1] if len(attack_layers) > 1 else 'N/A'}")
    print(f"  - Last layer: {attack_layers[2] if len(attack_layers) > 2 else 'N/A'}")
    print(f"Permutation type: {args.perm_type}")
    print(f"Config file: {args.config}")
    
    # 加载样本
    test_samples = load_lmsys_samples(tokenizer, num_samples=args.num_samples, 
                                      max_tokens=args.max_tokens, seed=RANDOM_SEED)
    
    print(f"\nLoaded {len(test_samples)} samples\n")
    
    # 设置输出文件
    if args.output is None:
        perm_suffix = "no_perm" if args.perm_type == "None" else "with_perm"
        args.output = f"kv_attack_results_{args.model}_{perm_suffix}.json"
    
    # 运行攻击
    results = {
        'model_name': args.model,
        'perm_type': args.perm_type,
        'num_samples': len(test_samples),
        'attack_layers': attack_layers,
        'layers': []
    }
    perm_type = args.perm_type
    next_token_proposal = True
    
    for layer in attack_layers:
        print(f"\n{'='*80}")
        print(f"Attacking Layer {layer}")
        print(f"{'='*80}\n")
        
        # 从配置加载matching_eps
        matching_eps = load_config(args.config, args.model, layer)
        print(f"Using matching_eps: {matching_eps:.6f}")
        
        layer_results = {
            'layer': layer,
            'matching_eps': matching_eps,
            'perm_type': perm_type,
            'samples': []
        }
        
        for idx, prompt in enumerate(tqdm(test_samples, desc=f"Layer {layer}")):
            try:
                ground_truth_tokens, decoded_tokens = run_kv_attack(
                    model,
                    tokenizer,
                    prompt,
                    layer,
                    perm_type=perm_type,
                    batch_sz=args.batch_size,
                    matching_eps=matching_eps,
                    next_token_proposal=next_token_proposal,
                    max_proposal_candidates=args.max_proposal_candidates,
                    device_map=device_map,
                )
                
                original_text = tokenizer.decode(ground_truth_tokens, skip_special_tokens=True)
                predicted_text = tokenizer.decode(decoded_tokens, skip_special_tokens=True)
                success = (original_text == predicted_text)
                
                sample_result = {
                    'sample_idx': idx,
                    'original_text': original_text,
                    'predicted_text': predicted_text,
                    'original_tokens': ground_truth_tokens,
                    'predicted_tokens': decoded_tokens,
                    'success': success,
                    'num_tokens': len(ground_truth_tokens)
                }
                
                layer_results['samples'].append(sample_result)
                
                status = "✓" if success else "✗"
                print(f"\n{status} Sample {idx}: {'SUCCESS' if success else 'FAILED'}")
                print(f"  Original : {original_text[:80]}{'...' if len(original_text) > 80 else ''}")
                print(f"  Predicted: {predicted_text[:80]}{'...' if len(predicted_text) > 80 else ''}")
            
            except Exception as e:
                print(f"\n❌ Error processing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
        
        # 统计成功率
        successful = sum(1 for s in layer_results['samples'] if s['success'])
        total = len(layer_results['samples'])
        success_rate = successful / total if total > 0 else 0
        
        layer_results['statistics'] = {
            'total_samples': total,
            'successful': successful,
            'success_rate': success_rate
        }
        
        results['layers'].append(layer_results)
        
        print(f"\n{'='*80}")
        print(f"Layer {layer} 完成!")
        print(f"  Success Rate: {successful}/{total} ({success_rate:.2%})")
        print(f"{'='*80}\n")
    
    # 保存结果
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"🎉 攻击完成! 最终结果")
    print(f"{'='*80}")
    for layer_result in results['layers']:
        layer = layer_result['layer']
        stats = layer_result['statistics']
        print(f"Layer {layer}: {stats['successful']}/{stats['total_samples']} ({stats['success_rate']:.2%})")
    
    print(f"\n✅ 结果已保存到: {args.output}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

