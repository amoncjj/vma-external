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

def generate_permutation(N: int, d: int, perm_type: str, device: torch.device) -> tuple:
    """生成置换索引，用于K和V共享同一置换"""
    seq_perm = None
    dim_perm = None
    
    if perm_type == "S":
        seq_perm = torch.randperm(N, device=device)
    elif perm_type == "D":
        dim_perm = torch.randperm(d, device=device)
    elif perm_type == "SD":
        seq_perm = torch.randperm(N, device=device)
        dim_perm = torch.randperm(d, device=device)
    
    return seq_perm, dim_perm

def apply_permutation(states: torch.Tensor, seq_perm: torch.Tensor, dim_perm: torch.Tensor) -> torch.Tensor:
    """应用预生成的置换"""
    result = states
    if dim_perm is not None:
        result = result[:, dim_perm]
    if seq_perm is not None:
        result = result[seq_perm]
    return result

def permute_states(states: torch.Tensor, perm_type: str) -> torch.Tensor:
    """对states进行置换（单独使用时，不与其他tensor共享置换）"""
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

def compute_kv_errors(
    model,
    tokenizer,
    sentence: str,
    layer: int,
    perm_type: str = "None",
    device_map: str = "cuda",
) -> list[dict]:
    """
    直接使用正确的token序列，计算每个token的K和V的matching error
    
    Args:
        model: 模型
        tokenizer: tokenizer
        sentence: 输入句子
        layer: 要测试的层
        perm_type: 置换类型
        device_map: 设备
    
    Returns:
        每个token的error信息列表
    """
    # 获取ground truth tokens
    ground_truth_tokens = tokenizer.encode(sentence, add_special_tokens=False)
    num_tokens = len(ground_truth_tokens)
    
    # 生成permuted K和V states
    k_states_list, v_states_list = gen_kv_states(model, tokenizer, sentence, layers=[layer], device_map=device_map)
    k_states = k_states_list[0]
    v_states = v_states_list[0]
    
    # 生成共享的置换（K和V使用相同的置换）
    N, d = k_states.size()
    seq_perm, dim_perm = generate_permutation(N, d, perm_type, k_states.device)
    
    # 应用相同的置换到K和V
    perm_k_states = apply_permutation(k_states, seq_perm, dim_perm)
    perm_v_states = apply_permutation(v_states, seq_perm, dim_perm)
    
    # 记录每个token的error
    error_logs = []
    
    # 根据 perm_type 决定是否使用排序
    use_sort = (perm_type == "D" or perm_type == "SD")
    
    # 逐步构建token序列，每次添加一个正确的token
    for i in range(num_tokens):
        current_tokens = ground_truth_tokens[:i]
        next_token = ground_truth_tokens[i]
        
        if i == 0:
            input_ids = torch.tensor([[next_token]], device=device_map)
        else:
            input_ids = torch.tensor([current_tokens + [next_token]], device=device_map)
        
        # Forward pass获取K和V states
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True, output_hidden_states=True)
        
        # 提取最后一个token的K和V
        k_cache = outputs.past_key_values[layer][0]
        v_cache = outputs.past_key_values[layer][1]
        
        batch_size, num_heads, seq_len, head_dim = k_cache.shape
        k_last = k_cache[:, :, -1, :].reshape(num_heads * head_dim)
        v_last = v_cache[:, :, -1, :].reshape(num_heads * head_dim)
        
        # 计算L1距离
        perm_k_row = perm_k_states[i, :]
        perm_v_row = perm_v_states[i, :]
        
        # 根据 perm_type 决定是否排序
        if use_sort:
            sorted_perm_k, _ = torch.sort(perm_k_row)
            sorted_k, _ = torch.sort(k_last)
            sorted_perm_v, _ = torch.sort(perm_v_row)
            sorted_v, _ = torch.sort(v_last)
        else:
            sorted_perm_k = perm_k_row
            sorted_k = k_last
            sorted_perm_v = perm_v_row
            sorted_v = v_last
        
        k_error = torch.sum(torch.abs(sorted_perm_k - sorted_k)).item()
        v_error = torch.sum(torch.abs(sorted_perm_v - sorted_v)).item()
        
        total_error = k_error + v_error
        
        token_log = {
            'token_index': i,
            'token_id': next_token,
            'token_text': tokenizer.decode([next_token]),
            'k_error': k_error,
            'v_error': v_error,
            'total_error': total_error,
        }
        error_logs.append(token_log)
        
        if (i + 1) % 10 == 0:
            print(f"  Token {i+1}/{num_tokens}: '{tokenizer.decode([next_token])}', "
                  f"k_err={k_error:.4f}, v_err={v_error:.4f}, total={total_error:.4f}")
        
        del outputs
        torch.cuda.empty_cache()
    
    return error_logs

def truncate_prompt(tokenizer, text: str, num_tokens: int) -> str:
    """截断prompt到指定的token数量"""
    token_ids = tokenizer.encode(text, add_special_tokens=False)[:num_tokens]
    return tokenizer.decode(token_ids, skip_special_tokens=True)

def load_lmsys_samples(tokenizer, data_file, num_samples=100, max_tokens=50, seed=RANDOM_SEED):
    """从lmsys数据集加载样本"""
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

def print_layer_recommendations(layer, all_total_errors, num_samples, total_tokens):
    """打印该层的详细推荐配置"""
    print(f"\n{'='*80}")
    print(f"Layer {layer} 测试完成 - 详细推荐配置")
    print(f"{'='*80}")
    print(f"样本数: {num_samples}")
    print(f"总Token数: {total_tokens}")
    
    # 基本统计
    print(f"\n📊 Error统计:")
    print(f"  最小值: {np.min(all_total_errors):.6f}")
    print(f"  最大值: {np.max(all_total_errors):.6f}")
    print(f"  平均值: {np.mean(all_total_errors):.6f}")
    print(f"  中位数: {np.median(all_total_errors):.6f}")
    print(f"  标准差: {np.std(all_total_errors):.6f}")
    
    # 百分位数统计
    print(f"\n📈 百分位数分布:")
    percentiles = [50, 75, 80, 85, 90, 95, 99, 99.5, 99.9]
    for p in percentiles:
        val = np.percentile(all_total_errors, p)
        print(f"  {p:5.1f}%: {val:.6f}  (覆盖 {p:.1f}% 的token)")
    
    # 推荐配置（以成功率优先）
    print(f"\n🎯 推荐的matching_eps配置 (按成功率优先):")
    print(f"{'策略':<20} {'matching_eps':<15} {'覆盖率':<15} {'说明':<30}")
    print(f"{'-'*80}")
    
    recommendations = [
        ("极致成功率", 99.9, "几乎100%成功"),
        ("非常保守", 99.5, "99.5%以上成功"),
        ("保守", 99, "99%以上成功"),
        ("推荐(平衡)", 95, "95%以上成功，推荐使用"),
        ("激进", 90, "90%以上成功"),
        ("非常激进", 85, "85%以上成功"),
        ("高风险", 80, "80%以上成功，可能失败较多"),
        ("实验性", 75, "75%以上成功，仅供实验"),
    ]
    
    for strategy, percentile, desc in recommendations:
        eps_value = np.percentile(all_total_errors, percentile)
        print(f"{strategy:<20} {eps_value:<15.6f} {percentile:<15.1f}%  {desc:<30}")
    
    # 默认推荐
    recommended_eps = np.percentile(all_total_errors, 95)
    print(f"\n⭐ 默认推荐: {recommended_eps:.6f} (95%分位，平衡成功率和严格性)")
    print(f"{'='*80}\n")

def test_model(model_name: str, model_path: str, num_samples: int, max_tokens: int, perm_type: str = "None"):
    """测试单个模型并返回推荐配置"""
    
    print(f"\n{'='*80}")
    print(f"测试模型: {model_name}")
    print(f"模型路径: {model_path}")
    print(f"置换类型: {perm_type}")
    print(f"{'='*80}\n")
    
    # 加载模型
    device_map = "cuda"
    model_dtype = torch.bfloat16
    
    print(f"Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=model_dtype, attn_implementation="eager"
    ).to(device_map)
    model.eval()
    
    num_hidden_layers = model.config.num_hidden_layers
    print(f"Model loaded. Number of layers: {num_hidden_layers}")
    
    # 确定要测试的层
    # 从layer 0开始（第一层）
    first_layer = 0
    middle_layer = num_hidden_layers // 2
    last_layer = num_hidden_layers - 1
    
    test_layers = [first_layer, middle_layer, last_layer]
    
    print(f"\nTesting layers: {test_layers}")
    print(f"  - First layer: {first_layer} (layer 0)")
    print(f"  - Middle layer: {middle_layer}")
    print(f"  - Last layer: {last_layer}")
    
    # 加载数据
    data_file = "/home/junjie_chen/datasets/lmsys-chat-1m-data-1000/data.json"
    test_samples = load_lmsys_samples(tokenizer, data_file, num_samples=num_samples, 
                                      max_tokens=max_tokens, seed=RANDOM_SEED)
    
    print(f"\nLoaded {len(test_samples)} samples\n")
    
    # 对每一层运行测试
    layer_results = {}
    
    for layer in test_layers:
        print(f"\n{'='*80}")
        print(f"Testing Layer {layer}")
        print(f"{'='*80}\n")
        
        sample_results = []
        
        for idx, prompt in enumerate(tqdm(test_samples, desc=f"Layer {layer}")):
            try:
                error_logs = compute_kv_errors(model, tokenizer, prompt, layer, perm_type, device_map)
                
                total_errors = [log['total_error'] for log in error_logs]
                k_errors = [log['k_error'] for log in error_logs]
                v_errors = [log['v_error'] for log in error_logs]
                
                sample_result = {
                    'sample_idx': idx,
                    'text': prompt,
                    'num_tokens': len(error_logs),
                    'error_statistics': {
                        'total': {
                            'min': float(np.min(total_errors)),
                            'max': float(np.max(total_errors)),
                            'mean': float(np.mean(total_errors)),
                            'median': float(np.median(total_errors)),
                            'std': float(np.std(total_errors)),
                        },
                        'k': {
                            'min': float(np.min(k_errors)),
                            'max': float(np.max(k_errors)),
                            'mean': float(np.mean(k_errors)),
                            'median': float(np.median(k_errors)),
                        },
                        'v': {
                            'min': float(np.min(v_errors)),
                            'max': float(np.max(v_errors)),
                            'mean': float(np.mean(v_errors)),
                            'median': float(np.median(v_errors)),
                        }
                    },
                    'error_logs': error_logs
                }
                
                sample_results.append(sample_result)
            
            except Exception as e:
                print(f"\n❌ Error processing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
        
        # 计算该层的全局统计
        all_total_errors = []
        all_k_errors = []
        all_v_errors = []
        
        for result in sample_results:
            if 'error_logs' in result:
                for log in result['error_logs']:
                    all_total_errors.append(log['total_error'])
                    all_k_errors.append(log['k_error'])
                    all_v_errors.append(log['v_error'])
        
        # 立即打印该层的推荐配置
        print_layer_recommendations(layer, all_total_errors, len(sample_results), len(all_total_errors))
        
        layer_statistics = {
            'layer': layer,
            'num_samples': len(sample_results),
            'total_tokens': len(all_total_errors),
            'global_statistics': {
                'total_error': {
                    'min': float(np.min(all_total_errors)),
                    'max': float(np.max(all_total_errors)),
                    'mean': float(np.mean(all_total_errors)),
                    'median': float(np.median(all_total_errors)),
                    'std': float(np.std(all_total_errors)),
                    'percentiles': {
                        '50': float(np.percentile(all_total_errors, 50)),
                        '75': float(np.percentile(all_total_errors, 75)),
                        '80': float(np.percentile(all_total_errors, 80)),
                        '85': float(np.percentile(all_total_errors, 85)),
                        '90': float(np.percentile(all_total_errors, 90)),
                        '95': float(np.percentile(all_total_errors, 95)),
                        '99': float(np.percentile(all_total_errors, 99)),
                        '99.5': float(np.percentile(all_total_errors, 99.5)),
                        '99.9': float(np.percentile(all_total_errors, 99.9)),
                    }
                },
                'k_error': {
                    'mean': float(np.mean(all_k_errors)),
                    'median': float(np.median(all_k_errors)),
                },
                'v_error': {
                    'mean': float(np.mean(all_v_errors)),
                    'median': float(np.median(all_v_errors)),
                }
            }
        }
        
        layer_results[f"layer_{layer}"] = layer_statistics
        
        # 立即保存当前层的结果到临时文件
        temp_output = {
            'model_name': model_name,
            'model_path': model_path,
            'perm_type': perm_type,
            'layer_results': layer_results
        }
        temp_file = f"kv_test_temp_{model_name.replace('.', '_')}_{perm_type}.json"
        with open(temp_file, 'w') as f:
            json.dump(temp_output, f, indent=2)
        print(f"💾 已保存临时结果到: {temp_file}\n")
    
    # 清理模型以释放内存
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return layer_results

def main():
    parser = argparse.ArgumentParser(description="测试KV cache攻击的matching_eps配置")
    parser.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS.keys()),
                        choices=list(MODEL_CONFIGS.keys()),
                        help="要测试的模型列表")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="每个模型测试的样本数")
    parser.add_argument("--max_tokens", type=int, default=50,
                        help="每个样本的最大token数")
    parser.add_argument("--perm_type", type=str, default="None",
                        choices=["None", "D"],
                        help="置换类型: None=无置换, D=维度置换")
    parser.add_argument("--output", type=str, default=None,
                        help="输出配置文件路径（默认根据perm_type自动命名）")
    
    args = parser.parse_args()
    
    # 根据perm_type自动设置输出文件名
    if args.output is None:
        if args.perm_type == "None":
            args.output = "kv_attack_config_no_perm.json"
        elif args.perm_type == "D":
            args.output = "kv_attack_config_with_perm.json"
        else:
            args.output = "kv_attack_config.json"
    
    # 设置随机种子
    set_seed(RANDOM_SEED)
    print(f"Random seed set to: {RANDOM_SEED}")
    
    print(f"\n{'='*80}")
    print(f"KV Cache 攻击配置测试")
    print(f"{'='*80}")
    print(f"测试模型: {args.models}")
    print(f"样本数: {args.num_samples}")
    print(f"最大token数: {args.max_tokens}")
    print(f"置换类型: {args.perm_type}")
    print(f"输出文件: {args.output}")
    print(f"{'='*80}\n")
    
    # 测试所有模型
    all_results = {}
    
    for model_name in args.models:
        model_path = MODEL_CONFIGS[model_name]
        
        try:
            layer_results = test_model(model_name, model_path, args.num_samples, args.max_tokens, args.perm_type)
            
            # 提取推荐配置（使用95%分位作为默认）
            recommended_config = {
                'model_name': model_name,
                'model_path': model_path,
                'num_layers': len(layer_results),
                'perm_type': args.perm_type,
                'layers': {}
            }
            
            for layer_key, stats in layer_results.items():
                layer = stats['layer']
                recommended_eps = stats['global_statistics']['total_error']['percentiles']['95']
                
                recommended_config['layers'][str(layer)] = {
                    'matching_eps': recommended_eps,
                    'mean_error': stats['global_statistics']['total_error']['mean'],
                    'median_error': stats['global_statistics']['total_error']['median'],
                    'total_tokens': stats['total_tokens'],
                    'all_percentiles': stats['global_statistics']['total_error']['percentiles']
                }
            
            all_results[model_name] = recommended_config
            
            # 每测试完一个模型就保存一次最终配置
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"💾 已保存配置到: {args.output}")
            
        except Exception as e:
            print(f"\n❌ Error testing model {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 最终保存配置
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"🎉 所有测试完成!")
    print(f"{'='*80}\n")
    
    # 打印推荐配置总结
    print("推荐的配置总结 (95%分位):\n")
    for model_name, config in all_results.items():
        print(f"{model_name}:")
        for layer, layer_config in config['layers'].items():
            print(f"  Layer {layer}: matching_eps = {layer_config['matching_eps']:.6f}")
        print()
    
    print(f"✅ 配置已保存到: {args.output}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

