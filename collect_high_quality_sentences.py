#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
收集高质量句子的脚本：
1. 先从已完成的攻击结果中收集在18种情况下匹配率都超过95%的句子
2. 然后从lmsys-chat-1m-data数据集继续采样，挨个尝试攻击
3. 只保留英文句子，避免重复
4. 目标是收集100个句子
5. 每收集一个句子就实时写入JSON
6. 详细的日志记录
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
import random
import json
import glob
import logging
from datetime import datetime
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from typing import Optional

# 随机种子
RANDOM_SEED = 42

# 设置日志
def setup_logging(log_file="collect_sentences.log"):
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

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
    "llama3.2-1B": {
        "path": "/home/junjie_chen/models/llama3.2-1B",
        "layers": {"no_perm": [0, 8, 15], "with_perm": [0, 8, 15]}
    },
    "llama3-8B": {
        "path": "/home/junjie_chen/models/llama3-8B",
        "layers": {"no_perm": [0, 16, 31], "with_perm": [0, 16, 31]}
    },
    "qwen3-8B": {
        "path": "/home/junjie_chen/models/qwen3-8B",
        "layers": {"no_perm": [0, 18, 35], "with_perm": [0, 18, 35]}
    },
    "chatglm3-6B": {
        "path": "/home/junjie_chen/models/chatglm3-6B",
        "layers": {"no_perm": [0, 16, 31], "with_perm": [0, 16, 31]}
    },
    "llama3.2-3B": {
        "path": "/home/junjie_chen/models/llama3.2-3B",
        "layers": {"no_perm": [0, 8, 15], "with_perm": [0, 8, 15]}
    },
}

def is_english(text):
    """检查文本是否主要是英文"""
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    ratio = ascii_chars / len(text)
    return ratio > 0.8

def load_existing_qualified_sentences(tokenizer, threshold=0.95):
    """从已有的_sentences.json文件中加载所有符合条件的句子"""
    sentence_files = glob.glob("*_sentences.json")
    all_sentences = {}  # {sentence_idx: {config_key: {"original": ..., "predicted": ...}}}
    
    logger.info(f"找到 {len(sentence_files)} 个句子文件")
    
    for filepath in sentence_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        model_name = data.get('model_name', 'unknown')
        perm_type = data.get('perm_type', 'unknown')
        layer = data.get('layer', 'unknown')
        
        config_key = f"{model_name}_{perm_type}_layer{layer}"
        
        sentences = data.get('sentences', [])
        for sent in sentences:
            idx = sent.get('sample_idx', -1)
            if idx not in all_sentences:
                all_sentences[idx] = {}
            all_sentences[idx][config_key] = {
                'original': sent.get('original', ''),
                'predicted': sent.get('predicted', '')
            }
    
    # 统计有多少种配置
    all_configs = set()
    for idx, configs in all_sentences.items():
        all_configs.update(configs.keys())
    
    logger.info(f"找到 {len(all_configs)} 种配置")
    
    qualified_sentences = []
    
    for idx, configs in sorted(all_sentences.items()):
        # 检查是否有足够的配置数据
        if len(configs) < len(all_configs):
            continue
        
        # 获取原始句子
        original_sentence = list(configs.values())[0]['original']
        
        # 检查是否是英文
        if not is_english(original_sentence):
            continue
        
        # 检查所有配置的匹配率
        all_qualified = True
        match_rates = {}
        results_for_sentence = {}
        
        for config_key, data in configs.items():
            original = data['original']
            predicted = data['predicted']
            
            orig_tokens = tokenizer.encode(original, add_special_tokens=False)
            pred_tokens = tokenizer.encode(predicted, add_special_tokens=False)
            
            if len(orig_tokens) == 0:
                match_rate = 0.0
            else:
                matches = sum(1 for o, p in zip(orig_tokens, pred_tokens) if o == p)
                match_rate = matches / len(orig_tokens)
            
            match_rates[config_key] = match_rate
            results_for_sentence[config_key] = {
                'original': original,
                'predicted': predicted,
                'match_rate': match_rate
            }
            
            if match_rate < threshold:
                all_qualified = False
        
        if all_qualified:
            qualified_sentences.append({
                'sentence': original_sentence,
                'sample_idx': idx,
                'source': 'existing_results',
                'match_rates': match_rates,
                'results': results_for_sentence
            })
    
    return qualified_sentences

def save_qualified_sentence(output_file, sentence_data):
    """保存单个符合条件的句子到JSON（自动去重）"""
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {'sentences': []}
    
    # 检查是否已存在相同句子（防止重复）
    sentence_text = sentence_data.get('sentence', '')
    existing_sentences = [s.get('sentence', '') for s in data.get('sentences', [])]
    
    if sentence_text not in existing_sentences:
        data['sentences'].append(sentence_data)
        data['total_count'] = len(data['sentences'])
        data['last_updated'] = datetime.now().isoformat()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        logger.warning(f"句子已存在，跳过保存: {sentence_text[:50]}...")

def save_sentence_results(results_file, sentence_idx, sentence, results):
    """保存单个句子在18种配置下的详细结果"""
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {'sentence_results': []}
    
    data['sentence_results'].append({
        'sentence_idx': sentence_idx,
        'original_sentence': sentence,
        'configs': results,
        'timestamp': datetime.now().isoformat()
    })
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_config(config_file: str, model_name: str, layer: int) -> float:
    """从配置文件加载matching_eps"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        if model_name in config:
            layer_str = str(layer)
            if layer_str in config[model_name]['layers']:
                return config[model_name]['layers'][layer_str]['matching_eps']
        
        logger.warning(f"未找到 {model_name} layer {layer} 的配置，使用默认值 1.0")
        return 1.0
    except FileNotFoundError:
        logger.warning(f"配置文件 {config_file} 不存在，使用默认值 1.0")
        return 1.0

def gen_kv_states(model, tokenizer, sentence, layers=[1], device_map="cuda"):
    """生成指定层的K和V states"""
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

def generate_permutation(N: int, d: int, perm_type: str, device: torch.device) -> tuple:
    """生成置换索引"""
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

def kv_matching_attack(
    model,
    tokenizer,
    perm_k_states: torch.Tensor,
    perm_v_states: torch.Tensor,
    layer: int,
    perm_type: str = "None",
    batch_sz: int = 128,
    matching_eps: float = 1.0,
    max_proposal_candidates: int = 5000,
    device_map: str = "cuda",
    ground_truth_tokens: list = None,
    verbose: bool = True,
    token_log_path: Optional[str] = None,
) -> tuple[list, bool]:
    """使用KV cache执行vocabulary matching attack"""
    use_sort = (perm_type == "D" or perm_type == "SD")
    
    vocab_sz = model.config.vocab_size
    num_tokens = perm_k_states.shape[0]
    
    input_tokens = []
    aborted_due_to_eps = False
    token_log_f = None
    if token_log_path:
        os.makedirs(os.path.dirname(token_log_path), exist_ok=True)
        # JSONL：每个token一行，便于实时追加与恢复查看
        token_log_f = open(token_log_path, "a", encoding="utf-8")
    
    for i in range(num_tokens):
        # 第一个token直接使用正确的token
        if i == 0 and ground_truth_tokens is not None and len(ground_truth_tokens) > 0:
            correct_token = ground_truth_tokens[0]
            input_tokens.append(correct_token)
            if verbose:
                print(f"        ✓ Token {i}: {correct_token} ('{tokenizer.decode([correct_token])}') [使用正确的token]")
            if token_log_f:
                token_log_f.write(json.dumps({
                    "token_index": i,
                    "token_id": int(correct_token),
                    "token_text": tokenizer.decode([correct_token]),
                    "best_error": 0.0,
                    "eps": float(matching_eps),
                    "status": "gt_first_token",
                }, ensure_ascii=False) + "\n")
                token_log_f.flush()
            continue
        
        global_best_error = float('inf')
        global_best_token = None
        
        token_ids = gen_next_proposal(
            model,
            torch.LongTensor(input_tokens).unsqueeze(0).to(device_map)
        )
        max_search_tokens = min(max_proposal_candidates, vocab_sz)
        
        for batch_start in range(0, max_search_tokens, batch_sz):
            batch_end = min(batch_start + batch_sz, max_search_tokens)
            actual_batch_sz = batch_end - batch_start
            
            batch_ids = token_ids[batch_start:batch_end].reshape(-1, 1)
            
            if i > 0:
                batch_input_tokens = (
                    torch.tensor(input_tokens)
                    .to(device_map)
                    .reshape(1, -1)
                    .repeat(actual_batch_sz, 1)
                )
                batch_ids = torch.cat([batch_input_tokens, batch_ids], dim=-1).long()
            
            with torch.no_grad():
                outputs = model(batch_ids, use_cache=True, output_hidden_states=True)
            
            k_cache = outputs.past_key_values[layer][0]
            v_cache = outputs.past_key_values[layer][1]
            
            batch_size, num_heads, seq_len, head_dim = k_cache.shape
            batch_k = k_cache[:, :, -1, :].reshape(batch_size, num_heads * head_dim)
            batch_v = v_cache[:, :, -1, :].reshape(batch_size, num_heads * head_dim)
            
            perm_k_row = perm_k_states[i, :]
            perm_v_row = perm_v_states[i, :]
            
            if use_sort:
                sorted_perm_k, _ = torch.sort(perm_k_row)
                sorted_perm_v, _ = torch.sort(perm_v_row)
            else:
                sorted_perm_k = perm_k_row
                sorted_perm_v = perm_v_row
            
            batch_best_error = float('inf')
            batch_best_token = None
            
            for j in range(actual_batch_sz):
                if use_sort:
                    sorted_k, _ = torch.sort(batch_k[j])
                    sorted_v, _ = torch.sort(batch_v[j])
                else:
                    sorted_k = batch_k[j]
                    sorted_v = batch_v[j]
                
                k_error = torch.sum(torch.abs(sorted_perm_k - sorted_k)).item()
                v_error = torch.sum(torch.abs(sorted_perm_v - sorted_v)).item()
                total_error = k_error + v_error
                
                if total_error < global_best_error:
                    global_best_error = total_error
                    global_best_token = token_ids[batch_start + j].item()
                
                if total_error < batch_best_error:
                    batch_best_error = total_error
                    batch_best_token = token_ids[batch_start + j].item()
            
            del outputs
            torch.cuda.empty_cache()
            
            if batch_best_error < matching_eps:
                global_best_error = batch_best_error
                global_best_token = batch_best_token
                break
            
            # 如果这是最后一个batch且还没找到低于eps的，打印警告
            if batch_end >= max_search_tokens and global_best_error > matching_eps:
                if verbose:
                    print(f"        ⚠ No match for token {i} under eps={matching_eps:.4f}")
                    print(f"           Best error: {global_best_error:.4f} for token {global_best_token} ('{tokenizer.decode([global_best_token])}')")
        
        input_tokens.append(global_best_token)
        
        # 打印进度
        if verbose:
            status = "✓" if global_best_error < matching_eps else "⚠"
            print(f"        {status} Token {i}: {global_best_token} ('{tokenizer.decode([global_best_token])}'), "
                  f"error={global_best_error:.4f}, eps={matching_eps:.4f}")
        
        # 实时记录每个token（JSONL）
        if token_log_f:
            token_log_f.write(json.dumps({
                "token_index": i,
                "token_id": int(global_best_token) if global_best_token is not None else None,
                "token_text": tokenizer.decode([global_best_token]) if global_best_token is not None else "",
                "best_error": float(global_best_error),
                "eps": float(matching_eps),
                "below_eps": bool(global_best_error < matching_eps),
            }, ensure_ascii=False) + "\n")
            token_log_f.flush()
        
        # 如果当前token的最优误差仍然超过matching_eps，则提前终止本句子的攻击，切换到下一个句子
        if global_best_error > matching_eps:
            aborted_due_to_eps = True
            if verbose:
                print(f"        ⏭️  提前终止该句子：token {i} 最优误差 {global_best_error:.4f} > eps {matching_eps:.4f}")
            break
    
    if token_log_f:
        token_log_f.close()
    return input_tokens, aborted_due_to_eps

def run_single_attack(
    model,
    tokenizer,
    sentence,
    layer,
    perm_type,
    matching_eps,
    device_map="cuda",
    verbose=True,
    token_log_path: Optional[str] = None,
):
    """执行单个攻击配置"""
    ground_truth_tokens = tokenizer.encode(sentence, add_special_tokens=False)
    
    if verbose:
        print(f"      🔑 Attacking KV Cache at layer {layer}, perm_type={perm_type}")
        print(f"      📝 原文 ({len(ground_truth_tokens)} tokens): {sentence[:60]}{'...' if len(sentence) > 60 else ''}")
    
    k_states_list, v_states_list = gen_kv_states(model, tokenizer, sentence, layers=[layer], device_map=device_map)
    k_states = k_states_list[0]
    v_states = v_states_list[0]
    
    N, d = k_states.size()
    seq_perm, dim_perm = generate_permutation(N, d, perm_type, k_states.device)
    
    perm_k_states = apply_permutation(k_states, seq_perm, dim_perm)
    perm_v_states = apply_permutation(v_states, seq_perm, dim_perm)
    
    decoded_tokens, aborted_due_to_eps = kv_matching_attack(
        model,
        tokenizer,
        perm_k_states,
        perm_v_states,
        layer,
        perm_type=perm_type,
        batch_sz=128,
        matching_eps=matching_eps,
        max_proposal_candidates=5000,
        device_map=device_map,
        ground_truth_tokens=ground_truth_tokens,
        verbose=verbose,
        token_log_path=token_log_path,
    )
    
    if aborted_due_to_eps:
        if verbose:
            print(f"      ⏭️  本配置提前终止：误差超过 matching_eps，切换到下一个句子")
        return {
            'original': tokenizer.decode(ground_truth_tokens, skip_special_tokens=True),
            'predicted': "",
            'match_rate': 0.0,
            'success': False,
            'aborted_due_to_eps': True,
        }
    
    original_text = tokenizer.decode(ground_truth_tokens, skip_special_tokens=True)
    predicted_text = tokenizer.decode(decoded_tokens, skip_special_tokens=True)
    
    # 计算匹配率
    if len(ground_truth_tokens) == 0:
        match_rate = 0.0
    else:
        matches = sum(1 for o, p in zip(ground_truth_tokens, decoded_tokens) if o == p)
        match_rate = matches / len(ground_truth_tokens)
    
    if verbose:
        status = "✓ SUCCESS" if original_text == predicted_text else "✗ FAILED"
        print(f"      📊 结果: {status}")
        print(f"         Original : {original_text[:70]}{'...' if len(original_text) > 70 else ''}")
        print(f"         Predicted: {predicted_text[:70]}{'...' if len(predicted_text) > 70 else ''}")
        print(f"         Match Rate: {match_rate:.4f}")
    
    return {
        'original': original_text,
        'predicted': predicted_text,
        'match_rate': match_rate,
        'success': original_text == predicted_text
    }

def test_sentence_all_configs(models, tokenizers, sentence, threshold=0.95, verbose=True, token_log_dir: Optional[str] = None, sentence_tag: Optional[str] = None):
    """
    测试一个句子在所有18种配置下的匹配率
    如果某个配置的匹配率低于阈值，立即返回False
    返回: (是否全部通过, 所有配置的结果, 失败的配置名)
    """
    results = {}
    config_count = 0
    total_configs = 18  # 3 models * 2 perm_types * 3 layers
    
    for model_name in ["llama3-8B", "llama3.2-1B", "qwen3-8B"]:
        model = models[model_name]['model']
        tokenizer = tokenizers[model_name]
        
        for perm_type in ["None", "D"]:
            config_file = "kv_attack_config_no_perm.json" if perm_type == "None" else "kv_attack_config_with_perm.json"
            perm_key = "no_perm" if perm_type == "None" else "with_perm"
            layers = MODEL_CONFIGS[model_name]["layers"][perm_key]
            
            for layer in layers:
                config_count += 1
                config_key = f"{model_name}_{perm_type}_layer{layer}"
                
                print(f"\n    ===== 配置 {config_count}/{total_configs}: {config_key} =====")
                logger.info(f"    测试配置 [{config_count}/{total_configs}]: {config_key}")
                
                matching_eps = load_config(config_file, model_name, layer)
                print(f"    使用 matching_eps: {matching_eps:.6f}")
                
                try:
                    token_log_path = None
                    if token_log_dir:
                        # sentence_tag 用于区分不同句子（例如 tested_count），避免覆盖
                        tag = sentence_tag or "sentence"
                        token_log_path = os.path.join(token_log_dir, f"{tag}__{config_key}.jsonl")
                    result = run_single_attack(
                        model,
                        tokenizer,
                        sentence,
                        layer,
                        perm_type,
                        matching_eps,
                        verbose=verbose,
                        token_log_path=token_log_path,
                    )
                    results[config_key] = result
                    
                    logger.info(f"      match_rate={result['match_rate']:.4f}, eps={matching_eps:.4f}")
                    if result.get('aborted_due_to_eps'):
                        print(f"\n    ⏭️  提前终止句子：{config_key} 误差超过 matching_eps，直接切换下一个句子")
                        logger.warning(f"    ⏭️  句子提前终止：{config_key} 误差超过 matching_eps")
                        return False, results, config_key
                    
                    if result['match_rate'] < threshold:
                        print(f"\n    ❌❌❌ 失败于 {config_key}: match_rate={result['match_rate']:.4f} < {threshold}")
                        logger.warning(f"    ❌ 失败于 {config_key}: match_rate={result['match_rate']:.4f} < {threshold}")
                        return False, results, config_key
                    else:
                        print(f"    ✅ 配置 {config_key} 通过 (match_rate={result['match_rate']:.4f} >= {threshold})")
                except Exception as e:
                    print(f"\n    ❌❌❌ 攻击出错于 {config_key}: {str(e)}")
                    logger.error(f"    ❌ 攻击出错于 {config_key}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return False, results, config_key
    
    print(f"\n    🎉🎉🎉 所有 {total_configs} 个配置全部通过！")
    return True, results, None

def load_lmsys_data(data_dir):
    """从lmsys数据集加载英文数据"""
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "data", "*.parquet")))
    
    for pf in parquet_files:
        logger.info(f"读取文件: {pf}")
        df = pd.read_parquet(pf)
        english_df = df[df['language'] == 'English']
        
        for idx, row in english_df.iterrows():
            try:
                conversation = row['conversation']
                if conversation is not None and len(conversation) > 0:
                    first_msg = conversation[0]
                    if isinstance(first_msg, dict):
                        user_message = first_msg.get('content', '')
                    else:
                        continue
                    if user_message and len(user_message) > 20:
                        yield user_message
            except Exception:
                continue

def truncate_prompt(tokenizer, text: str, num_tokens: int) -> str:
    """截断prompt到指定的token数量"""
    token_ids = tokenizer.encode(text, add_special_tokens=False)[:num_tokens]
    return tokenizer.decode(token_ids, skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="收集高质量句子")
    parser.add_argument("--target", type=int, default=100, help="目标句子数量")
    parser.add_argument("--threshold", type=float, default=0.95, help="匹配率阈值")
    parser.add_argument("--max_tokens", type=int, default=50, help="每个句子的最大token数")
    parser.add_argument("--output", type=str, default="high_quality_sentences.json", help="输出文件（符合条件的句子）")
    parser.add_argument("--results_output", type=str, default="sentence_attack_results.json", help="输出文件（18种配置的详细结果）")
    parser.add_argument("--skip_existing", action="store_true", help="跳过从已有结果收集，直接从数据集开始")
    parser.add_argument("--verbose", action="store_true", default=True, help="显示详细攻击过程")
    parser.add_argument("--quiet", action="store_true", help="静默模式，不显示详细攻击过程")
    parser.add_argument("--token_log_dir", type=str, default="token_logs",
                        help="逐token实时日志目录（JSONL）；为空则不记录")
    args = parser.parse_args()
    
    # 处理verbose参数
    verbose = not args.quiet
    
    set_seed(RANDOM_SEED)
    
    logger.info("=" * 80)
    logger.info("收集高质量句子")
    logger.info(f"目标数量: {args.target}")
    logger.info(f"匹配率阈值: {args.threshold}")
    logger.info(f"输出文件: {args.output}")
    logger.info(f"详细结果文件: {args.results_output}")
    logger.info("=" * 80)
    
    # 加载一个tokenizer用于计算匹配率
    base_tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS["llama3-8B"]["path"])
    
    # 已收集的句子
    collected_texts = set()  # 用于去重
    collected_count = 0
    
    # 检查并加载已有的输出文件（防止重复收集）
    if os.path.exists(args.output):
        logger.info(f"发现已有输出文件: {args.output}，正在加载...")
        try:
            with open(args.output, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            existing_sentences = existing_data.get('sentences', [])
            for sent_data in existing_sentences:
                sentence = sent_data.get('sentence', '')
                if sentence:
                    collected_texts.add(sentence)
                    collected_count = max(collected_count, sent_data.get('idx', 0))
            
            logger.info(f"已加载 {len(existing_sentences)} 个已有句子，当前计数: {collected_count}")
        except Exception as e:
            logger.warning(f"加载已有文件失败: {e}，将重新开始")
            collected_texts = set()
            collected_count = 0
    
    # 初始化或更新输出文件
    if collected_count == 0:
        # 如果没有任何已有数据，初始化文件
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({'sentences': [], 'total_count': 0, 'target': args.target, 'threshold': args.threshold}, f, indent=2, ensure_ascii=False)
        
        with open(args.results_output, 'w', encoding='utf-8') as f:
            json.dump({'sentence_results': []}, f, indent=2, ensure_ascii=False)
    else:
        logger.info(f"将从已有进度继续：已收集 {collected_count} 个句子，目标 {args.target} 个")
    
    # 1. 先从已有结果中收集
    if not args.skip_existing:
        logger.info("\n" + "=" * 40)
        logger.info("步骤1: 从已有攻击结果中收集符合条件的句子")
        logger.info("=" * 40)
        
        qualified = load_existing_qualified_sentences(base_tokenizer, args.threshold)
        
        for item in qualified:
            if collected_count >= args.target:
                break
            
            sentence = item['sentence']
            if sentence in collected_texts:
                continue
            
            collected_texts.add(sentence)
            collected_count += 1
            
            # 保存到输出文件
            save_qualified_sentence(args.output, {
                'idx': collected_count,
                'sentence': sentence,
                'source': 'existing_results',
                'sample_idx': item['sample_idx'],
                'match_rates': item['match_rates']
            })
            
            # 保存详细结果
            save_sentence_results(args.results_output, collected_count, sentence, item['results'])
            
            logger.info(f"✅ 收集句子 #{collected_count}: {sentence[:60]}...")
        
        logger.info(f"从已有结果中收集到 {collected_count} 个符合条件的英文句子")
    
    # 2. 如果不够，从lmsys数据集继续采样
    if collected_count < args.target:
        logger.info("\n" + "=" * 40)
        logger.info(f"步骤2: 从lmsys数据集采样，还需要 {args.target - collected_count} 个句子")
        logger.info("=" * 40)
        
        logger.info("加载模型...")
        device_map = "cuda"
        model_dtype = torch.bfloat16
        
        models = {}
        tokenizers = {}
        
        for model_name, config in MODEL_CONFIGS.items():
            logger.info(f"  加载 {model_name}...")
            tokenizers[model_name] = AutoTokenizer.from_pretrained(config["path"])
            models[model_name] = {
                'model': AutoModelForCausalLM.from_pretrained(
                    config["path"], 
                    torch_dtype=model_dtype, 
                    attn_implementation="eager"
                ).to(device_map)
            }
            models[model_name]['model'].eval()
        
        logger.info("模型加载完成！")
        
        data_dir = "/home/junjie_chen/datasets/lmsys-chat-1m-data"
        tested_count = 0
        
        for sentence in load_lmsys_data(data_dir):
            if collected_count >= args.target:
                break
            
            # 截断句子
            sentence = truncate_prompt(base_tokenizer, sentence, args.max_tokens)
            
            # 检查是否重复
            if sentence in collected_texts:
                continue
            
            # 检查是否是英文
            if not is_english(sentence):
                continue
            
            # 检查长度
            if len(sentence) < 20:
                continue
            
            tested_count += 1
            print("\n" + "=" * 80)
            print(f"🔍 测试句子 #{tested_count} (已收集: {collected_count}/{args.target})")
            print("=" * 80)
            print(f"📝 句子内容: {sentence}")
            print(f"📏 句子长度: {len(sentence)} 字符")
            logger.info(f"测试句子 #{tested_count} (已收集: {collected_count}/{args.target})")
            logger.info(f"  句子: {sentence[:80]}{'...' if len(sentence) > 80 else ''}")
            
            # 测试所有配置
            qualified, results, failed_config = test_sentence_all_configs(
                models,
                tokenizers,
                sentence,
                args.threshold,
                verbose=verbose,
                token_log_dir=(args.token_log_dir if args.token_log_dir else None),
                sentence_tag=f"tested_{tested_count}_collected_{collected_count}",
            )
            
            if qualified:
                collected_count += 1
                collected_texts.add(sentence)
                
                match_rates = {k: v['match_rate'] for k, v in results.items()}
                min_rate = min(match_rates.values())
                mean_rate = sum(match_rates.values()) / len(match_rates)
                
                # 保存到输出文件
                save_qualified_sentence(args.output, {
                    'idx': collected_count,
                    'sentence': sentence,
                    'source': 'lmsys_dataset',
                    'match_rates': match_rates,
                    'min_match_rate': min_rate,
                    'mean_match_rate': mean_rate
                })
                
                # 保存详细结果
                save_sentence_results(args.results_output, collected_count, sentence, results)
                
                print("\n" + "🎊" * 20)
                print(f"✅✅✅ 句子 #{collected_count} 合格！已保存到JSON")
                print(f"    最小匹配率: {min_rate:.4f}")
                print(f"    平均匹配率: {mean_rate:.4f}")
                print(f"    当前进度: {collected_count}/{args.target}")
                print("🎊" * 20)
                logger.info(f"  ✅ 句子合格！(min={min_rate:.4f}, mean={mean_rate:.4f})")
                logger.info(f"  当前进度: {collected_count}/{args.target}")
            else:
                print("\n" + "❌" * 20)
                print(f"❌ 句子不合格，失败于: {failed_config}")
                print("❌" * 20)
                logger.info(f"  ❌ 句子不合格，失败于: {failed_config}")
    
    logger.info("\n" + "=" * 80)
    logger.info(f"收集完成！共收集 {collected_count} 个高质量句子")
    logger.info(f"符合条件的句子已保存到: {args.output}")
    logger.info(f"详细攻击结果已保存到: {args.results_output}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
