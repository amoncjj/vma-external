#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析sentence_attack_results.json文件，计算四个维度的指标：
1. BERTScore
2. 余弦相似度
3. ROUGE-L
4. Token匹配率
"""

import json
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
from collections import defaultdict
import warnings
import os

# 尝试导入可选的依赖
try:
    from bert_score import score as bert_score
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    print("⚠️  Warning: bert_score not installed. BERTScore will be skipped.")
    print("   Install with: pip install bert-score")

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("⚠️  Warning: rouge-score not installed. ROUGE-L will be skipped.")
    print("   Install with: pip install rouge-score")

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("⚠️  Warning: sentence-transformers not installed. Cosine similarity will be skipped.")
    print("   Install with: pip install sentence-transformers")

def calculate_token_match_rate(tokenizer, original, predicted):
    """计算token匹配率"""
    orig_tokens = tokenizer.encode(original, add_special_tokens=False)
    pred_tokens = tokenizer.encode(predicted, add_special_tokens=False)
    
    if len(orig_tokens) == 0:
        return 0.0
    
    matches = sum(1 for o, p in zip(orig_tokens, pred_tokens) if o == p)
    return matches / len(orig_tokens)

def calculate_bertscore(originals, predicteds):
    """计算BERTScore"""
    if not HAS_BERTSCORE:
        return None
    
    try:
        # 抑制transformers的警告信息
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        import logging
        logging.getLogger('transformers').setLevel(logging.ERROR)
        
        # 使用warnings抑制警告
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', message='.*Some weights.*')
            P, R, F1 = bert_score(predicteds, originals, lang='en', verbose=False)
        
        return {
            'precision': float(P.mean().item()),
            'recall': float(R.mean().item()),
            'f1': float(F1.mean().item()),
            'precision_scores': [float(p.item()) for p in P],
            'recall_scores': [float(r.item()) for r in R],
            'f1_scores': [float(f.item()) for f in F1]
        }
    except Exception as e:
        print(f"⚠️  BERTScore计算错误: {e}")
        return None

def calculate_rouge_l(originals, predicteds):
    """计算ROUGE-L"""
    if not HAS_ROUGE:
        return None
    
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = []
        
        for orig, pred in zip(originals, predicteds):
            scores = scorer.score(orig, pred)
            rouge_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'mean': float(np.mean(rouge_scores)),
            'std': float(np.std(rouge_scores)),
            'scores': [float(s) for s in rouge_scores]
        }
    except Exception as e:
        print(f"⚠️  ROUGE-L计算错误: {e}")
        return None

def calculate_cosine_similarity(originals, predicteds, model):
    """计算余弦相似度"""
    if not HAS_SENTENCE_TRANSFORMERS:
        return None
    
    try:
        orig_embeddings = model.encode(originals, convert_to_numpy=True)
        pred_embeddings = model.encode(predicteds, convert_to_numpy=True)
        
        # 计算余弦相似度
        cosine_scores = []
        for orig_emb, pred_emb in zip(orig_embeddings, pred_embeddings):
            cosine = np.dot(orig_emb, pred_emb) / (np.linalg.norm(orig_emb) * np.linalg.norm(pred_emb))
            cosine_scores.append(float(cosine))
        
        return {
            'mean': float(np.mean(cosine_scores)),
            'std': float(np.std(cosine_scores)),
            'scores': cosine_scores
        }
    except Exception as e:
        print(f"⚠️  余弦相似度计算错误: {e}")
        return None

def analyze_sentence_results(json_file, output_file=None):
    """分析句子攻击结果"""
    # 设置环境变量抑制警告
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    import logging
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)
    
    print("=" * 80)
    print("分析句子攻击结果")
    print("=" * 80)
    
    # 加载数据
    print(f"\n加载文件: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sentence_results = data.get('sentence_results', [])
    print(f"找到 {len(sentence_results)} 个句子的结果")
    
    # 加载tokenizer
    print("\n加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("/home/junjie_chen/models/llama3-8B")
    
    # 加载sentence transformer模型（如果可用）
    sentence_model = None
    if HAS_SENTENCE_TRANSFORMERS:
        print("加载sentence transformer模型...")
        try:
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"⚠️  无法加载sentence transformer模型: {e}")
    
    # 收集所有配置的数据
    all_configs = set()
    for result in sentence_results:
        all_configs.update(result['configs'].keys())
    
    print(f"\n找到 {len(all_configs)} 种配置")
    
    # 为每种配置收集数据
    config_data = defaultdict(lambda: {'originals': [], 'predicteds': []})
    
    for result in sentence_results:
        for config_name, config_result in result['configs'].items():
            config_data[config_name]['originals'].append(config_result['original'])
            config_data[config_name]['predicteds'].append(config_result['predicted'])
    
    # 分析结果
    analysis_results = {
        'total_sentences': len(sentence_results),
        'total_configs': len(all_configs),
        'configs': {}
    }
    
    print("\n" + "=" * 80)
    print("开始计算指标...")
    print("=" * 80)
    
    for config_name in sorted(all_configs):
        print(f"\n分析配置: {config_name}")
        originals = config_data[config_name]['originals']
        predicteds = config_data[config_name]['predicteds']
        
        config_result = {}
        
        # 1. Token匹配率
        print("  计算Token匹配率...")
        token_match_rates = []
        for orig, pred in tqdm(zip(originals, predicteds), total=len(originals), desc="  Token匹配", leave=False):
            match_rate = calculate_token_match_rate(tokenizer, orig, pred)
            token_match_rates.append(match_rate)
        
        config_result['token_match_rate'] = {
            'mean': float(np.mean(token_match_rates)),
            'std': float(np.std(token_match_rates)),
            'scores': [float(x) for x in token_match_rates]
        }
        
        # 2. BERTScore
        if HAS_BERTSCORE:
            print("  计算BERTScore...", end=" ", flush=True)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                bertscore_result = calculate_bertscore(originals, predicteds)
            if bertscore_result:
                config_result['bertscore'] = bertscore_result
                print(f"完成 (F1={bertscore_result['f1']:.4f})")
            else:
                print("失败")
        
        # 3. ROUGE-L
        if HAS_ROUGE:
            print("  计算ROUGE-L...")
            rouge_result = calculate_rouge_l(originals, predicteds)
            if rouge_result:
                config_result['rouge_l'] = rouge_result
        
        # 4. 余弦相似度
        if HAS_SENTENCE_TRANSFORMERS and sentence_model:
            print("  计算余弦相似度...")
            cosine_result = calculate_cosine_similarity(originals, predicteds, sentence_model)
            if cosine_result:
                config_result['cosine_similarity'] = cosine_result
        
        analysis_results['configs'][config_name] = config_result
    
    # 计算总体统计
    print("\n" + "=" * 80)
    print("计算总体统计...")
    print("=" * 80)
    
    # 汇总所有配置的平均值
    all_token_match = []
    all_bertscore_f1 = []
    all_rouge_l = []
    all_cosine = []
    
    for config_name, config_result in analysis_results['configs'].items():
        if 'token_match_rate' in config_result:
            all_token_match.append(config_result['token_match_rate']['mean'])
        
        if 'bertscore' in config_result:
            all_bertscore_f1.append(config_result['bertscore']['f1'])
        
        if 'rouge_l' in config_result:
            all_rouge_l.append(config_result['rouge_l']['mean'])
        
        if 'cosine_similarity' in config_result:
            all_cosine.append(config_result['cosine_similarity']['mean'])
    
    analysis_results['overall_statistics'] = {}
    
    if all_token_match:
        analysis_results['overall_statistics']['token_match_rate'] = {
            'mean': float(np.mean(all_token_match)),
            'std': float(np.std(all_token_match))
        }
    
    if all_bertscore_f1:
        analysis_results['overall_statistics']['bertscore_f1'] = {
            'mean': float(np.mean(all_bertscore_f1)),
            'std': float(np.std(all_bertscore_f1))
        }
    
    if all_rouge_l:
        analysis_results['overall_statistics']['rouge_l'] = {
            'mean': float(np.mean(all_rouge_l)),
            'std': float(np.std(all_rouge_l))
        }
    
    if all_cosine:
        analysis_results['overall_statistics']['cosine_similarity'] = {
            'mean': float(np.mean(all_cosine)),
            'std': float(np.std(all_cosine))
        }
    
    # 保存结果
    if output_file is None:
        output_file = json_file.replace('.json', '_analysis.json')
    
    print(f"\n保存分析结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("分析结果摘要")
    print("=" * 80)
    
    print(f"\n总句子数: {analysis_results['total_sentences']}")
    print(f"总配置数: {analysis_results['total_configs']}")
    
    if 'overall_statistics' in analysis_results:
        print("\n总体平均指标:")
        stats = analysis_results['overall_statistics']
        
        if 'token_match_rate' in stats:
            print(f"  Token匹配率: {stats['token_match_rate']['mean']:.4f} ± {stats['token_match_rate']['std']:.4f}")
        
        if 'bertscore_f1' in stats:
            print(f"  BERTScore-F1: {stats['bertscore_f1']['mean']:.4f} ± {stats['bertscore_f1']['std']:.4f}")
        
        if 'rouge_l' in stats:
            print(f"  ROUGE-L: {stats['rouge_l']['mean']:.4f} ± {stats['rouge_l']['std']:.4f}")
        
        if 'cosine_similarity' in stats:
            print(f"  余弦相似度: {stats['cosine_similarity']['mean']:.4f} ± {stats['cosine_similarity']['std']:.4f}")
    
    print("\n各配置详细结果:")
    print("-" * 80)
    print(f"{'配置名称':<40} {'Token匹配率':<15} {'BERTScore-F1':<15} {'ROUGE-L':<15} {'余弦相似度':<15}")
    print("-" * 80)
    
    for config_name in sorted(analysis_results['configs'].keys()):
        config_result = analysis_results['configs'][config_name]
        
        token_match = "N/A"
        if 'token_match_rate' in config_result:
            token_match = f"{config_result['token_match_rate']['mean']:.4f}"
        
        bertscore_f1 = "N/A"
        if 'bertscore' in config_result:
            bertscore_f1 = f"{config_result['bertscore']['f1']:.4f}"
        
        rouge_l = "N/A"
        if 'rouge_l' in config_result:
            rouge_l = f"{config_result['rouge_l']['mean']:.4f}"
        
        cosine = "N/A"
        if 'cosine_similarity' in config_result:
            cosine = f"{config_result['cosine_similarity']['mean']:.4f}"
        
        print(f"{config_name:<40} {token_match:<15} {bertscore_f1:<15} {rouge_l:<15} {cosine:<15}")
    
    print("=" * 80)
    print(f"\n✅ 详细分析结果已保存到: {output_file}")
    print("=" * 80)
    
    return analysis_results

def main():
    parser = argparse.ArgumentParser(description="分析句子攻击结果")
    parser.add_argument("--input", type=str, default="sentence_attack_results.json",
                        help="输入JSON文件路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出JSON文件路径（默认：输入文件名_analysis.json）")
    
    args = parser.parse_args()
    
    analyze_sentence_results(args.input, args.output)

if __name__ == "__main__":
    main()

