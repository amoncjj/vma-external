#!/usr/bin/env python3
"""
评估KV Cache攻击结果
计算4个指标：
1. Token匹配率（先算单个句子，再算平均）
2. BERTScore
3. ROUGE-L
4. 余弦相似度
"""

import json
import glob
import os
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# 导入评估库
try:
    from bert_score import score as bert_score
    HAS_BERTSCORE = True
except ImportError:
    print("警告: bert_score未安装，将跳过BERTScore计算")
    print("安装命令: pip install bert-score")
    HAS_BERTSCORE = False

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    print("警告: rouge-score未安装，将跳过ROUGE-L计算")
    print("安装命令: pip install rouge-score")
    HAS_ROUGE = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    print("警告: sentence-transformers未安装，将跳过余弦相似度计算")
    print("安装命令: pip install sentence-transformers")
    HAS_SENTENCE_TRANSFORMERS = False

from transformers import AutoTokenizer

def calculate_token_match_rate(original, predicted, tokenizer):
    """
    计算token匹配率
    返回单个句子的匹配率
    """
    # 将句子tokenize
    orig_tokens = tokenizer.encode(original, add_special_tokens=False)
    pred_tokens = tokenizer.encode(predicted, add_special_tokens=False)
    
    # 计算匹配的token数量
    min_len = min(len(orig_tokens), len(pred_tokens))
    matched = sum(1 for i in range(min_len) if orig_tokens[i] == pred_tokens[i])
    
    # 匹配率 = 匹配的token数 / 原始token数
    if len(orig_tokens) == 0:
        return 0.0
    
    match_rate = matched / len(orig_tokens)
    return match_rate

def calculate_bertscore(originals, predicteds, lang='en'):
    """
    计算BERTScore
    """
    if not HAS_BERTSCORE:
        return None
    
    try:
        # BERTScore参数：predictions在前，references在后
        # 检查是否有GPU
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        P, R, F1 = bert_score(predicteds, originals, lang=lang, verbose=False, device=device)
        # 转换为Python原生类型
        return {
            'precision': float(P.mean().item()),
            'recall': float(R.mean().item()),
            'f1': float(F1.mean().item()),
            'precision_scores': [float(x) for x in P.tolist()],
            'recall_scores': [float(x) for x in R.tolist()],
            'f1_scores': [float(x) for x in F1.tolist()]
        }
    except Exception as e:
        print(f"  BERTScore计算错误: {e}")
        return None

def calculate_rouge_l(originals, predicteds):
    """
    计算ROUGE-L
    """
    if not HAS_ROUGE:
        return None
    
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_l_scores = []
        
        for orig, pred in zip(originals, predicteds):
            scores = scorer.score(orig, pred)
            rouge_l_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'mean': float(np.mean(rouge_l_scores)),
            'std': float(np.std(rouge_l_scores)),
            'scores': [float(x) for x in rouge_l_scores]
        }
    except Exception as e:
        print(f"ROUGE-L计算错误: {e}")
        return None

def calculate_cosine_similarity(originals, predicteds):
    """
    计算余弦相似度
    使用sentence-transformers获取句子嵌入，然后计算余弦相似度
    """
    if not HAS_SENTENCE_TRANSFORMERS:
        return None
    
    try:
        # 使用多语言模型
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # 获取嵌入
        orig_embeddings = model.encode(originals, show_progress_bar=False)
        pred_embeddings = model.encode(predicteds, show_progress_bar=False)
        
        # 计算余弦相似度
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = []
        for orig_emb, pred_emb in zip(orig_embeddings, pred_embeddings):
            sim = cosine_similarity([orig_emb], [pred_emb])[0][0]
            similarities.append(sim)
        
        return {
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'scores': [float(x) for x in similarities]
        }
    except Exception as e:
        print(f"余弦相似度计算错误: {e}")
        return None

def load_sentences_file(filepath):
    """加载句子对文件，支持两种格式：
    1. _sentences.json 格式：直接包含sentences数组
    2. 主结果文件格式：包含layers数组，每个layer包含samples数组
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查是否是_sentences.json格式
    if 'sentences' in data:
        # 新格式：_sentences.json
        sentences = data.get('sentences', [])
        originals = [s['original'] for s in sentences]
        predicteds = [s['predicted'] for s in sentences]
        layer = data.get('layer', 'unknown')
    elif 'layers' in data:
        # 旧格式：主结果文件，需要从layers中提取
        # 只处理第一个layer（因为旧格式通常只有一个layer）
        if len(data['layers']) == 0:
            return None
        layer_data = data['layers'][0]
        samples = layer_data.get('samples', [])
        originals = [s['original_text'] for s in samples]
        predicteds = [s['predicted_text'] for s in samples]
        layer = layer_data.get('layer', 'unknown')
    else:
        return None
    
    return {
        'model_name': data.get('model_name', 'unknown'),
        'perm_type': data.get('perm_type', 'unknown'),
        'layer': layer,
        'originals': originals,
        'predicteds': predicteds,
        'num_samples': len(originals)
    }

def evaluate_file(filepath, tokenizer):
    """评估单个文件"""
    print(f"\n处理文件: {os.path.basename(filepath)}")
    
    # 加载数据
    data = load_sentences_file(filepath)
    originals = data['originals']
    predicteds = data['predicteds']
    
    if len(originals) == 0:
        print("  警告: 文件为空，跳过")
        return None
    
    results = {
        'file': os.path.basename(filepath),
        'model_name': data['model_name'],
        'perm_type': data['perm_type'],
        'layer': data['layer'],
        'num_samples': data['num_samples']
    }
    
    # 1. Token匹配率
    print("  计算Token匹配率...")
    token_match_rates = []
    for orig, pred in tqdm(zip(originals, predicteds), total=len(originals), desc="  Token匹配", leave=False):
        match_rate = calculate_token_match_rate(orig, pred, tokenizer)
        token_match_rates.append(match_rate)
    
    results['token_match_rate'] = {
        'mean': float(np.mean(token_match_rates)),
        'std': float(np.std(token_match_rates)),
        'scores': [float(x) for x in token_match_rates]
    }
    
    # 2. BERTScore
    if HAS_BERTSCORE:
        print("  计算BERTScore...")
        bertscore_result = calculate_bertscore(originals, predicteds)
        if bertscore_result:
            results['bertscore'] = bertscore_result
    
    # 3. ROUGE-L
    if HAS_ROUGE:
        print("  计算ROUGE-L...")
        rouge_result = calculate_rouge_l(originals, predicteds)
        if rouge_result:
            results['rouge_l'] = {
                'mean': rouge_result['mean'],
                'std': rouge_result['std']
            }
    
    # 4. 余弦相似度
    if HAS_SENTENCE_TRANSFORMERS:
        print("  计算余弦相似度...")
        cosine_result = calculate_cosine_similarity(originals, predicteds)
        if cosine_result:
            results['cosine_similarity'] = {
                'mean': cosine_result['mean'],
                'std': cosine_result['std']
            }
    
    return results

def main():
    # 查找所有句子对文件和主结果文件
    sentence_files = sorted(glob.glob("*_sentences.json"))
    
    # 也查找主结果文件（旧格式，如 kv_attack_results_llama3.2-1B_no_perm.json）
    # 但排除已经包含layer信息的文件（新格式）
    main_result_files = []
    for f in sorted(glob.glob("kv_attack_results_*.json")):
        # 排除_sentences.json文件和已经包含layer信息的文件
        if not f.endswith("_sentences.json") and "_layer" not in f:
            main_result_files.append(f)
    
    # 合并文件列表
    all_files = sentence_files + main_result_files
    
    if len(all_files) == 0:
        print("错误: 未找到任何结果文件")
        return
    
    print(f"找到 {len(sentence_files)} 个句子对文件")
    if main_result_files:
        print(f"找到 {len(main_result_files)} 个主结果文件（旧格式）")
    print(f"总共 {len(all_files)} 个文件")
    print("=" * 80)
    
    # 加载tokenizer（使用一个通用的tokenizer来计算token匹配率）
    # 使用bert-base-uncased作为通用tokenizer
    print("加载tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        print("✓ 成功加载tokenizer")
    except Exception as e:
        print(f"错误: 无法加载tokenizer: {e}")
        print("请确保已安装transformers: pip install transformers")
        return
    
    # 评估所有文件
    all_results = []
    for filepath in tqdm(all_files, desc="处理文件"):
        result = evaluate_file(filepath, tokenizer)
        if result:
            all_results.append(result)
    
    # 保存结果
    output_file = "attack_evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 打印汇总
    print("\n" + "=" * 80)
    print("评估结果汇总")
    print("=" * 80)
    
    # 按模型和置换类型分组
    grouped_results = defaultdict(list)
    for result in all_results:
        key = f"{result['model_name']}_{result['perm_type']}_layer{result['layer']}"
        grouped_results[key].append(result)
    
    # 打印每个文件的指标
    print("\n详细结果:")
    print("-" * 80)
    print(f"{'文件':<50} {'Token匹配率':<15} {'BERTScore-F1':<15} {'ROUGE-L':<15} {'余弦相似度':<15}")
    print("-" * 80)
    
    for result in all_results:
        file_name = result['file'][:48]
        token_match = f"{result['token_match_rate']['mean']:.4f}"
        
        bertscore_f1 = "N/A"
        if 'bertscore' in result:
            bertscore_f1 = f"{result['bertscore']['f1']:.4f}"
        
        rouge_l = "N/A"
        if 'rouge_l' in result:
            rouge_l = f"{result['rouge_l']['mean']:.4f}"
        
        cosine = "N/A"
        if 'cosine_similarity' in result:
            cosine = f"{result['cosine_similarity']['mean']:.4f}"
        
        print(f"{file_name:<50} {token_match:<15} {bertscore_f1:<15} {rouge_l:<15} {cosine:<15}")
    
    # 计算总体平均
    print("\n" + "-" * 80)
    print("总体平均:")
    print("-" * 80)
    
    all_token_match = [r['token_match_rate']['mean'] for r in all_results]
    print(f"Token匹配率: {np.mean(all_token_match):.4f} ± {np.std(all_token_match):.4f}")
    
    all_bertscore = [r['bertscore']['f1'] for r in all_results if 'bertscore' in r]
    if all_bertscore:
        print(f"BERTScore-F1: {np.mean(all_bertscore):.4f} ± {np.std(all_bertscore):.4f}")
    
    all_rouge = [r['rouge_l']['mean'] for r in all_results if 'rouge_l' in r]
    if all_rouge:
        print(f"ROUGE-L: {np.mean(all_rouge):.4f} ± {np.std(all_rouge):.4f}")
    
    all_cosine = [r['cosine_similarity']['mean'] for r in all_results if 'cosine_similarity' in r]
    if all_cosine:
        print(f"余弦相似度: {np.mean(all_cosine):.4f} ± {np.std(all_cosine):.4f}")
    
    print(f"\n✅ 详细结果已保存到: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
