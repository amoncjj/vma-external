#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
筛选在所有18种情况下score都 >= 0.95 的句子
"""

import json

def filter_high_score_sentences(json_file, threshold=0.95):
    """
    筛选在所有情况下score都 >= threshold的句子
    
    Args:
        json_file: JSON文件路径
        threshold: 分数阈值，默认0.95
    
    Returns:
        符合条件的句子索引列表
    """
    # 加载JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        all_results = json.load(f)
    
    print(f"总共找到 {len(all_results)} 个评估结果")
    
    # 确认每个结果都有100个句子
    num_sentences = all_results[0]['num_samples']
    print(f"每个结果包含 {num_sentences} 个句子")
    
    # 对于每个句子索引，检查在所有情况下score是否都 >= threshold
    qualified_sentences = []
    
    for sentence_idx in range(num_sentences):
        all_qualified = True
        scores_for_this_sentence = []
        
        for result in all_results:
            if 'token_match_rate' in result and 'scores' in result['token_match_rate']:
                score = result['token_match_rate']['scores'][sentence_idx]
                scores_for_this_sentence.append(score)
                if score < threshold:
                    all_qualified = False
                    break
            else:
                all_qualified = False
                break
        
        if all_qualified:
            qualified_sentences.append({
                'index': sentence_idx,
                'scores': scores_for_this_sentence,
                'min_score': min(scores_for_this_sentence),
                'max_score': max(scores_for_this_sentence),
                'mean_score': sum(scores_for_this_sentence) / len(scores_for_this_sentence)
            })
    
    return qualified_sentences, all_results

def main():
    json_file = 'attack_evaluation_results.json'
    threshold = 0.95
    
    print("=" * 80)
    print(f"筛选在所有情况下 score >= {threshold} 的句子")
    print("=" * 80)
    
    qualified_sentences, all_results = filter_high_score_sentences(json_file, threshold)
    
    print(f"\n找到 {len(qualified_sentences)} 个符合条件的句子（在所有18种情况下score都 >= {threshold}）")
    print("\n符合条件的句子索引和分数统计：")
    print("-" * 80)
    print(f"{'索引':<8} {'最小分数':<12} {'最大分数':<12} {'平均分数':<12} {'所有分数'}")
    print("-" * 80)
    
    for item in qualified_sentences:
        scores_str = ', '.join([f"{s:.4f}" for s in item['scores']])
        print(f"{item['index']:<8} {item['min_score']:<12.4f} {item['max_score']:<12.4f} {item['mean_score']:<12.4f} [{scores_str}]")
    
    # 保存结果到文件
    output_file = 'high_score_sentences.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'threshold': threshold,
            'total_qualified': len(qualified_sentences),
            'qualified_sentence_indices': [item['index'] for item in qualified_sentences],
            'detailed_results': qualified_sentences
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存到: {output_file}")
    print("=" * 80)
    
    # 打印每个评估结果的名称
    print("\n18种评估情况：")
    print("-" * 80)
    for i, result in enumerate(all_results, 1):
        print(f"{i:2d}. {result['file']} (模型: {result['model_name']}, 置换: {result['perm_type']}, 层: {result['layer']})")
    print("=" * 80)

if __name__ == "__main__":
    main()

