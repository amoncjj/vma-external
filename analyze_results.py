"""
分析 SCX 攻击结果
读取 scx_attack_results.json 并生成统计报告
"""

import json
import sys
from collections import defaultdict

def load_results(filename="scx_attack_results.json"):
    """加载结果文件"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到结果文件 '{filename}'")
        print("请先运行攻击: python vocab_matching_attack_scx.py")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"错误: '{filename}' 不是有效的 JSON 文件")
        sys.exit(1)

def analyze_layer_results(layer_data):
    """分析单层结果"""
    layer = layer_data['layer']
    stats = layer_data['statistics']
    samples = layer_data['samples']
    
    # 基本统计
    print(f"\n{'='*80}")
    print(f"Layer {layer} 详细分析")
    print(f"{'='*80}")
    
    print(f"\n基本统计:")
    print(f"  总样本数: {stats['total_samples']}")
    print(f"  成功数: {stats['successful']}")
    print(f"  失败数: {stats['total_samples'] - stats['successful']}")
    print(f"  成功率: {stats['success_rate']*100:.2f}%")
    
    # Token 长度分析
    token_lengths = [s.get('num_tokens', 0) for s in samples if 'num_tokens' in s]
    if token_lengths:
        print(f"\nToken 长度统计:")
        print(f"  平均长度: {sum(token_lengths)/len(token_lengths):.1f}")
        print(f"  最短: {min(token_lengths)}")
        print(f"  最长: {max(token_lengths)}")
    
    # 失败样本分析
    failed_samples = [s for s in samples if not s['success']]
    if failed_samples:
        print(f"\n失败样本分析 (共 {len(failed_samples)} 个):")
        print(f"  前5个失败样本:")
        for i, sample in enumerate(failed_samples[:5]):
            print(f"\n  [{i+1}] 样本 #{sample['sample_idx']}")
            if 'error' in sample:
                print(f"      错误: {sample['error']}")
            else:
                print(f"      原始: {sample['original'][:80]}...")
                if sample['predicted']:
                    print(f"      预测: {sample['predicted'][:80]}...")
    
    # 成功样本示例
    success_samples = [s for s in samples if s['success']]
    if success_samples:
        print(f"\n成功样本示例 (共 {len(success_samples)} 个):")
        for i, sample in enumerate(success_samples[:3]):
            print(f"\n  [{i+1}] 样本 #{sample['sample_idx']}")
            print(f"      文本: {sample['original'][:80]}...")
            print(f"      Token数: {sample.get('num_tokens', 'N/A')}")

def analyze_cross_layer(results):
    """跨层分析"""
    print(f"\n{'='*80}")
    print("跨层分析")
    print(f"{'='*80}")
    
    layers = [r['layer'] for r in results]
    success_rates = [r['statistics']['success_rate'] for r in results]
    
    print(f"\n各层成功率对比:")
    print(f"  {'层':<10} {'成功率':<15} {'可视化'}")
    print(f"  {'-'*10} {'-'*15} {'-'*50}")
    
    for layer, rate in zip(layers, success_rates):
        bar = '█' * int(rate * 50)
        print(f"  Layer {layer:<4} {rate*100:>6.2f}%        {bar}")
    
    print(f"\n总体统计:")
    print(f"  平均成功率: {sum(success_rates)/len(success_rates)*100:.2f}%")
    print(f"  最高成功率: {max(success_rates)*100:.2f}% (Layer {layers[success_rates.index(max(success_rates))]})")
    print(f"  最低成功率: {min(success_rates)*100:.2f}% (Layer {layers[success_rates.index(min(success_rates))]})")
    
    # 层深度趋势
    print(f"\n层深度趋势分析:")
    if success_rates[0] > success_rates[-1]:
        print(f"  ✓ 浅层 (Layer {layers[0]}) 成功率更高")
    elif success_rates[0] < success_rates[-1]:
        print(f"  ✓ 深层 (Layer {layers[-1]}) 成功率更高")
    else:
        print(f"  - 各层成功率相近")

def analyze_token_length_impact(results):
    """分析 token 长度对成功率的影响"""
    print(f"\n{'='*80}")
    print("Token 长度影响分析")
    print(f"{'='*80}")
    
    # 按长度分组
    length_buckets = defaultdict(lambda: {'total': 0, 'success': 0})
    
    for layer_data in results:
        for sample in layer_data['samples']:
            if 'num_tokens' in sample:
                length = sample['num_tokens']
                # 分组: 1-10, 11-20, 21-30, 31-40, 41-50
                bucket = ((length - 1) // 10) * 10 + 1
                length_buckets[bucket]['total'] += 1
                if sample['success']:
                    length_buckets[bucket]['success'] += 1
    
    if length_buckets:
        print(f"\n按 Token 长度范围统计:")
        print(f"  {'长度范围':<15} {'样本数':<10} {'成功率':<15} {'可视化'}")
        print(f"  {'-'*15} {'-'*10} {'-'*15} {'-'*40}")
        
        for bucket in sorted(length_buckets.keys()):
            data = length_buckets[bucket]
            rate = data['success'] / data['total'] if data['total'] > 0 else 0
            bar = '█' * int(rate * 40)
            print(f"  {bucket:>2}-{bucket+9:<10} {data['total']:>6}     {rate*100:>6.2f}%        {bar}")

def generate_summary(results):
    """生成总体摘要"""
    print(f"\n{'='*80}")
    print("攻击总体摘要")
    print(f"{'='*80}")
    
    total_samples = sum(r['statistics']['total_samples'] for r in results)
    total_success = sum(r['statistics']['successful'] for r in results)
    overall_rate = total_success / total_samples if total_samples > 0 else 0
    
    print(f"\n整体性能:")
    print(f"  攻击层数: {len(results)}")
    print(f"  总测试样本: {total_samples}")
    print(f"  总成功数: {total_success}")
    print(f"  总失败数: {total_samples - total_success}")
    print(f"  总体成功率: {overall_rate*100:.2f}%")
    
    # 攻击效果评估
    print(f"\n攻击效果评估:")
    if overall_rate >= 0.9:
        print(f"  ✓✓✓ 优秀 - 攻击非常有效")
    elif overall_rate >= 0.7:
        print(f"  ✓✓ 良好 - 攻击较为有效")
    elif overall_rate >= 0.5:
        print(f"  ✓ 中等 - 攻击部分有效")
    elif overall_rate >= 0.3:
        print(f"  - 较弱 - 攻击效果有限")
    else:
        print(f"  ✗ 很弱 - 攻击基本无效")
    
    # 建议
    print(f"\n改进建议:")
    if overall_rate < 0.5:
        print(f"  • 尝试增加 matching_eps 阈值")
        print(f"  • 确保 next_token_proposal 和 use_cache 已启用")
        print(f"  • 考虑使用不同的匹配函数")
    elif overall_rate < 0.8:
        print(f"  • 可以尝试微调 matching_eps")
        print(f"  • 考虑增加 batch_sz 提高搜索精度")
    else:
        print(f"  • 当前配置已经很好")
        print(f"  • 可以尝试更长的样本或更多层")

def export_csv(results, filename="scx_attack_results.csv"):
    """导出为 CSV 格式"""
    try:
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Layer', 'Sample_ID', 'Success', 'Num_Tokens', 'Original', 'Predicted'])
            
            for layer_data in results:
                layer = layer_data['layer']
                for sample in layer_data['samples']:
                    writer.writerow([
                        layer,
                        sample['sample_idx'],
                        sample['success'],
                        sample.get('num_tokens', ''),
                        sample['original'][:100],
                        sample.get('predicted', '')[:100] if sample.get('predicted') else ''
                    ])
        
        print(f"\n✓ 结果已导出到: {filename}")
        return True
    except Exception as e:
        print(f"\n✗ 导出 CSV 失败: {e}")
        return False

def main():
    """主函数"""
    print("="*80)
    print("SCX 攻击结果分析工具")
    print("="*80)
    
    # 加载结果
    results = load_results()
    
    if not results:
        print("\n错误: 结果文件为空")
        return
    
    print(f"\n✓ 成功加载 {len(results)} 层的攻击结果")
    
    # 生成各项分析
    for layer_data in results:
        analyze_layer_results(layer_data)
    
    if len(results) > 1:
        analyze_cross_layer(results)
    
    analyze_token_length_impact(results)
    generate_summary(results)
    
    # 询问是否导出 CSV
    print(f"\n{'='*80}")
    try:
        response = input("\n是否导出为 CSV 格式? (y/n): ").strip().lower()
        if response == 'y':
            export_csv(results)
    except KeyboardInterrupt:
        print("\n\n分析完成。")
    
    print(f"\n{'='*80}")
    print("分析完成！")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

