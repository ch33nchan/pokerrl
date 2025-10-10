# Deep CFR Architecture Study - Comprehensive Analysis Report
============================================================

## Executive Summary
- Total experiments: 94
- Successful experiments: 94
- Success rate: 100.0%

## Performance Rankings (by mean exploitability)

1. **Tabular Cfr Baseline**
   - Mean: 71.375 ± 14.007 mBB/100
   - 95% CI: [62.983, 79.768] mBB/100
   - Samples: 14

2. **Deep Cfr Wide**
   - Mean: 339.957 ± 21.078 mBB/100
   - 95% CI: [329.836, 350.078] mBB/100
   - Samples: 20

3. **Deep Cfr Baseline**
   - Mean: 398.779 ± 26.133 mBB/100
   - 95% CI: [386.231, 411.328] mBB/100
   - Samples: 20

4. **Deep Cfr Deep**
   - Mean: 404.355 ± 24.383 mBB/100
   - 95% CI: [392.647, 416.063] mBB/100
   - Samples: 20

5. **Deep Cfr Fast**
   - Mean: 436.191 ± 15.813 mBB/100
   - 95% CI: [428.598, 443.784] mBB/100
   - Samples: 20

## Key Findings

1. **Performance Gap**: Tabular Cfr Baseline outperforms Deep Cfr Fast by 83.6%

2. **Tabular vs Deep CFR**: Tabular CFR outperforms best Deep CFR by 376.3%

3. **Architecture Impact**: Wide architecture outperforms deep architecture by 14.8%

## Parameter Efficiency Analysis

**Most Parameter Efficient**: Deep Cfr Wide
- Performance: 339.957 mBB/100

## Statistical Significance

Statistically significant differences (95% CI non-overlapping):
- Tabular Cfr Baseline vs Deep Cfr Baseline: 82.1% improvement
- Tabular Cfr Baseline vs Deep Cfr Wide: 79.0% improvement
- Tabular Cfr Baseline vs Deep Cfr Deep: 82.3% improvement
- Tabular Cfr Baseline vs Deep Cfr Fast: 83.6% improvement
- Deep Cfr Baseline vs Deep Cfr Wide: -17.3% improvement

## Training Efficiency

**Fastest Training**: Tabular Cfr Baseline
- Mean time: 0.11 seconds
- Performance: 71.375 mBB/100

## Recommendations

2. **For Deep CFR**: Use Wide architecture when neural approximation is required

3. **For Resource-Constrained Environments**: Consider Fast architecture for quick training

## Limitations

- Study limited to Kuhn Poker (3-card game)
- 500 training iterations per experiment
- External sampling traversal only
