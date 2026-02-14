#!/usr/bin/env python3
"""
Convolution Benchmark Visualization
Compares C implementation vs Assembly (SIMD) implementation
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

# Read benchmark data
try:
    df = pd.read_csv('plot/benchmark_results.csv')
except FileNotFoundError:
    print("Error: benchmark_results.csv not found!")
    print("Please run the C benchmark program first to generate the data.")
    exit(1)

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# ============================================================================
# Plot 1: Execution Time Comparison (Line Plot)
# ============================================================================
ax1 = plt.subplot(3, 3, 1)
ax1.plot(df['iterations'], df['c_time'], 'o-', linewidth=2, markersize=8, 
         label='C Implementation', color='#e74c3c')
ax1.plot(df['iterations'], df['asm_time'], 's-', linewidth=2, markersize=8, 
         label='Assembly (SIMD)', color='#2ecc71')
ax1.set_xlabel('Number of Iterations', fontsize=11, fontweight='bold')
ax1.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
ax1.set_title('Execution Time vs Iterations', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# ============================================================================
# Plot 2: Execution Time Comparison (Bar Chart)
# ============================================================================
ax2 = plt.subplot(3, 3, 2)
x = np.arange(len(df))
width = 0.35
bars1 = ax2.bar(x - width/2, df['c_time'], width, label='C Implementation', 
                color='#e74c3c', alpha=0.8)
bars2 = ax2.bar(x + width/2, df['asm_time'], width, label='Assembly (SIMD)', 
                color='#2ecc71', alpha=0.8)
ax2.set_xlabel('Test Number', fontsize=11, fontweight='bold')
ax2.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
ax2.set_title('Execution Time Comparison (Bar Chart)', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f"T{i+1}" for i in range(len(df))])
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    if height > 0.01:
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}s', ha='center', va='bottom', fontsize=7)

# ============================================================================
# Plot 3: Speedup Factor
# ============================================================================
ax3 = plt.subplot(3, 3, 3)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
bars = ax3.bar(range(len(df)), df['speedup'], color=colors, alpha=0.8, edgecolor='black')
ax3.axhline(y=1, color='red', linestyle='--', linewidth=2, label='No speedup (1x)')
ax3.set_xlabel('Test Number', fontsize=11, fontweight='bold')
ax3.set_ylabel('Speedup Factor', fontsize=11, fontweight='bold')
ax3.set_title('Assembly Speedup over C', fontsize=13, fontweight='bold')
ax3.set_xticks(range(len(df)))
ax3.set_xticklabels([f"{int(df.iloc[i]['iterations'])}" for i in range(len(df))])
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Calculate average speedup
avg_speedup = df['speedup'].mean()
ax3.text(0.98, 0.98, f'Avg Speedup: {avg_speedup:.2f}x', 
         transform=ax3.transAxes, fontsize=11, fontweight='bold',
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# ============================================================================
# Plot 4: Time Saved (Absolute)
# ============================================================================
ax4 = plt.subplot(3, 3, 4)
time_saved = df['c_time'] - df['asm_time']
ax4.fill_between(df['iterations'], 0, time_saved, alpha=0.5, color='#3498db')
ax4.plot(df['iterations'], time_saved, 'o-', linewidth=2, markersize=8, color='#2c3e50')
ax4.set_xlabel('Number of Iterations', fontsize=11, fontweight='bold')
ax4.set_ylabel('Time Saved (seconds)', fontsize=11, fontweight='bold')
ax4.set_title('Absolute Time Savings (C - Assembly)', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Add annotations for max savings
max_saved_idx = time_saved.idxmax()
max_saved = time_saved[max_saved_idx]
max_iter = df.iloc[max_saved_idx]['iterations']
ax4.annotate(f'Max: {max_saved:.3f}s\n@{int(max_iter)} iter',
            xy=(max_iter, max_saved), xytext=(max_iter*0.7, max_saved*0.7),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# ============================================================================
# Plot 5: Percentage Time Saved
# ============================================================================
ax5 = plt.subplot(3, 3, 5)
percent_saved = ((df['c_time'] - df['asm_time']) / df['c_time']) * 100
ax5.bar(range(len(df)), percent_saved, color='#9b59b6', alpha=0.7, edgecolor='black')
ax5.set_xlabel('Number of Iterations', fontsize=11, fontweight='bold')
ax5.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
ax5.set_title('Percentage Time Saved', fontsize=13, fontweight='bold')
ax5.set_xticks(range(len(df)))
ax5.set_xticklabels([f"{int(df.iloc[i]['iterations'])}" for i in range(len(df))])
ax5.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, pct in enumerate(percent_saved):
    ax5.text(i, pct, f'{pct:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

avg_pct = percent_saved.mean()
ax5.axhline(y=avg_pct, color='red', linestyle='--', linewidth=2, 
           label=f'Average: {avg_pct:.1f}%')
ax5.legend(fontsize=10)

# ============================================================================
# Plot 6: Throughput (Iterations per Second)
# ============================================================================
ax6 = plt.subplot(3, 3, 6)
throughput_c = df['iterations'] / df['c_time']
throughput_asm = df['iterations'] / df['asm_time']

ax6.plot(df['iterations'], throughput_c, 'o-', linewidth=2, markersize=8, 
         label='C Implementation', color='#e74c3c')
ax6.plot(df['iterations'], throughput_asm, 's-', linewidth=2, markersize=8, 
         label='Assembly (SIMD)', color='#2ecc71')
ax6.set_xlabel('Number of Iterations', fontsize=11, fontweight='bold')
ax6.set_ylabel('Throughput (iterations/sec)', fontsize=11, fontweight='bold')
ax6.set_title('Processing Throughput', fontsize=13, fontweight='bold')
ax6.legend(loc='best', fontsize=10)
ax6.grid(True, alpha=0.3)

# ============================================================================
# Plot 7: Log Scale Comparison
# ============================================================================
ax7 = plt.subplot(3, 3, 7)
ax7.semilogy(df['iterations'], df['c_time'], 'o-', linewidth=2, markersize=8, 
            label='C Implementation', color='#e74c3c')
ax7.semilogy(df['iterations'], df['asm_time'], 's-', linewidth=2, markersize=8, 
            label='Assembly (SIMD)', color='#2ecc71')
ax7.set_xlabel('Number of Iterations', fontsize=11, fontweight='bold')
ax7.set_ylabel('Execution Time (seconds, log scale)', fontsize=11, fontweight='bold')
ax7.set_title('Execution Time (Logarithmic Scale)', fontsize=13, fontweight='bold')
ax7.legend(loc='upper left', fontsize=10)
ax7.grid(True, alpha=0.3, which='both')

# ============================================================================
# Plot 8: Efficiency Score
# ============================================================================
ax8 = plt.subplot(3, 3, 8)
efficiency = (1 - df['asm_time'] / df['c_time']) * 100
ax8.plot(df['iterations'], efficiency, 'o-', linewidth=2.5, markersize=10, 
        color='#16a085', markerfacecolor='#1abc9c', markeredgewidth=2, markeredgecolor='#0e6655')
ax8.fill_between(df['iterations'], 0, efficiency, alpha=0.3, color='#16a085')
ax8.set_xlabel('Number of Iterations', fontsize=11, fontweight='bold')
ax8.set_ylabel('Efficiency (%)', fontsize=11, fontweight='bold')
ax8.set_title('SIMD Optimization Efficiency', fontsize=13, fontweight='bold')
ax8.grid(True, alpha=0.3)
ax8.set_ylim([0, 100])

# Add horizontal lines for reference
ax8.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% efficient')
ax8.axhline(y=75, color='green', linestyle='--', alpha=0.5, label='75% efficient')
ax8.legend(fontsize=9)

# ============================================================================
# Plot 9: Statistics Table
# ============================================================================
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

stats_data = [
    ['Metric', 'Value'],
    ['─' * 30, '─' * 15],
    ['Average C Time', f"{df['c_time'].mean():.4f} sec"],
    ['Average ASM Time', f"{df['asm_time'].mean():.4f} sec"],
    ['Average Speedup', f"{avg_speedup:.2f}x"],
    ['Max Speedup', f"{df['speedup'].max():.2f}x"],
    ['Min Speedup', f"{df['speedup'].min():.2f}x"],
    ['Total Time Saved', f"{(df['c_time'].sum() - df['asm_time'].sum()):.4f} sec"],
    ['Avg % Improvement', f"{avg_pct:.2f}%"],
    ['Total Tests', f"{len(df)}"],
]

table = ax9.table(cellText=stats_data, cellLoc='left', loc='center',
                 colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style the header
for i in range(2):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style the separator
for i in range(2):
    table[(1, i)].set_facecolor('#ecf0f1')

# Alternate row colors
for i in range(2, len(stats_data)):
    for j in range(2):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')
        else:
            table[(i, j)].set_facecolor('white')

ax9.set_title('Performance Statistics Summary', fontsize=13, fontweight='bold', pad=20)

# ============================================================================
# Add main title
# ============================================================================
fig.suptitle('Convolution Performance Analysis: C vs Assembly (SIMD)', 
             fontsize=16, fontweight='bold', y=0.995)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save the figure
plt.savefig('plot/benchmark_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Saved: benchmark_visualization.png")

print("\n" + "="*60)
print("BENCHMARK SUMMARY")
print("="*60)
print(f"Average Speedup:     {avg_speedup:.2f}x")
print(f"Best Speedup:        {df['speedup'].max():.2f}x (at {int(df.iloc[df['speedup'].idxmax()]['iterations'])} iterations)")
print(f"Total Time Saved:    {(df['c_time'].sum() - df['asm_time'].sum()):.4f} seconds")
print(f"Average % Saved:     {avg_pct:.2f}%")
print(f"SIMD Efficiency:     {(avg_speedup/4.0)*100:.1f}% of theoretical max (4x)")
print("="*60)
print("\nAll visualizations saved successfully!")
print("  - benchmark_visualization.png (main dashboard)")
