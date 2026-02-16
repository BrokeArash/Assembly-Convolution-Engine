#!/usr/bin/env python3
"""
Pattern Recognition Analysis & Visualization Tool
Analyzes similarity scores and visualizes pattern detection results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import subprocess
import json
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

class PatternAnalyzer:
    """Analyzes pattern recognition results and creates visualizations"""
    
    def __init__(self, dataset_path='dataset', output_path='images'):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.dataset_size = 100
        self.results = []
        
    def extract_features_python(self, image_path):
        """
        Extract vertical and horizontal edge features from image
        (Python equivalent of the C code for analysis)
        """
        from PIL import Image, ImageFilter
        import numpy as np
        
        # Load and convert to grayscale
        try:
            img = Image.open(image_path).convert('L')
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            print("Trying to load as different format...")
            # Try loading with different methods
            try:
                img = Image.open(image_path).convert('RGB').convert('L')
            except:
                raise ValueError(f"Cannot load image: {image_path}")
        
        img_array = np.array(img, dtype=np.float32)
        
        # Sobel filters
        K_VERT = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)
        
        K_HORI = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=np.float32)
        
        # Apply filters (simplified convolution)
        from scipy.ndimage import convolve
        
        v_response = convolve(img_array, K_VERT)
        h_response = convolve(img_array, K_HORI)
        
        # Calculate densities
        v_density = np.sum(np.abs(v_response)) / img_array.size
        h_density = np.sum(np.abs(h_response)) / img_array.size
        
        return v_density, h_density
    
    def load_dataset_features(self):
        """Load features for all dataset images"""
        print("Loading dataset features...")
        database = []
        
        for i in range(self.dataset_size):
            filepath = f"{self.dataset_path}/test_{i}.jpg"
            
            if os.path.exists(filepath):
                try:
                    v_density, h_density = self.extract_features_python(filepath)
                    label = 0 if i % 2 == 0 else 1  # Even=Vertical, Odd=Horizontal
                    
                    database.append({
                        'id': i,
                        'v_density': v_density,
                        'h_density': h_density,
                        'label': label,
                        'label_str': 'VERTICAL' if label == 0 else 'HORIZONTAL',
                        'filepath': filepath
                    })
                except Exception as e:
                    print(f"  Warning: Could not load test_{i}.jpg: {e}")
            
            if (i + 1) % 20 == 0:
                print(f"  Loaded {i + 1}/{self.dataset_size} images...")
        
        print(f"✓ Loaded {len(database)} images from dataset\n")
        return database
    
    def calculate_similarity_scores(self, input_path, database):
        """Calculate similarity between input and all database images"""
        print(f"Analyzing input image: {input_path}")
        
        # Extract input features
        input_v, input_h = self.extract_features_python(input_path)
        print(f"  Input features: [V: {input_v:.2f}, H: {input_h:.2f}]")
        
        # Calculate distances to all database images
        results = []
        for entry in database:
            diff_v = input_v - entry['v_density']
            diff_h = input_h - entry['h_density']
            distance = np.sqrt(diff_v**2 + diff_h**2)
            
            results.append({
                'id': entry['id'],
                'label': entry['label'],
                'label_str': entry['label_str'],
                'v_density': entry['v_density'],
                'h_density': entry['h_density'],
                'distance': distance,
                'similarity_score': 100 / (1 + distance),  # Convert to 0-100 score
                'filepath': entry['filepath']
            })
        
        # Sort by distance (closest first)
        results.sort(key=lambda x: x['distance'])
        
        # Add rankings
        for rank, result in enumerate(results, 1):
            result['rank'] = rank
        
        self.input_features = (input_v, input_h)
        self.results = results
        
        return results
    
    def generate_report(self, output_file='analysis_report.txt'):
        """Generate detailed text report"""
        if not self.results:
            print("No results to report. Run calculate_similarity_scores first.")
            return
        
        report = []
        report.append("=" * 80)
        report.append("PATTERN RECOGNITION ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Input features
        report.append("INPUT IMAGE FEATURES:")
        report.append("-" * 80)
        report.append(f"Vertical Density:   {self.input_features[0]:.4f}")
        report.append(f"Horizontal Density: {self.input_features[1]:.4f}")
        report.append(f"Dominant Pattern:   {'VERTICAL' if self.input_features[0] > self.input_features[1] else 'HORIZONTAL'}")
        report.append("")
        
        # Best match
        best = self.results[0]
        report.append("BEST MATCH:")
        report.append("-" * 80)
        report.append(f"Dataset Image:      test_{best['id']}.jpg")
        report.append(f"Pattern Type:       {best['label_str']}")
        report.append(f"Similarity Score:   {best['similarity_score']:.2f}%")
        report.append(f"Distance:           {best['distance']:.4f}")
        report.append("")
        
        # Top 10 matches
        report.append("TOP 10 MOST SIMILAR IMAGES:")
        report.append("-" * 80)
        report.append(f"{'Rank':<6} {'Image':<15} {'Pattern':<12} {'Similarity':<12} {'Distance':<10}")
        report.append("-" * 80)
        
        for result in self.results[:10]:
            report.append(f"{result['rank']:<6} test_{result['id']}.jpg  {result['label_str']:<12} "
                         f"{result['similarity_score']:>6.2f}%      {result['distance']:>8.4f}")
        
        report.append("")
        
        # Statistics by pattern type
        vertical_results = [r for r in self.results if r['label'] == 0]
        horizontal_results = [r for r in self.results if r['label'] == 1]
        
        report.append("STATISTICS BY PATTERN TYPE:")
        report.append("-" * 80)
        
        report.append(f"\nVERTICAL PATTERNS ({len(vertical_results)} images):")
        if vertical_results:
            report.append(f"  Best Match:        test_{vertical_results[0]['id']}.jpg")
            report.append(f"  Avg Distance:      {np.mean([r['distance'] for r in vertical_results]):.4f}")
            report.append(f"  Min Distance:      {vertical_results[0]['distance']:.4f}")
        
        report.append(f"\nHORIZONTAL PATTERNS ({len(horizontal_results)} images):")
        if horizontal_results:
            report.append(f"  Best Match:        test_{horizontal_results[0]['id']}.jpg")
            report.append(f"  Avg Distance:      {np.mean([r['distance'] for r in horizontal_results]):.4f}")
            report.append(f"  Min Distance:      {horizontal_results[0]['distance']:.4f}")
        
        report.append("")
        
        # Distribution analysis
        report.append("SIMILARITY DISTRIBUTION:")
        report.append("-" * 80)
        all_distances = [r['distance'] for r in self.results]
        report.append(f"Mean Distance:      {np.mean(all_distances):.4f}")
        report.append(f"Std Dev:            {np.std(all_distances):.4f}")
        report.append(f"Min Distance:       {np.min(all_distances):.4f}")
        report.append(f"Max Distance:       {np.max(all_distances):.4f}")
        report.append(f"Median Distance:    {np.median(all_distances):.4f}")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n✓ Report saved to {output_file}")
        
        return report_text
    
    def visualize_results(self, input_path):
        """Create comprehensive visualization"""
        if not self.results:
            print("No results to visualize. Run calculate_similarity_scores first.")
            return
        
        fig = plt.figure(figsize=(20, 12))
        
        # Get data
        vertical_results = [r for r in self.results if r['label'] == 0]
        horizontal_results = [r for r in self.results if r['label'] == 1]
        
        # ============================================================
        # Plot 1: Top 10 Similar Images (Bar Chart)
        # ============================================================
        ax1 = plt.subplot(3, 4, 1)
        top_10 = self.results[:10]
        colors = ['#2ecc71' if r['label'] == 0 else '#3498db' for r in top_10]
        
        bars = ax1.barh([f"test_{r['id']}" for r in top_10], 
                        [r['similarity_score'] for r in top_10],
                        color=colors, alpha=0.8, edgecolor='black')
        
        ax1.set_xlabel('Similarity Score (%)', fontweight='bold')
        ax1.set_title('Top 10 Most Similar Images', fontweight='bold', fontsize=12)
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, result) in enumerate(zip(bars, top_10)):
            width = bar.get_width()
            ax1.text(width + 1, bar.get_y() + bar.get_height()/2,
                    f'{result["similarity_score"]:.1f}%',
                    ha='left', va='center', fontweight='bold', fontsize=8)
        
        # ============================================================
        # Plot 2: Feature Space Scatter Plot
        # ============================================================
        ax2 = plt.subplot(3, 4, 2)
        
        # Plot all database images
        v_verts = [r['v_density'] for r in vertical_results]
        h_verts = [r['h_density'] for r in vertical_results]
        v_horiz = [r['v_density'] for r in horizontal_results]
        h_horiz = [r['h_density'] for r in horizontal_results]
        
        ax2.scatter(v_verts, h_verts, c='#2ecc71', alpha=0.6, s=50, 
                   label='Vertical Patterns', edgecolors='black', linewidth=0.5)
        ax2.scatter(v_horiz, h_horiz, c='#3498db', alpha=0.6, s=50,
                   label='Horizontal Patterns', edgecolors='black', linewidth=0.5)
        
        # Plot input image (larger star)
        ax2.scatter(self.input_features[0], self.input_features[1], 
                   c='red', s=400, marker='*', edgecolors='black', 
                   linewidth=2, label='Input Image', zorder=5)
        
        # Draw line to closest match
        closest = self.results[0]
        ax2.plot([self.input_features[0], closest['v_density']],
                [self.input_features[1], closest['h_density']],
                'r--', linewidth=2, alpha=0.7, label=f'Closest Match (test_{closest["id"]})')
        
        ax2.set_xlabel('Vertical Density', fontweight='bold')
        ax2.set_ylabel('Horizontal Density', fontweight='bold')
        ax2.set_title('Feature Space Distribution', fontweight='bold', fontsize=12)
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(alpha=0.3)
        
        # ============================================================
        # Plot 3: Distance Distribution
        # ============================================================
        ax3 = plt.subplot(3, 4, 3)
        
        distances_v = [r['distance'] for r in vertical_results]
        distances_h = [r['distance'] for r in horizontal_results]
        
        ax3.hist(distances_v, bins=20, alpha=0.6, color='#2ecc71', 
                label='Vertical', edgecolor='black')
        ax3.hist(distances_h, bins=20, alpha=0.6, color='#3498db',
                label='Horizontal', edgecolor='black')
        
        ax3.axvline(self.results[0]['distance'], color='red', linestyle='--',
                   linewidth=2, label=f'Best Match: {self.results[0]["distance"]:.2f}')
        
        ax3.set_xlabel('Distance', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Distance Distribution', fontweight='bold', fontsize=12)
        ax3.legend(fontsize=9)
        ax3.grid(axis='y', alpha=0.3)
        
        # ============================================================
        # Plot 4: Similarity Score Distribution
        # ============================================================
        ax4 = plt.subplot(3, 4, 4)
        
        similarity_scores = [r['similarity_score'] for r in self.results]
        ax4.boxplot([similarity_scores], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='#9b59b6', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        
        ax4.set_ylabel('Similarity Score (%)', fontweight='bold')
        ax4.set_title('Similarity Score Distribution', fontweight='bold', fontsize=12)
        ax4.set_xticklabels(['All Matches'])
        ax4.grid(axis='y', alpha=0.3)
        
        # Add statistics
        mean_sim = np.mean(similarity_scores)
        median_sim = np.median(similarity_scores)
        ax4.text(1.15, mean_sim, f'Mean: {mean_sim:.1f}%', 
                fontsize=9, va='center')
        ax4.text(1.15, median_sim, f'Median: {median_sim:.1f}%',
                fontsize=9, va='center')
        
        # ============================================================
        # Plot 5: Ranking Chart
        # ============================================================
        ax5 = plt.subplot(3, 4, 5)
        
        ranks = [r['rank'] for r in self.results]
        distances = [r['distance'] for r in self.results]
        colors_rank = ['#2ecc71' if r['label'] == 0 else '#3498db' for r in self.results]
        
        ax5.scatter(ranks, distances, c=colors_rank, alpha=0.6, s=30)
        ax5.plot(ranks, distances, 'gray', alpha=0.3, linewidth=1)
        
        ax5.set_xlabel('Rank', fontweight='bold')
        ax5.set_ylabel('Distance', fontweight='bold')
        ax5.set_title('Rank vs Distance', fontweight='bold', fontsize=12)
        ax5.grid(alpha=0.3)
        
        # Highlight top 10
        top_10_ranks = ranks[:10]
        top_10_dist = distances[:10]
        ax5.scatter(top_10_ranks, top_10_dist, c='red', s=80, 
                   marker='o', edgecolors='black', linewidth=1.5,
                   label='Top 10', zorder=5)
        ax5.legend(fontsize=9)
        
        # ============================================================
        # Plot 6: Pattern Type Comparison (CLASSIFICATION DECISION)
        # ============================================================
        ax6 = plt.subplot(3, 4, 6)
        
        avg_dist_v = np.mean([r['distance'] for r in vertical_results])
        avg_dist_h = np.mean([r['distance'] for r in horizontal_results])
        
        # Determine classification based on average distance
        if avg_dist_v < avg_dist_h:
            predicted_class = 'VERTICAL'
            colors_bar = ['#2ecc71', '#95a5a6']  # Green for winner, gray for loser
            winner_idx = 0
        else:
            predicted_class = 'HORIZONTAL'
            colors_bar = ['#95a5a6', '#3498db']  # Gray for loser, blue for winner
            winner_idx = 1
        
        categories = ['Vertical\nPatterns', 'Horizontal\nPatterns']
        values = [avg_dist_v, avg_dist_h]
        
        bars = ax6.bar(categories, values, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Highlight the winner with thicker border
        bars[winner_idx].set_edgecolor('red')
        bars[winner_idx].set_linewidth(3)
        
        ax6.set_ylabel('Average Distance', fontweight='bold')
        ax6.set_title(f'Classification Decision: {predicted_class}', 
                     fontweight='bold', fontsize=12, color='red')
        ax6.grid(axis='y', alpha=0.3)
        
        # Add value labels with winner annotation
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            label = f'{val:.2f}'
            if i == winner_idx:
                label += '\n★ WINNER'
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        label, ha='center', va='bottom', fontweight='bold', 
                        fontsize=10, color='red')
            else:
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        label, ha='center', va='bottom', fontweight='bold')
        
        # Add margin information
        margin = abs(avg_dist_v - avg_dist_h)
        ax6.text(0.5, 0.95, f'Confidence Margin: {margin:.4f}', 
                transform=ax6.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontweight='bold', fontsize=9)
        
        # ============================================================
        # Plot 7: Input Image
        # ============================================================
        ax7 = plt.subplot(3, 4, 7)
        
        try:
            input_img = Image.open(input_path)
            ax7.imshow(input_img)
            ax7.axis('off')
            ax7.set_title('Input Image', fontweight='bold', fontsize=12)
        except:
            ax7.text(0.5, 0.5, 'Input image\nnot available',
                    ha='center', va='center', fontsize=12)
            ax7.axis('off')
        
        # ============================================================
        # Plot 8: Best Match Image
        # ============================================================
        ax8 = plt.subplot(3, 4, 8)
        
        best_match_path = self.results[0]['filepath']
        try:
            best_img = Image.open(best_match_path)
            ax8.imshow(best_img)
            ax8.axis('off')
            ax8.set_title(f'Best Match: test_{self.results[0]["id"]}.jpg\n'
                         f'Similarity: {self.results[0]["similarity_score"]:.1f}%',
                         fontweight='bold', fontsize=11)
        except:
            ax8.text(0.5, 0.5, 'Best match\nnot available',
                    ha='center', va='center', fontsize=12)
            ax8.axis('off')
        
        # ============================================================
        # Plot 9-11: Top 3 Additional Matches (Small Grid)
        # ============================================================
        match_positions = [9, 10, 11]  # Bottom row positions
        for plot_idx, pos in enumerate(match_positions):
            ax = plt.subplot(3, 4, pos)
            
            result_idx = plot_idx + 1  # Skip best match (already shown in plot 8)
            if result_idx < len(self.results):
                result = self.results[result_idx]
                try:
                    img = Image.open(result['filepath'])
                    ax.imshow(img)
                    ax.axis('off')
                    ax.set_title(f'#{result_idx + 1}: test_{result["id"]}\n'
                               f'{result["similarity_score"]:.1f}%',
                               fontsize=9, fontweight='bold')
                except:
                    ax.text(0.5, 0.5, f'Rank {result_idx + 1}\nNot available',
                           ha='center', va='center', fontsize=9)
                    ax.axis('off')
            else:
                ax.axis('off')
        
        # ============================================================
        # Plot 12: Summary Table
        # ============================================================
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        # Calculate classification based on average distance
        avg_dist_v = np.mean([r['distance'] for r in vertical_results])
        avg_dist_h = np.mean([r['distance'] for r in horizontal_results])
        
        if avg_dist_v < avg_dist_h:
            predicted_class = 'VERTICAL'
            confidence_margin = avg_dist_h - avg_dist_v
        else:
            predicted_class = 'HORIZONTAL'
            confidence_margin = avg_dist_v - avg_dist_h
        
        summary_data = [
            ['Metric', 'Value'],
            ['─' * 30, '─' * 20],
            ['CLASSIFICATION', ''],
            ['Predicted Pattern', predicted_class],
            ['Confidence Margin', f'{confidence_margin:.4f}'],
            ['', ''],
            ['AVERAGE DISTANCES', ''],
            ['To Vertical Class', f'{avg_dist_v:.4f}'],
            ['To Horizontal Class', f'{avg_dist_h:.4f}'],
            ['', ''],
            ['NEAREST NEIGHBOR', ''],
            ['Best Match', f'test_{self.results[0]["id"]}.jpg'],
            ['Distance', f'{self.results[0]["distance"]:.4f}'],
            ['Pattern', self.results[0]['label_str']],
        ]
        
        table = ax12.table(cellText=summary_data, cellLoc='left', loc='center',
                          colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.2)
        
        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style section headers
        for row_idx in [2, 6, 10]:  # Section header rows
            table[(row_idx, 0)].set_facecolor('#e74c3c')
            table[(row_idx, 0)].set_text_props(weight='bold', color='white')
            table[(row_idx, 1)].set_facecolor('#e74c3c')
        
        # Alternate row colors
        for i in range(2, len(summary_data)):
            if i not in [2, 6, 10]:  # Skip section headers
                for j in range(2):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#ecf0f1')
        
        # Highlight classification result
        table[(3, 0)].set_text_props(weight='bold')
        table[(3, 1)].set_text_props(weight='bold', color='red')
        
        ax12.set_title('Classification Summary', fontweight='bold', fontsize=12, pad=20)
        
        # Main title
        fig.suptitle('Pattern Recognition Analysis Dashboard', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save
        output_file = 'plot/pattern_analysis_dashboard.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to {output_file}")
        
        plt.show()
        
        return fig


def main():
    """Main analysis function"""
    print("=" * 80)
    print("PATTERN RECOGNITION ANALYSIS TOOL")
    print("=" * 80)
    print()
    
    # Get input image
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 pattern_analysis.py <input_image_path>")
        print("\nExample: python3 pattern_analysis.py input.jpg")
        return
    
    input_path = sys.argv[1]
    
    if not os.path.exists(input_path):
        print(f"✗ Error: Input image not found: {input_path}")
        return
    
    # Check for scipy
    try:
        import scipy
    except ImportError:
        print("Installing scipy for convolution...")
        os.system("pip install scipy")
    
    # Initialize analyzer
    analyzer = PatternAnalyzer(dataset_path='dataset', output_path='images')
    
    # Load dataset
    database = analyzer.load_dataset_features()
    
    if not database:
        print("✗ Error: No images found in dataset/ folder")
        print("Please ensure you have dataset images: dataset/test_0.jpg, test_1.jpg, etc.")
        return
    
    # Calculate similarities
    results = analyzer.calculate_similarity_scores(input_path, database)
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    analyzer.visualize_results(input_path)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  📊 pattern_analysis_dashboard.png  - Visual analysis")
    print("=" * 80)


if __name__ == "__main__":
    main()