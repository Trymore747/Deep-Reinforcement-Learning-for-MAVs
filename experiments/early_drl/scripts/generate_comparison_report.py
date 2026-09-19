#!/usr/bin/env python3
"""
Visualization and Comparison Tool
Generate plots, charts, and tables comparing RRT, A*, and PPO
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import json
import csv
import os
from datetime import datetime
import pandas as pd


class NavigationComparison:
    """Generate comprehensive comparison visualizations"""
    
    def __init__(self, data_dir="/tmp/nav_comparison"):
        self.data_dir = data_dir
        self.algorithms = ['RRT', 'ASTAR', 'PPO']
        self.metrics = {}
        self.trajectories = {}
        
        # Load data
        self.load_all_data()
    
    def load_all_data(self):
        """Load metrics and trajectory data for all algorithms"""
        for algo in self.algorithms:
            # Load metrics
            metrics_file = f"{self.data_dir}/{algo}_metrics.json"
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    self.metrics[algo] = json.load(f)
                print(f"Loaded metrics for {algo}")
            else:
                print(f"Warning: No metrics found for {algo}")
            
            # Load trajectory
            traj_file = f"{self.data_dir}/{algo}_trajectory.csv"
            if os.path.exists(traj_file):
                self.trajectories[algo] = pd.read_csv(traj_file)
                print(f"Loaded trajectory for {algo}: {len(self.trajectories[algo])} points")
            else:
                print(f"Warning: No trajectory found for {algo}")
    
    def plot_trajectories(self):
        """Plot 3D trajectories for all algorithms"""
        fig = plt.figure(figsize=(15, 5))
        
        # 2D top view
        ax1 = fig.add_subplot(131)
        for algo in self.algorithms:
            if algo in self.trajectories:
                data = self.trajectories[algo]
                ax1.plot(data['x'], data['y'], label=algo, linewidth=2)
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Top View - XY Trajectory')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Side view
        ax2 = fig.add_subplot(132)
        for algo in self.algorithms:
            if algo in self.trajectories:
                data = self.trajectories[algo]
                ax2.plot(data['x'], data['z'], label=algo, linewidth=2)
        
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Z Position (m)')
        ax2.set_title('Side View - XZ Trajectory')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3D view
        ax3 = fig.add_subplot(133, projection='3d')
        for algo in self.algorithms:
            if algo in self.trajectories:
                data = self.trajectories[algo]
                ax3.plot(data['x'], data['y'], data['z'], label=algo, linewidth=2)
        
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_zlabel('Z (m)')
        ax3.set_title('3D Trajectory')
        ax3.legend()
        
        plt.tight_layout()
        filename = f"{self.data_dir}/trajectories_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved trajectory plot: {filename}")
        plt.close()
    
    def plot_speed_profiles(self):
        """Plot speed over time for all algorithms"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Speed vs time
        ax1 = axes[0, 0]
        for algo in self.algorithms:
            if algo in self.trajectories:
                data = self.trajectories[algo]
                speed = np.sqrt(data['vx']**2 + data['vy']**2 + data['vz']**2)
                ax1.plot(data['time'], speed, label=algo, linewidth=2)
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Speed (m/s)')
        ax1.set_title('Speed Profile Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Speed distribution
        ax2 = axes[0, 1]
        speeds_data = []
        labels = []
        for algo in self.algorithms:
            if algo in self.trajectories:
                data = self.trajectories[algo]
                speed = np.sqrt(data['vx']**2 + data['vy']**2 + data['vz']**2)
                speeds_data.append(speed)
                labels.append(algo)
        
        ax2.boxplot(speeds_data, labels=labels)
        ax2.set_ylabel('Speed (m/s)')
        ax2.set_title('Speed Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Speed vs distance
        ax3 = axes[1, 0]
        for algo in self.algorithms:
            if algo in self.trajectories:
                data = self.trajectories[algo]
                speed = np.sqrt(data['vx']**2 + data['vy']**2 + data['vz']**2)
                ax3.plot(data['x'], speed, label=algo, linewidth=2)
        
        ax3.set_xlabel('X Position (m)')
        ax3.set_ylabel('Speed (m/s)')
        ax3.set_title('Speed vs Position')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Average speed comparison
        ax4 = axes[1, 1]
        avg_speeds = []
        max_speeds = []
        algo_names = []
        for algo in self.algorithms:
            if algo in self.metrics:
                avg_speeds.append(self.metrics[algo]['avg_speed'])
                max_speeds.append(self.metrics[algo]['max_speed'])
                algo_names.append(algo)
        
        x = np.arange(len(algo_names))
        width = 0.35
        ax4.bar(x - width/2, avg_speeds, width, label='Average Speed')
        ax4.bar(x + width/2, max_speeds, width, label='Max Speed')
        ax4.set_ylabel('Speed (m/s)')
        ax4.set_title('Speed Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(algo_names)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filename = f"{self.data_dir}/speed_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved speed plot: {filename}")
        plt.close()
    
    def plot_performance_metrics(self):
        """Plot key performance metrics comparison"""
        if not self.metrics:
            print("No metrics data available")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        metrics_to_plot = [
            ('mission_duration', 'Mission Duration (s)', axes[0, 0]),
            ('path_length', 'Path Length (m)', axes[0, 1]),
            ('path_efficiency', 'Path Efficiency (%)', axes[0, 2]),
            ('avg_acceleration', 'Avg Acceleration (m/s²)', axes[1, 0]),
            ('energy_consumption', 'Energy Consumption', axes[1, 1]),
            ('control_smoothness', 'Control Smoothness', axes[1, 2])
        ]
        
        for metric_key, title, ax in metrics_to_plot:
            values = []
            labels = []
            colors = []
            
            for algo in self.algorithms:
                if algo in self.metrics and metric_key in self.metrics[algo]:
                    value = self.metrics[algo][metric_key]
                    if metric_key == 'path_efficiency':
                        value *= 100  # Convert to percentage
                    values.append(value)
                    labels.append(algo)
                    
                    # Color by algorithm
                    if algo == 'RRT':
                        colors.append('#1f77b4')
                    elif algo == 'ASTAR':
                        colors.append('#ff7f0e')
                    else:  # PPO
                        colors.append('#2ca02c')
            
            if values:
                bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
                ax.set_title(title, fontweight='bold')
                ax.set_ylabel(title.split('(')[0].strip())
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('Performance Metrics Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        filename = f"{self.data_dir}/performance_metrics.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved performance metrics plot: {filename}")
        plt.close()
    
    def generate_comparison_table(self):
        """Generate detailed comparison table"""
        if not self.metrics:
            print("No metrics data available")
            return
        
        # Define metrics to compare
        comparison_metrics = [
            ('success', 'Success', lambda x: 'YES' if x else 'NO'),
            ('mission_duration', 'Time (s)', lambda x: f'{x:.2f}'),
            ('path_length', 'Path Length (m)', lambda x: f'{x:.2f}'),
            ('path_efficiency', 'Efficiency (%)', lambda x: f'{x*100:.1f}'),
            ('avg_speed', 'Avg Speed (m/s)', lambda x: f'{x:.2f}'),
            ('max_speed', 'Max Speed (m/s)', lambda x: f'{x:.2f}'),
            ('avg_acceleration', 'Avg Accel (m/s²)', lambda x: f'{x:.2f}'),
            ('control_smoothness', 'Smoothness', lambda x: f'{x:.3f}'),
            ('energy_consumption', 'Energy', lambda x: f'{x:.2f}'),
            ('collision_detected', 'Collision', lambda x: 'YES' if x else 'NO')
        ]
        
        # Create table
        table_data = []
        for metric_key, metric_name, formatter in comparison_metrics:
            row = [metric_name]
            for algo in self.algorithms:
                if algo in self.metrics and metric_key in self.metrics[algo]:
                    value = self.metrics[algo][metric_key]
                    row.append(formatter(value))
                else:
                    row.append('N/A')
            table_data.append(row)
        
        # Generate formatted table
        table_str = "\n" + "="*80 + "\n"
        table_str += "COMPREHENSIVE COMPARISON TABLE\n"
        table_str += "="*80 + "\n\n"
        
        # Header
        header = f"{'Metric':<25}"
        for algo in self.algorithms:
            header += f"{algo:>15}"
        table_str += header + "\n"
        table_str += "-"*80 + "\n"
        
        # Data rows
        for row in table_data:
            line = f"{row[0]:<25}"
            for val in row[1:]:
                line += f"{val:>15}"
            table_str += line + "\n"
        
        table_str += "="*80 + "\n"
        
        # Save table
        table_file = f"{self.data_dir}/comparison_table.txt"
        with open(table_file, 'w') as f:
            f.write(table_str)
        
        print(table_str)
        print(f"Saved comparison table: {table_file}")
        
        return table_str
    
    def generate_winner_analysis(self):
        """Determine winners in each category"""
        if not self.metrics:
            print("No metrics data available")
            return
        
        analysis = "\n" + "="*80 + "\n"
        analysis += "WINNER ANALYSIS - Best Performance by Category\n"
        analysis += "="*80 + "\n\n"
        
        categories = [
            ('mission_duration', 'Fastest Completion', 'min'),
            ('path_efficiency', 'Most Efficient Path', 'max'),
            ('avg_speed', 'Highest Average Speed', 'max'),
            ('max_speed', 'Highest Peak Speed', 'max'),
            ('control_smoothness', 'Smoothest Control', 'min'),
            ('energy_consumption', 'Lowest Energy', 'min'),
        ]
        
        for metric_key, category_name, optimize in categories:
            values = {}
            for algo in self.algorithms:
                if algo in self.metrics and metric_key in self.metrics[algo]:
                    values[algo] = self.metrics[algo][metric_key]
            
            if values:
                if optimize == 'max':
                    winner = max(values, key=values.get)
                    winner_value = values[winner]
                else:
                    winner = min(values, key=values.get)
                    winner_value = values[winner]
                
                analysis += f"{category_name}:\n"
                analysis += f"  Winner: {winner} ({winner_value:.3f})\n"
                analysis += f"  All values: {values}\n\n"
        
        # Overall recommendation
        analysis += "="*80 + "\n"
        analysis += "OVERALL RECOMMENDATIONS:\n\n"
        
        # Count wins
        wins = {algo: 0 for algo in self.algorithms}
        for metric_key, _, optimize in categories:
            values = {}
            for algo in self.algorithms:
                if algo in self.metrics and metric_key in self.metrics[algo]:
                    values[algo] = self.metrics[algo][metric_key]
            if values:
                if optimize == 'max':
                    winner = max(values, key=values.get)
                else:
                    winner = min(values, key=values.get)
                wins[winner] += 1
        
        for algo, win_count in sorted(wins.items(), key=lambda x: x[1], reverse=True):
            analysis += f"{algo}: {win_count} category wins\n"
        
        analysis += "\n" + "="*80 + "\n"
        
        # Save analysis
        analysis_file = f"{self.data_dir}/winner_analysis.txt"
        with open(analysis_file, 'w') as f:
            f.write(analysis)
        
        print(analysis)
        print(f"Saved winner analysis: {analysis_file}")
        
        return analysis
    
    def generate_full_report(self):
        """Generate complete comparison report with all visualizations"""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE COMPARISON REPORT")
        print("="*80 + "\n")
        
        # Generate all visualizations
        print("1. Plotting trajectories...")
        self.plot_trajectories()
        
        print("2. Plotting speed profiles...")
        self.plot_speed_profiles()
        
        print("3. Plotting performance metrics...")
        self.plot_performance_metrics()
        
        print("4. Generating comparison table...")
        self.generate_comparison_table()
        
        print("5. Generating winner analysis...")
        self.generate_winner_analysis()
        
        print("\n" + "="*80)
        print(f"COMPLETE! All reports saved to: {self.data_dir}")
        print("="*80 + "\n")
        
        print("Generated files:")
        for f in os.listdir(self.data_dir):
            if f.endswith(('.png', '.txt', '.json', '.csv')):
                filepath = os.path.join(self.data_dir, f)
                size = os.path.getsize(filepath)
                print(f"  - {f} ({size/1024:.1f} KB)")


if __name__ == '__main__':
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/nav_comparison"
    
    print(f"Loading data from: {data_dir}")
    comparison = NavigationComparison(data_dir)
    comparison.generate_full_report()
