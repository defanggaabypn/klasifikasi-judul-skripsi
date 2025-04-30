# Visualization utilities for model comparison
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from io import BytesIO
import base64
import pandas as pd

class ModelVisualizer:
    def __init__(self):
        """Initialize the visualization utilities"""
        self.plt_style = 'ggplot'
        plt.style.use(self.plt_style)
    
    def generate_confusion_matrix(self, y_true, y_pred, labels=None, title="Confusion Matrix", cmap='Blues'):
        """
        Generate a confusion matrix visualization
        
        Parameters:
        - y_true: True labels
        - y_pred: Predicted labels
        - labels: List of class labels
        - title: Chart title
        - cmap: Colormap for visualization
        
        Returns:
        - Base64-encoded image string
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=cmap)
        plt.title(title, fontsize=15)
        
        # Convert to base64 image
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return {
            'image': img_str,
            'matrix': cm.tolist(),
            'labels': labels
        }
    
    def compare_confusion_matrices(self, y_true, predictions_dict, labels=None):
        """
        Compare confusion matrices from multiple models
        
        Parameters:
        - y_true: True labels
        - predictions_dict: Dictionary mapping model names to their predictions
        - labels: List of class labels
        
        Returns:
        - Dictionary of base64-encoded images for each model
        """
        if not labels:
            labels = sorted(list(set(y_true)))
        
        results = {}
        cmaps = {
            'KNN': 'Blues',
            'Decision Tree': 'Greens',
            'Random Forest': 'Oranges',
            'SVM': 'Purples',
            'Naive Bayes': 'Reds'
        }
        
        for model_name, y_pred in predictions_dict.items():
            cmap = cmaps.get(model_name, 'viridis')
            title = f"Confusion Matrix - {model_name}"
            results[model_name] = self.generate_confusion_matrix(y_true, y_pred, labels, title, cmap)
        
        return results
    
    def generate_model_comparison_chart(self, metrics_dict, metric_name='accuracy'):
        """
        Generate a bar chart comparing a specific metric across models
        
        Parameters:
        - metrics_dict: Dictionary mapping model names to their metrics
        - metric_name: Name of the metric to compare
        
        Returns:
        - Base64-encoded image string
        """
        models = list(metrics_dict.keys())
        values = [metrics[metric_name] for metrics in metrics_dict.values()]
        
        plt.figure(figsize=(10, 6))
        
        # Create bars with custom colors
        colors = ['#4285F4', '#34A853', '#FBBC05', '#EA4335', '#8F00FF']
        
        bars = plt.bar(models, values, alpha=0.8, color=colors[:len(models)])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.ylim(0, max(1.0, max(values) + 0.1))
        plt.ylabel(metric_name.capitalize())
        plt.title(f'Model Comparison: {metric_name.capitalize()}')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add a horizontal reference line at 0.5
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)
        
        # Convert to base64 image
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return img_str
    
    def generate_multi_metric_comparison(self, metrics_dict):
        """
        Generate a comparison of multiple metrics across models
        
        Parameters:
        - metrics_dict: Dictionary mapping model names to their metrics
        
        Returns:
        - Base64-encoded image string
        """
        models = list(metrics_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        # Extract data
        data = []
        for model in models:
            model_data = []
            for metric in metrics:
                if metric in metrics_dict[model]:
                    model_data.append(metrics_dict[model][metric])
                else:
                    model_data.append(0)
            data.append(model_data)
        
        # Create bar chart
        x = np.arange(len(metric_labels))
        width = 0.8 / len(models)  # Width of the bars
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for i, (model, values) in enumerate(zip(models, data)):
            offset = width * (i - len(models)/2 + 0.5)
            bars = ax.bar(x + offset, values, width, label=model, alpha=0.8)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Customize chart
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(models))
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Convert to base64 image
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return img_str
    
    def generate_class_wise_comparison(self, class_metrics, metric='f1'):
        """
        Generate a comparison of model performance for each class
        
        Parameters:
        - class_metrics: Dictionary of per-class metrics for each model
        - metric: Metric to compare ('precision', 'recall', 'f1')
        
        Returns:
        - Base64-encoded image string
        """
        models = list(class_metrics.keys())
        all_classes = set()
        for model_metrics in class_metrics.values():
            all_classes.update(model_metrics.keys())
        classes = sorted(list(all_classes))
        
        # Extract data
        data = []
        for model in models:
            model_data = []
            for cls in classes:
                if cls in class_metrics[model] and metric in class_metrics[model][cls]:
                    model_data.append(class_metrics[model][cls][metric])
                else:
                    model_data.append(0)
            data.append(model_data)
        
        # Create bar chart
        x = np.arange(len(classes))
        width = 0.8 / len(models)  # Width of the bars
        
        fig, ax = plt.subplots(figsize=(max(10, len(classes) * 1.2), 7))
        
        for i, (model, values) in enumerate(zip(models, data)):
            offset = width * (i - len(models)/2 + 0.5)
            rects = ax.bar(x + offset, values, width, label=model, alpha=0.8)
        
        # Customize chart
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f'{metric.capitalize()} Score')
        ax.set_title(f'Class-wise {metric.capitalize()} Score Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(models))
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Convert to base64 image
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return img_str
    
    def generate_decision_boundary_chart(self, title="Model Decision Boundaries"):
        """
        Generate a placeholder for decision boundary visualization
        
        Note: This doesn't actually plot decision boundaries because we're working
        with high-dimensional data (IndoBERT embeddings), but serves as a placeholder
        for potential dimensionality reduction visualizations
        
        Returns:
        - Base64-encoded image string
        """
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "Decision boundary visualization not available for high-dimensional embeddings", 
                ha='center', va='center', fontsize=14)
        plt.title(title)
        plt.axis('off')
        
        # Convert to base64 image
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return img_str
    
    def generate_all_visualizations(self, y_true, predictions_dict, metrics_dict, class_metrics):
        """
        Generate all visualizations at once
        
        Parameters:
        - y_true: True labels
        - predictions_dict: Dictionary mapping model names to their predictions
        - metrics_dict: Dictionary mapping model names to their metrics
        - class_metrics: Dictionary of per-class metrics for each model
        
        Returns:
        - Dictionary of all visualizations
        """
        labels = sorted(list(set(y_true)))
        
        visualizations = {
            'confusion_matrices': self.compare_confusion_matrices(y_true, predictions_dict, labels),
            'accuracy_comparison': self.generate_model_comparison_chart(metrics_dict, 'accuracy'),
            'metrics_comparison': self.generate_multi_metric_comparison(metrics_dict),
            'f1_class_comparison': self.generate_class_wise_comparison(class_metrics, 'f1'),
            'precision_class_comparison': self.generate_class_wise_comparison(class_metrics, 'precision'),
            'recall_class_comparison': self.generate_class_wise_comparison(class_metrics, 'recall')
        }
        
        return visualizations