# Model evaluator to compare model performance
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from io import BytesIO
import base64
import json
import pandas as pd

class ModelEvaluator:
    def __init__(self):
        """Initialize the model evaluator to compare different models"""
        self.models = {}
        self.metrics = {}
        self.class_metrics = {}
        self.pred_results = {}
        self.features = None
        self.labels = None
    
    def add_model(self, name, model):
        """Add a model to the evaluator"""
        self.models[name] = model
        return self
    
    def evaluate_all(self, X_test, y_test):
        """Evaluate all models on the same test data"""
        if not self.models:
            raise ValueError("No models added for evaluation. Use add_model() first.")
        
        self.features = X_test
        self.labels = y_test
        
        # For each model, evaluate and store results
        for name, model in self.models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            self.pred_results[name] = y_pred
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Store metrics
            self.metrics[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'report': report
            }
            
            # Analyze per-class performance
            class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(y_test, y_pred)
            classes = sorted(list(set(y_test)))
            
            self.class_metrics[name] = {}
            for i, cls in enumerate(classes):
                self.class_metrics[name][cls] = {
                    'precision': class_precision[i],
                    'recall': class_recall[i],
                    'f1': class_f1[i],
                    'support': int(class_support[i])
                }
        
        return self.metrics
    
    def generate_accuracy_comparison(self):
        """Generate a bar chart comparing model accuracies"""
        if not self.metrics:
            raise ValueError("No evaluation metrics available. Call evaluate_all() first.")
        
        model_names = list(self.metrics.keys())
        accuracies = [metrics['accuracy'] for metrics in self.metrics.values()]
        
        plt.figure(figsize=(8, 5))
        bars = plt.bar(model_names, accuracies, color=['blue', 'green'])
        plt.ylim(0, 1)
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        # Add a horizontal line at 0.5 for reference
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
        
        # Convert plot to base64 image
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return img_str
    
    def generate_metrics_comparison(self):
        """Generate a detailed metrics comparison chart"""
        if not self.metrics:
            raise ValueError("No evaluation metrics available. Call evaluate_all() first.")
        
        model_names = list(self.metrics.keys())
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        # Prepare data
        data = []
        for name in model_names:
            metrics = self.metrics[name]
            data.append([
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1']
            ])
        
        # Plotting
        x = np.arange(len(metrics_names))
        width = 0.35 if len(model_names) <= 2 else 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, (name, values) in enumerate(zip(model_names, data)):
            offset = width * (i - (len(model_names) - 1) / 2)
            bars = ax.bar(x + offset, values, width, label=name, 
                         alpha=0.7, color=['blue', 'green'][i % 2])
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Convert plot to base64 image
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return img_str
    
    def generate_class_performance_comparison(self, metric='f1'):
        """Generate a comparison of per-class performance"""
        if not self.class_metrics:
            raise ValueError("No class metrics available. Call evaluate_all() first.")
        
        model_names = list(self.class_metrics.keys())
        all_classes = set()
        for model_metrics in self.class_metrics.values():
            all_classes.update(model_metrics.keys())
        all_classes = sorted(list(all_classes))
        
        # Prepare data
        data = []
        for name in model_names:
            model_data = []
            for cls in all_classes:
                if cls in self.class_metrics[name]:
                    model_data.append(self.class_metrics[name][cls][metric])
                else:
                    model_data.append(0)
            data.append(model_data)
        
        # Plotting
        x = np.arange(len(all_classes))
        width = 0.35 if len(model_names) <= 2 else 0.25
        
        fig, ax = plt.subplots(figsize=(max(8, len(all_classes) * 1.5), 6))
        
        for i, (name, values) in enumerate(zip(model_names, data)):
            offset = width * (i - (len(model_names) - 1) / 2)
            bars = ax.bar(x + offset, values, width, label=name, 
                         alpha=0.7, color=['blue', 'green'][i % 2])
        
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(f'{metric.capitalize()} Score')
        ax.set_title(f'Per-Class {metric.capitalize()} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(all_classes, rotation=45)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Convert plot to base64 image
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return img_str
    
    def generate_prediction_comparison_table(self):
        """Generate a comparison table of model predictions"""
        if not self.pred_results:
            raise ValueError("No prediction results available. Call evaluate_all() first.")
        
        model_names = list(self.pred_results.keys())
        
        # Create a table of results
        results = []
        for i, (true_label, features) in enumerate(zip(self.labels, self.features)):
            row = {
                'sample_index': i,
                'true_label': true_label
            }
            
            # Add predictions from each model
            correct_count = 0
            for name in model_names:
                pred = self.pred_results[name][i]
                row[f'{name}_prediction'] = pred
                if pred == true_label:
                    correct_count += 1
            
            # Add consensus status
            if correct_count == len(model_names):
                row['consensus'] = 'All Correct'
            elif correct_count == 0:
                row['consensus'] = 'All Wrong'
            else:
                row['consensus'] = 'Mixed'
            
            results.append(row)
        
        return results
    
    def get_detailed_report(self):
        """Generate a detailed report with all comparisons"""
        if not self.metrics:
            raise ValueError("No evaluation metrics available. Call evaluate_all() first.")
        
        report = {
            'overall_performance': {},
            'per_class_performance': {},
            'model_comparison': {}
        }
        
        # Overall performance
        for name, metrics in self.metrics.items():
            report['overall_performance'][name] = {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1']
            }
        
        # Per-class performance
        for name, class_metrics in self.class_metrics.items():
            report['per_class_performance'][name] = class_metrics
        
        # Model comparison
        all_classes = set()
        for model_metrics in self.class_metrics.values():
            all_classes.update(model_metrics.keys())
        all_classes = sorted(list(all_classes))
        
        # Best model per class
        best_models = {}
        for cls in all_classes:
            best_f1 = -1
            best_model = None
            for name in self.class_metrics:
                if cls in self.class_metrics[name]:
                    f1 = self.class_metrics[name][cls]['f1']
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model = name
            best_models[cls] = {
                'best_model': best_model,
                'f1_score': best_f1
            }
        
        report['model_comparison']['best_per_class'] = best_models
        
        # Best overall model
        best_overall_acc = -1
        best_overall_model = None
        for name, metrics in self.metrics.items():
            if metrics['accuracy'] > best_overall_acc:
                best_overall_acc = metrics['accuracy']
                best_overall_model = name
        
        report['model_comparison']['best_overall'] = {
            'model': best_overall_model,
            'accuracy': best_overall_acc
        }
        
        return report
    
    def generate_comparison_images(self):
        """Generate all comparison visualizations"""
        return {
            'accuracy_comparison': self.generate_accuracy_comparison(),
            'metrics_comparison': self.generate_metrics_comparison(),
            'class_f1_comparison': self.generate_class_performance_comparison(metric='f1'),
            'class_precision_comparison': self.generate_class_performance_comparison(metric='precision'),
            'class_recall_comparison': self.generate_class_performance_comparison(metric='recall')
        }