# Decision Tree Model implementation for text classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class DTModel:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 criterion='gini', max_features=None, random_state=42):
        """
        Initialize the Decision Tree model with customizable parameters
        
        Parameters:
        - max_depth: Maximum depth of the tree
        - min_samples_split: Minimum samples required to split a node
        - min_samples_leaf: Minimum samples required at a leaf node
        - criterion: Function to measure quality of split ('gini' or 'entropy')
        - max_features: Number of features to consider for best split
        - random_state: Random state for reproducibility
        """
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            max_features=max_features,
            random_state=random_state
        )
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        self.trained = False
        self.metrics = {}
        self.class_metrics = {}
        self.confusion_matrix = None
        self.classification_report_dict = None
        
    def fit(self, X_train, y_train):
        """Train the Decision Tree model on the given data"""
        self.model.fit(X_train, y_train)
        self.trained = True
        self.X_train = X_train
        self.y_train = y_train
        return self
        
    def predict(self, X):
        """Make predictions with the trained model"""
        if not self.trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance on test data and store metrics
        
        Parameters:
        - X_test: Test features
        - y_test: Test labels
        
        Returns:
        - Dictionary with various performance metrics
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Basic metrics
        self.metrics['accuracy'] = accuracy_score(y_test, y_pred)
        
        # Get confusion matrix
        self.confusion_matrix = confusion_matrix(y_test, y_pred)
        
        # Get classification report as dict
        self.classification_report_dict = classification_report(y_test, y_pred, output_dict=True)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
        classes = sorted(list(set(y_test)))
        
        self.class_metrics = {}
        for i, cls in enumerate(classes):
            self.class_metrics[cls] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': int(support[i])
            }
        
        # Analyze misclassifications
        misclassified_indices = [i for i, (true, pred) in enumerate(zip(y_test, y_pred)) if true != pred]
        if misclassified_indices:
            self.metrics['misclassified_count'] = len(misclassified_indices)
            
            # Get probabilities for misclassified samples
            proba = self.predict_proba([X_test[i] for i in misclassified_indices])
            
            # Store detailed info for a few examples
            misclassified = []
            for i, idx in enumerate(misclassified_indices[:5]):  # Limit to 5 examples
                misclassified.append({
                    'index': idx,
                    'true_label': y_test[idx],
                    'predicted_label': y_pred[idx],
                    'probabilities': {cls: float(prob) for cls, prob in zip(self.model.classes_, proba[i])}
                })
            self.metrics['misclassified_examples'] = misclassified
        
        # Add Decision Tree specific metrics
        self.metrics['tree_depth'] = self.model.get_depth()
        self.metrics['n_leaves'] = self.model.get_n_leaves()
        self.metrics['feature_importance'] = {
            i: float(imp) for i, imp in enumerate(self.model.feature_importances_)
        }
        
        # Get class weights at leaf nodes
        self.metrics['leaf_class_distribution'] = self._get_leaf_class_distribution()
        
        return self.metrics
    
    def _get_leaf_class_distribution(self):
        """Get class distribution at each leaf node"""
        n_nodes = self.model.tree_.node_count
        children_left = self.model.tree_.children_left
        children_right = self.model.tree_.children_right
        feature = self.model.tree_.feature
        threshold = self.model.tree_.threshold
        
        # Determine leaf nodes
        is_leaf = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1)]  # (node_id, parent_id)
        while len(stack) > 0:
            node_id, parent_id = stack.pop()
            
            # If we have a test node
            if children_left[node_id] != children_right[node_id]:
                stack.append((children_left[node_id], node_id))
                stack.append((children_right[node_id], node_id))
            else:
                is_leaf[node_id] = True
        
        # Get class distribution for leaf nodes
        leaf_distribution = {}
        for i in range(n_nodes):
            if is_leaf[i]:
                node_value = self.model.tree_.value[i][0]
                total_samples = sum(node_value)
                if total_samples > 0:
                    class_distribution = {
                        str(self.model.classes_[j]): float(node_value[j] / total_samples) 
                        for j in range(len(node_value))
                    }
                    leaf_distribution[i] = {
                        'samples': int(total_samples),
                        'class_distribution': class_distribution
                    }
        
        return leaf_distribution
    
    def generate_confusion_matrix_image(self, labels=None):
        """Generate a confusion matrix visualization as base64 image"""
        if self.confusion_matrix is None:
            raise ValueError("Confusion matrix not available. Call evaluate() first.")
        
        plt.figure(figsize=(8, 6))
        plt.imshow(self.confusion_matrix, interpolation='nearest', cmap=plt.cm.Greens)
        plt.title('Confusion Matrix - Decision Tree')
        plt.colorbar()
        
        if labels is None:
            labels = sorted(list(set(self.y_train)))
            
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        
        # Add text annotations
        thresh = self.confusion_matrix.max() / 2.
        for i in range(self.confusion_matrix.shape[0]):
            for j in range(self.confusion_matrix.shape[1]):
                plt.text(j, i, format(self.confusion_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if self.confusion_matrix[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Convert plot to base64 image
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return img_str
    
    def generate_tree_visualization(self, class_names=None, feature_names=None, max_depth=3):
        """Generate a visualization of the decision tree structure"""
        if not self.trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        plt.figure(figsize=(20, 10))
        plot_tree(self.model, 
                  filled=True, 
                  rounded=True, 
                  class_names=class_names or self.model.classes_,
                  feature_names=feature_names,
                  max_depth=max_depth)
        
        plt.title(f"Decision Tree (Showing max_depth={max_depth})")
        
        # Convert plot to base64 image
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return img_str
    
    def generate_feature_importance_chart(self, top_n=10):
        """Generate a feature importance chart"""
        if not self.trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Get feature importance
        importances = self.model.feature_importances_
        
        # Sort importance and get indices
        if len(importances) > top_n:
            indices = np.argsort(importances)[-top_n:]
            importances = importances[indices]
        else:
            indices = np.arange(len(importances))
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importances)), importances, align='center')
        plt.yticks(range(len(importances)), [f"Feature {i}" for i in indices])
        plt.title('Decision Tree Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        # Convert plot to base64 image
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return img_str
    
    def get_detailed_report(self):
        """Get a detailed performance report with all metrics"""
        if not self.metrics:
            raise ValueError("Metrics not available. Call evaluate() first.")
        
        report = {
            'model_type': 'Decision Tree',
            'parameters': {
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'criterion': self.criterion,
                'max_features': self.max_features,
                'random_state': self.random_state
            },
            'performance': {
                'accuracy': self.metrics['accuracy'],
                'per_class': self.class_metrics,
                'classification_report': self.classification_report_dict,
                'confusion_matrix': self.confusion_matrix.tolist() if self.confusion_matrix is not None else None,
            },
            'tree_specific': {
                'tree_depth': self.metrics['tree_depth'],
                'n_leaves': self.metrics['n_leaves'],
                'feature_importance': self.metrics['feature_importance']
            }
        }
        
        if 'misclassified_examples' in self.metrics:
            report['analysis'] = {
                'misclassified_count': self.metrics['misclassified_count'],
                'misclassified_examples': self.metrics['misclassified_examples']
            }
        
        return report