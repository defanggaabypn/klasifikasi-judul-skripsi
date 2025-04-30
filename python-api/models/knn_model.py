# KNN Model implementation for text classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class KNNModel:
    def __init__(self, n_neighbors=3, weights='uniform', algorithm='auto', metric='minkowski'):
        """
        Initialize the KNN model with customizable parameters
        
        Parameters:
        - n_neighbors: Number of neighbors to use for kNN
        - weights: Weight function used in prediction ('uniform' or 'distance')
        - algorithm: Algorithm used to compute nearest neighbors
        - metric: Distance metric to use
        """
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            metric=metric
        )
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.metric = metric
        self.trained = False
        self.metrics = {}
        self.class_metrics = {}
        self.confusion_matrix = None
        self.classification_report_dict = None
        
    def fit(self, X_train, y_train):
        """Train the KNN model on the given data"""
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
    
    def kneighbors(self, X, n_neighbors=None):
        """Find the K-neighbors of a point"""
        if not self.trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        return self.model.kneighbors(X, n_neighbors=n_neighbors or self.n_neighbors)
    
    def get_neighbor_details(self, X, indices, distances, y_train):
        """Get details about the nearest neighbors for a sample"""
        neighbors = []
        for i, (idx_array, dist_array) in enumerate(zip(indices, distances)):
            sample_neighbors = []
            for j, (idx, dist) in enumerate(zip(idx_array, dist_array)):
                sample_neighbors.append({
                    'index': int(idx),
                    'distance': float(dist),
                    'label': y_train[idx],
                    'similarity': 1.0 / (1.0 + dist)  # Convert distance to similarity
                })
            neighbors.append(sample_neighbors)
        return neighbors
    
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
                'support': int(support[i]),
                'neighbors': {}  # Will hold neighbor info for each misclassified instance
            }
        
        # Analyze misclassifications
        misclassified_indices = [i for i, (true, pred) in enumerate(zip(y_test, y_pred)) if true != pred]
        if misclassified_indices:
            X_misclassified = [X_test[i] for i in misclassified_indices]
            y_true_misclassified = [y_test[i] for i in misclassified_indices]
            
            # Get nearest neighbors for misclassified instances
            distances, indices = self.kneighbors(X_misclassified)
            
            # Store neighbor info
            for i, (idx, true_label) in enumerate(zip(misclassified_indices, y_true_misclassified)):
                self.metrics.setdefault('misclassified', []).append({
                    'index': idx,
                    'true_label': true_label,
                    'predicted_label': y_pred[idx],
                    'neighbors': self.get_neighbor_details([X_test[idx]], 
                                                       indices[i].reshape(1, -1), 
                                                       distances[i].reshape(1, -1), 
                                                       self.y_train)[0]
                })
        
        # Add more detailed KNN-specific metrics
        self.metrics['average_distance_within_class'] = self._calculate_avg_distance_within_class()
        self.metrics['class_distribution'] = self._calculate_class_distribution()
        
        return self.metrics
    
    def _calculate_avg_distance_within_class(self):
        """Calculate average distance between points within the same class"""
        class_distances = {}
        classes = sorted(list(set(self.y_train)))
        
        for cls in classes:
            class_indices = [i for i, label in enumerate(self.y_train) if label == cls]
            if len(class_indices) <= 1:
                class_distances[cls] = 0.0
                continue
                
            # Take a sample of points if class is large
            if len(class_indices) > 50:
                import random
                class_indices = random.sample(class_indices, 50)
                
            X_class = [self.X_train[i] for i in class_indices]
            
            # Calculate average distance between all pairs of points
            total_dist = 0.0
            count = 0
            for i in range(len(X_class)):
                distances, _ = self.kneighbors([X_class[i]], n_neighbors=min(self.n_neighbors, len(X_class)))
                # Skip first distance (distance to self is 0)
                if len(distances[0]) > 1:
                    total_dist += sum(distances[0][1:])
                    count += len(distances[0]) - 1
            
            class_distances[cls] = total_dist / count if count > 0 else 0.0
        
        return class_distances
    
    def _calculate_class_distribution(self):
        """Calculate class distribution in training data"""
        class_counts = {}
        for label in self.y_train:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # Convert to percentages
        total = len(self.y_train)
        class_distribution = {cls: count / total for cls, count in class_counts.items()}
        
        return class_distribution
    
    def generate_confusion_matrix_image(self, labels=None):
        """Generate a confusion matrix visualization as base64 image"""
        if self.confusion_matrix is None:
            raise ValueError("Confusion matrix not available. Call evaluate() first.")
        
        plt.figure(figsize=(8, 6))
        plt.imshow(self.confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix - KNN')
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
    
    def get_detailed_report(self):
        """Get a detailed performance report with all metrics"""
        if not self.metrics:
            raise ValueError("Metrics not available. Call evaluate() first.")
        
        report = {
            'model_type': 'K-Nearest Neighbors',
            'parameters': {
                'n_neighbors': self.n_neighbors,
                'weights': self.weights,
                'algorithm': self.algorithm,
                'metric': self.metric
            },
            'performance': {
                'accuracy': self.metrics['accuracy'],
                'per_class': self.class_metrics,
                'classification_report': self.classification_report_dict,
                'confusion_matrix': self.confusion_matrix.tolist() if self.confusion_matrix is not None else None,
            },
            'knn_specific': {
                'avg_distance_within_class': self.metrics.get('average_distance_within_class', {}),
                'class_distribution': self.metrics.get('class_distribution', {})
            }
        }
        
        if 'misclassified' in self.metrics:
            report['analysis'] = {
                'misclassified_count': len(self.metrics['misclassified']),
                'misclassified_examples': self.metrics['misclassified'][:5]  # Limit to 5 examples
            }
        
        return report