# Text processing with IndoBERT
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import re
import os
import json
from datetime import datetime

class IndoBERTProcessor:
    def __init__(self, cache_dir=None, max_length=128):
        """
        Initialize the IndoBERT processor for text embedding
        
        Parameters:
        - cache_dir: Directory to store embedding cache
        - max_length: Maximum token length for sequences
        """
        print("Loading IndoBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
        self.model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
        print("IndoBERT model loaded successfully!")
        
        self.max_length = max_length
        self.embedding_cache = {}
        self.cache_file = os.path.join(cache_dir, 'embedding_cache.json') if cache_dir else None
        
        # Load cache if it exists
        if self.cache_file and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.embedding_cache = json.load(f)
                print(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                print(f"Error loading embedding cache: {str(e)}")
                self.embedding_cache = {}
    
    def preprocess_text(self, text):
        """Clean and preprocess text before embedding"""
        if not text or text is None:
            return ""
        
        text = str(text).lower()
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_embedding(self, text):
        """Get text embedding with caching"""
        preprocessed_text = self.preprocess_text(text)
        
        # Check if embedding is in cache
        if preprocessed_text in self.embedding_cache:
            return np.array(self.embedding_cache[preprocessed_text])
        
        # If not in cache, compute new embedding
        inputs = self.tokenizer(preprocessed_text, 
                               return_tensors="pt", 
                               truncation=True, 
                               padding=True, 
                               max_length=self.max_length)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use CLS token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        
        # Save to cache
        self.embedding_cache[preprocessed_text] = cls_embedding.tolist()
        
        # Save cache to disk periodically
        if self.cache_file and len(self.embedding_cache) % 10 == 0:
            self.save_cache()
        
        return cls_embedding
    
    def get_embeddings_batch(self, texts, batch_size=32):
        """Process a batch of texts to embeddings"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Check cache first
            batch_embeddings = []
            uncached_indices = []
            uncached_texts = []
            
            for j, text in enumerate(batch_texts):
                preprocessed = self.preprocess_text(text)
                if preprocessed in self.embedding_cache:
                    batch_embeddings.append(np.array(self.embedding_cache[preprocessed]))
                else:
                    # Mark for processing
                    uncached_indices.append(j)
                    uncached_texts.append(preprocessed)
            
            # Process uncached texts
            if uncached_texts:
                inputs = self.tokenizer(uncached_texts, 
                                       return_tensors="pt", 
                                       truncation=True, 
                                       padding=True, 
                                       max_length=self.max_length)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                new_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                
                # Insert new embeddings into result and update cache
                for j, (idx, embedding) in enumerate(zip(uncached_indices, new_embeddings)):
                    # Find correct position to insert
                    insert_pos = 0
                    for k in range(len(batch_embeddings)):
                        if k < idx:
                            insert_pos += 1
                    
                    batch_embeddings.insert(insert_pos, embedding)
                    
                    # Update cache
                    self.embedding_cache[uncached_texts[j]] = embedding.tolist()
            
            embeddings.extend(batch_embeddings)
            
            # Save cache after each batch
            if self.cache_file:
                self.save_cache()
        
        return embeddings
    
    def save_cache(self):
        """Save embedding cache to disk"""
        if not self.cache_file:
            return
            
        try:
            # Create temp file first to avoid corrupting cache on write error
            temp_file = f"{self.cache_file}.temp"
            with open(temp_file, 'w') as f:
                json.dump(self.embedding_cache, f)
            
            # Rename temp file to actual cache file
            os.replace(temp_file, self.cache_file)
            
            print(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            print(f"Error saving embedding cache: {str(e)}")
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.embedding_cache = {}
        if self.cache_file and os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
                print("Cache file removed")
            except Exception as e:
                print(f"Error removing cache file: {str(e)}")