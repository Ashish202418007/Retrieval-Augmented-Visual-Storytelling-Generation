import faiss
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import torch

from config.settings import settings

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        self.story_index: faiss.Index = None
        self.image_index: faiss.Index = None
        self.story_data: List[Dict] = []
        self.image_data: List[Dict] = []
        
        self.story_db_path = settings.DATABASE_DIR / "stories.json"
        self.image_db_path = settings.DATABASE_DIR / "images.json"
        self.story_index_path = settings.DATABASE_DIR / "story_index.faiss"
        self.image_index_path = settings.DATABASE_DIR / "image_index.faiss"
    
    def _create_index(self, dimension: int) -> faiss.Index:
        """Create FAISS index with GPU support"""
        if settings.FAISS_INDEX_TYPE == "IVFFlat":
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(
                quantizer, 
                dimension, 
                settings.FAISS_NLIST, 
                faiss.METRIC_L2
            )
        else:
            index = faiss.IndexFlatL2(dimension)
        
        # Move to GPU if available
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        return index
    
    def load_or_create_indices(self):
        """Load existing indices or create new ones"""
        dimension = settings.EMBEDDING_DIM
        
        # Story index
        if self.story_index_path.exists():
            logger.info("Loading story index...")
            self.story_index = faiss.read_index(str(self.story_index_path))
            if faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                self.story_index = faiss.index_cpu_to_gpu(res, 0, self.story_index)
        else:
            logger.info("Creating new story index...")
            self.story_index = self._create_index(dimension)
        
        # Image index
        if self.image_index_path.exists():
            logger.info("Loading image index...")
            self.image_index = faiss.read_index(str(self.image_index_path))
            if faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                self.image_index = faiss.index_cpu_to_gpu(res, 0, self.image_index)
        else:
            logger.info("Creating new image index...")
            self.image_index = self._create_index(dimension)
        
        # Load metadata
        if self.story_db_path.exists():
            with open(self.story_db_path, 'r') as f:
                self.story_data = json.load(f)
        
        if self.image_db_path.exists():
            with open(self.image_db_path, 'r') as f:
                self.image_data = json.load(f)
        
        logger.info(f"Loaded {len(self.story_data)} stories, {len(self.image_data)} images")
    
    def add_story(self, embedding: torch.Tensor, story: str, metadata: Dict = None):
        """Add story to RAG database"""
        emb_np = embedding.numpy().astype('float32')
        
        if self.story_index.ntotal == 0 and isinstance(self.story_index, faiss.IndexIVFFlat):
            # Train index if empty
            self.story_index.train(emb_np)
        
        self.story_index.add(emb_np)
        
        self.story_data.append({
            "story": story,
            "metadata": metadata or {}
        })
        
        self._save_stories()
    
    def add_image(self, embedding: torch.Tensor, metadata: Dict):
        """Add image metadata to RAG database"""
        emb_np = embedding.numpy().astype('float32')
        
        if self.image_index.ntotal == 0 and isinstance(self.image_index, faiss.IndexIVFFlat):
            self.image_index.train(emb_np)
        
        self.image_index.add(emb_np)
        
        self.image_data.append(metadata)
        
        self._save_images()
    
    def retrieve_stories(self, query_embedding: torch.Tensor, k: int = 5) -> List[str]:
        """Retrieve top-k similar stories"""
        if self.story_index.ntotal == 0:
            return []
        
        query_np = query_embedding.numpy().astype('float32')
        distances, indices = self.story_index.search(query_np, min(k, self.story_index.ntotal))
        
        return [self.story_data[idx]["story"] for idx in indices[0] if idx < len(self.story_data)]
    
    def retrieve_image_metadata(self, query_embedding: torch.Tensor, k: int = 5) -> List[Dict]:
        """Retrieve top-k similar image metadata"""
        if self.image_index.ntotal == 0:
            return []
        
        query_np = query_embedding.numpy().astype('float32')
        distances, indices = self.image_index.search(query_np, min(k, self.image_index.ntotal))
        
        return [self.image_data[idx] for idx in indices[0] if idx < len(self.image_data)]
    
    def _save_stories(self):
        """Save story metadata to disk"""
        with open(self.story_db_path, 'w') as f:
            json.dump(self.story_data, f, indent=2)
        
        # Save index to CPU first
        if faiss.get_num_gpus() > 0:
            cpu_index = faiss.index_gpu_to_cpu(self.story_index)
            faiss.write_index(cpu_index, str(self.story_index_path))
        else:
            faiss.write_index(self.story_index, str(self.story_index_path))
    
    def _save_images(self):
        """Save image metadata to disk"""
        with open(self.image_db_path, 'w') as f:
            json.dump(self.image_data, f, indent=2)
        
        if faiss.get_num_gpus() > 0:
            cpu_index = faiss.index_gpu_to_cpu(self.image_index)
            faiss.write_index(cpu_index, str(self.image_index_path))
        else:
            faiss.write_index(self.image_index, str(self.image_index_path))

rag_engine = RAGEngine()