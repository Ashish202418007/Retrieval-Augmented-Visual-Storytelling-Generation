import time
import logging
from functools import wraps
from typing import Optional, Dict, Any
import torch

from config.settings import settings

logger = logging.getLogger(__name__)

# Conditional Langfuse import
if settings.ENABLE_LANGFUSE:
    try:
        from langfuse import Langfuse
        langfuse_client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST
        )
    except Exception as e:
        logger.warning(f"Langfuse init failed: {e}")
        langfuse_client = None
else:
    langfuse_client = None

class Observer:
    @staticmethod
    def get_gpu_memory() -> Dict[str, float]:
        """Get current GPU memory usage"""
        if not torch.cuda.is_available():
            return {}
        
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9
        }
    
    @staticmethod
    def trace_generation(generation_type: str):
        """Decorator to trace generation calls"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                initial_memory = Observer.get_gpu_memory()
                
                trace_id = None
                if langfuse_client:
                    trace = langfuse_client.trace(
                        name=f"{generation_type}_generation"
                    )
                    trace_id = trace.id
                
                try:
                    result = func(*args, **kwargs)
                    
                    duration = time.time() - start_time
                    final_memory = Observer.get_gpu_memory()
                    
                    metrics = {
                        "duration_seconds": duration,
                        "initial_memory": initial_memory,
                        "final_memory": final_memory,
                        "generation_type": generation_type
                    }
                    
                    logger.info(f"{generation_type} generation: {duration:.2f}s, "
                              f"GPU: {final_memory.get('allocated_gb', 0):.2f}GB")
                    
                    if langfuse_client and trace_id:
                        langfuse_client.score(
                            trace_id=trace_id,
                            name="generation_time",
                            value=duration
                        )
                    
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    logger.error(f"{generation_type} failed after {duration:.2f}s: {e}")
                    
                    if langfuse_client and trace_id:
                        langfuse_client.score(
                            trace_id=trace_id,
                            name="error",
                            value=str(e)
                        )
                    
                    raise
            
            return wrapper
        return decorator
    
    @staticmethod
    def log_embedding(embedding_type: str, dimension: int):
        """Log embedding extraction"""
        if langfuse_client:
            langfuse_client.generation(
                name=f"{embedding_type}_embedding",
                metadata={"dimension": dimension}
            )
        
        logger.debug(f"Extracted {embedding_type} embedding (dim={dimension})")
    
    @staticmethod
    def log_rag_retrieval(content_type: str, k: int, num_results: int):
        """Log RAG retrieval"""
        if langfuse_client:
            langfuse_client.generation(
                name=f"rag_retrieval_{content_type}",
                metadata={"k": k, "results": num_results}
            )
        
        logger.debug(f"RAG retrieved {num_results}/{k} {content_type} examples")

observer = Observer()