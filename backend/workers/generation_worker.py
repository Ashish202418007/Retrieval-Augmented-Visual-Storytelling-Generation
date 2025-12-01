import logging
import time
from PIL import Image

from core.model_manager import model_manager
from core.rag_engine import rag_engine
from core.queue_manager import queue_manager, JobStatus
from core.observability import observer
from utils.image_utils import decode_base64_image, encode_image_to_base64
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GenerationWorker:
    def __init__(self, worker_type: str):
        self.worker_type = worker_type  # 'story' or 'image'
        logger.info(f"Initialized {worker_type} worker")
    
    def process_story_job(self, job_id: str, job_data: dict):
        """Process image-to-story generation"""
        try:
            queue_manager.update_job_status(job_id, JobStatus.RUNNING)
            
            # Decode image
            image = decode_base64_image(job_data["image"])
            
            # Get embedding
            image_embedding = model_manager.get_image_embedding(image)
            observer.log_embedding("image", image_embedding.shape[1])
            
            # RAG retrieval
            similar_stories = rag_engine.retrieve_stories(
                image_embedding, 
                k=settings.TOP_K_RETRIEVAL
            )
            observer.log_rag_retrieval("story", settings.TOP_K_RETRIEVAL, len(similar_stories))
            
            # Build context
            context = "\n\n".join([f"Example {i+1}:\n{s}" 
                                  for i, s in enumerate(similar_stories)])
            
            # Generate story
            story = model_manager.generate_story(image, context)
            
            # Store in RAG
            story_embedding = model_manager.get_text_embedding(story)
            rag_engine.add_story(story_embedding, story, {"job_id": job_id})
            
            # Update job
            queue_manager.update_job_status(
                job_id, 
                JobStatus.COMPLETED, 
                result={"story": story}
            )
            
            logger.info(f"Completed story job {job_id}")
            
        except Exception as e:
            logger.error(f"Story job {job_id} failed: {e}", exc_info=True)
            queue_manager.update_job_status(
                job_id, 
                JobStatus.FAILED, 
                error=str(e)
            )
    
    def process_image_job(self, job_id: str, job_data: dict):
        """Process story-to-image generation"""
        try:
            queue_manager.update_job_status(job_id, JobStatus.RUNNING)
            
            story = job_data["story"]
            
            # Get embedding
            text_embedding = model_manager.get_text_embedding(story)
            observer.log_embedding("text", text_embedding.shape[1])
            
            # RAG retrieval
            similar_metadata = rag_engine.retrieve_image_metadata(
                text_embedding,
                k=settings.TOP_K_RETRIEVAL
            )
            observer.log_rag_retrieval("image", settings.TOP_K_RETRIEVAL, len(similar_metadata))
            
            # Extract style hints
            style_hints = ", ".join([
                meta.get("style", "") 
                for meta in similar_metadata 
                if meta.get("style")
            ])
            
            # Generate image
            image = model_manager.generate_image(story[:500], style_hints)
            
            # Encode to base64
            image_b64 = encode_image_to_base64(image)
            
            # Store in RAG
            image_embedding = model_manager.get_image_embedding(image)
            rag_engine.add_image(image_embedding, {
                "job_id": job_id,
                "style": style_hints,
                "story_excerpt": story[:200]
            })
            
            # Update job
            queue_manager.update_job_status(
                job_id,
                JobStatus.COMPLETED,
                result={"image": image_b64}
            )
            
            logger.info(f"Completed image job {job_id}")
            
        except Exception as e:
            logger.error(f"Image job {job_id} failed: {e}", exc_info=True)
            queue_manager.update_job_status(
                job_id,
                JobStatus.FAILED,
                error=str(e)
            )
    
    def run(self):
        """Main worker loop"""
        logger.info(f"Starting {self.worker_type} worker...")
        
        # Load models
        model_manager.load_embedding_model()
        rag_engine.load_or_create_indices()
        
        if self.worker_type == "story":
            model_manager.load_vlm()
        else:
            model_manager.load_t2i()
        
        logger.info("Worker ready, waiting for jobs...")
        
        while True:
            try:
                # Get next job
                job_id = queue_manager.get_next_job(self.worker_type)
                
                if not job_id:
                    continue
                
                job_data = queue_manager.get_job(job_id)
                if not job_data:
                    continue
                
                logger.info(f"Processing {self.worker_type} job {job_id}")
                
                # Process job
                if self.worker_type == "story":
                    self.process_story_job(job_id, job_data["data"])
                else:
                    self.process_image_job(job_id, job_data["data"])
                
            except KeyboardInterrupt:
                logger.info("Worker shutting down...")
                break
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
                time.sleep(1)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python generation_worker.py [story|image]")
        sys.exit(1)
    
    worker_type = sys.argv[1]
    if worker_type not in ["story", "image"]:
        print("Worker type must be 'story' or 'image'")
        sys.exit(1)
    
    worker = GenerationWorker(worker_type)
    worker.run()