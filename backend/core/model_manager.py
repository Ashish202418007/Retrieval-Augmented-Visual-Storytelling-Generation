import torch
from transformers import (
    AutoProcessor, 
    LlavaNextForConditionalGeneration,
    Blip2Processor,
    Blip2Model
)
from diffusers import StableDiffusion3Pipeline
from PIL import Image
import logging
from typing import Optional
import gc

from config.settings import settings

logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self.device = settings.DEVICE
        self.dtype = torch.float16 if settings.USE_FP16 else torch.float32
        logger.info(f"Device: {self.device}, Dtype: {self.dtype}")
        
        # Model placeholders
        self.embedding_model: Optional[Blip2Model] = None
        self.embedding_processor: Optional[Blip2Processor] = None
        self.vlm_model: Optional[LlavaNextForConditionalGeneration] = None
        self.vlm_processor: Optional[AutoProcessor] = None
        self.t2i_pipeline: Optional[StableDiffusion3Pipeline] = None
        
        self._initialized = True
    
    def _clear_cache(self):
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def load_embedding_model(self):
        """Load BLIP-2 for high-quality multimodal embeddings"""
        if self.embedding_model is not None:
            return
        
        logger.info("Loading embedding model (BLIP-2)...")
        self.embedding_processor = Blip2Processor.from_pretrained(
            settings.EMBEDDING_MODEL,
            cache_dir=settings.MODELS_DIR
        )
        self.embedding_model = Blip2Model.from_pretrained(
            settings.EMBEDDING_MODEL,
            torch_dtype=self.dtype,
            cache_dir=settings.MODELS_DIR
        ).to(self.device)
        self.embedding_model.eval()
        logger.info("Embedding model loaded")
    
    def load_vlm(self):
        """Load LLaVA-NeXT for image-to-story"""
        if self.vlm_model is not None:
            return
        
        logger.info("Loading Vision-Language model...")
        self.vlm_processor = AutoProcessor.from_pretrained(
            settings.VISION_LANGUAGE_MODEL,
            cache_dir=settings.MODELS_DIR
        )
        
        load_kwargs = {
            "torch_dtype": self.dtype,
            "cache_dir": settings.MODELS_DIR,
            "low_cpu_mem_usage": True
        }
        
        if settings.USE_8BIT:
            load_kwargs["load_in_8bit"] = True
        
        self.vlm_model = LlavaNextForConditionalGeneration.from_pretrained(
            settings.VISION_LANGUAGE_MODEL,
            **load_kwargs
        )
        
        if not settings.USE_8BIT:
            self.vlm_model = self.vlm_model.to(self.device)
        
        self.vlm_model.eval()
        logger.info("VLM loaded")
    
    def load_t2i(self):
        """Load Stable Diffusion 3.5 Turbo for story-to-image"""
        if self.t2i_pipeline is not None:
            return
        
        logger.info("Loading Text-to-Image pipeline...")
        self.t2i_pipeline = StableDiffusion3Pipeline.from_pretrained(
            settings.TEXT_TO_IMAGE_MODEL,
            torch_dtype=self.dtype,
            cache_dir=settings.MODELS_DIR
        ).to(self.device)
        logger.info("T2I pipeline loaded")
    
    def unload_vlm(self):
        """Free VLM memory"""
        if self.vlm_model is not None:
            del self.vlm_model
            del self.vlm_processor
            self.vlm_model = None
            self.vlm_processor = None
            self._clear_cache()
    
    def unload_t2i(self):
        """Free T2I memory"""
        if self.t2i_pipeline is not None:
            del self.t2i_pipeline
            self.t2i_pipeline = None
            self._clear_cache()
    
    @torch.no_grad()
    def get_image_embedding(self, image: Image.Image) -> torch.Tensor:
        """Extract image embedding using BLIP-2"""
        inputs = self.embedding_processor(
            images=image, 
            return_tensors="pt"
        ).to(self.device, self.dtype)
        
        outputs = self.embedding_model.get_image_features(**inputs)
        return outputs.cpu().float()
    
    @torch.no_grad()
    def get_text_embedding(self, text: str) -> torch.Tensor:
        """Extract text embedding using BLIP-2"""
        inputs = self.embedding_processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.device)
        
        outputs = self.embedding_model.get_text_features(**inputs)
        return outputs.cpu().float()
    
    @torch.no_grad()
    def generate_story(self, image: Image.Image, context: str) -> str:
        """Generate story from image with RAG context"""
        prompt = f"""[INST] <image>
You are a creative storyteller. Based on this image and inspired by these examples:

{context}

Write an engaging short story (3-4 paragraphs) about this scene. Use vivid descriptions and create an emotional narrative. [/INST]"""
        
        inputs = self.vlm_processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        output_ids = self.vlm_model.generate(
            **inputs,
            max_new_tokens=settings.MAX_NEW_TOKENS,
            do_sample=True,
            temperature=settings.TEMPERATURE,
            top_p=settings.TOP_P,
            pad_token_id=self.vlm_processor.tokenizer.pad_token_id
        )
        
        story = self.vlm_processor.decode(
            output_ids[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return story
    
    @torch.no_grad()
    def generate_image(self, prompt: str, style_hints: str = "") -> Image.Image:
        """Generate image from story with style guidance"""
        full_prompt = f"{prompt}"
        if style_hints:
            full_prompt += f", {style_hints}"
        full_prompt += ", masterpiece, highly detailed, 8k, cinematic"
        
        negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy"
        
        image = self.t2i_pipeline(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=settings.NUM_INFERENCE_STEPS,
            guidance_scale=settings.GUIDANCE_SCALE,
            height=settings.IMAGE_SIZE,
            width=settings.IMAGE_SIZE
        ).images[0]
        
        return image

model_manager = ModelManager()