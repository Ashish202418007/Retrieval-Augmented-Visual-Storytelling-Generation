# RAVSG: Story-Image Bidirectional Generator
 Studio Ghibli-inspired bidirectional story-image generation using SOTA open-source models with RAG enhancement.

### Authors : 
Ashish Kar(202418007)

Misty Roy(202418033)

Yashraj Sinh(202418064)

Pragnya Dandavate(202418065)

## Architecture

```
┌─────────────┐
│   FastAPI   │ ← REST API
└──────┬──────┘
       │
       ├──→ Redis Queue ──→ Story Worker (LLaVA)
       │                   └─→ CLIP Embeddings
       │                       └─→ FAISS RAG
       │
       └──→ Redis Queue ──→ Image Worker (Stable Diffusion)
                           └─→ CLIP Embeddings
                               └─→ FAISS RAG
```

## Models Used

* **Embeddings**: `openai/clip-vit-base-patch16` (512D multimodal embeddings)
* **Image→Story**: `llava-hf/llava-1.5-7b-hf`
* **Story→Image**: `runwayml/stable-diffusion-v1-5`
* **Vector DB**: FAISS (Flat or IVFFlat)

## Quick Start

### Local Setup

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Redis
redis-server

# Start API server
python api/server.py

# Start workers (in separate terminals)
python workers/generation_worker.py story
python workers/generation_worker.py image
```

<!-- 
### Docker Setup

```bash
# Build and run all services
docker-compose up --build

# Scale workers
docker-compose up --scale story_worker=2 --scale image_worker=2 
```-->

## API Usage

### Generate Story from Image

```bash
# Submit job
curl -X POST http://localhost:8000/generate-story \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/png;base64,..."}'

# Response: {"job_id": "abc-123", "status": "pending"}

# Poll for results
curl http://localhost:8000/job/abc-123

# Response when complete:
{
  "job_id": "abc-123",
  "status": "completed",
  "result": {"story": "Once upon a time..."}
}
```

### Generate Image from Story

```bash
# Submit job
curl -X POST http://localhost:8000/generate-image \
  -H "Content-Type: application/json" \
  -d '{"story": "A magical forest at sunset..."}'

# Poll for results
curl http://localhost:8000/job/xyz-456

# Response:
{
  "job_id": "xyz-456",
  "status": "completed",
  "result": {"image": "data:image/png;base64,..."}
}
```

## Configuration

Edit `config/settings.py` or create `.env`:

```env
# Models
VISION_LANGUAGE_MODEL=llava-hf/llava-1.5-7b-hf
TEXT_TO_IMAGE_MODEL=runwayml/stable-diffusion-v1-5
EMBEDDING_MODEL=openai/clip-vit-base-patch16

# Performance
DEVICE=cuda
USE_FP16=true
USE_8BIT=false

# RAG
EMBEDDING_DIM=512
TOP_K_RETRIEVAL=5
FAISS_INDEX_TYPE=Flat   # Options: Flat, IVFFlat
FAISS_NLIST=100

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Langfuse (optional)
ENABLE_LANGFUSE=false
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
```

## GPU Memory Management

For RTX-class GPUs:

1. **Single worker per GPU**: Each worker loads one model at a time
2. **Lazy loading**: Models loaded on-demand, unloaded after use
3. **FP16 precision**: Reduces memory footprint
4. **Optional 8-bit**: For memory-constrained environments

Approximate memory usage:

* CLIP ViT-B/16: ~1GB
* LLaVA 1.5 7B: ~13GB
* Stable Diffusion v1.5: ~6GB

## Project Structure

```
root
├── backend
│   ├── api/
│   │   └── server.py              # FastAPI endpoints
│   ├── config/
│   │   └── settings.py            # Configuration
│   ├── core/
│   │   ├── model_manager.py       # Model loading & inference
│   │   ├── rag_engine.py          # FAISS vector database
│   │   ├── queue_manager.py       # Redis job queue
│   │   └── observability.py       # Langfuse tracing
│   ├── workers/
│   │   └── generation_worker.py   # Async job processor
│   ├── utils/
│   │   └── image_utils.py         # Image encoding/decoding
│   ├── database/                  # FAISS indices
│   ├── models/                    # Model weights cache
│   ├── media/                     # Generated content
│   └── logs/                      # Application logs
│
├── frontend
│   └── app.html                   # Frontend UI (static or SPA entry)
│
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules

```

## RAG System

The system maintains two FAISS indices:

1. **Story Index**: Stores CLIP embeddings of generated stories
2. **Image Index**: Stores CLIP embeddings and metadata of generated images

During generation:

* Retrieves top-K similar examples
* Provides context to generation models
* Improves consistency and quality over time

## Observability

Enable Langfuse for production monitoring:

```python
# config/settings.py
ENABLE_LANGFUSE = True
LANGFUSE_PUBLIC_KEY = "pk-..."
LANGFUSE_SECRET_KEY = "sk-..."
```

Tracks:

* Generation latency
* GPU memory usage
* Embedding extraction
* RAG retrieval performance
* Error rates

## Performance Tips

1. **Batch processing**: Queue multiple jobs for efficiency
2. **Model caching**: Keep frequently-used models loaded
3. **FAISS tuning**: Adjust `FAISS_NLIST` based on dataset size
4. **Worker scaling**: Add more workers for parallel processing
5. **GPU pinning**: Use separate GPUs for workers if available

## Troubleshooting

**Out of Memory**:

```bash
# Enable 8-bit quantization
USE_8BIT=true

# Or reduce image size
IMAGE_SIZE=512
```

**Slow generation**:

```bash
# Reduce inference steps
NUM_INFERENCE_STEPS=20

# Scale workers
docker-compose up --scale story_worker=3
```

**Redis connection failed**:

```bash
# Check Redis status
redis-cli ping

# Restart Redis
redis-server --daemonize yes
```

## 🧪 Testing

```bash
# Test CLIP embeddings
python -c "from models.model_manager import ModelManager; m = ModelManager(); m.load_clip(); print('CLIP loaded!')"

# Test RAG
python -c "from rag.rag_engine import RAGEngine; from models.model_manager import ModelManager; m = ModelManager(); m.load_clip(); r = RAGEngine(m.clip_model, m.clip_processor, m.device); r.build_index(); print(f'RAG loaded with {len(r.examples)} examples')"
```

## 🤝 Contributing

To add new features:

1. Models: Edit `models/model_manager.py`
2. RAG logic: Edit `rag/rag_engine.py`
3. API endpoints: Edit `main.py`

## 📝 License

MIT License - Free for research and commercial use

---

**Questions?** Check FastAPI auto-docs at `/docs` endpoint!


<!-- Setting: The desolate, red-hued surface of the moon.
Subjects: A lone astronaut in a retro-futuristic spacesuit carefully plants a single, bright red flower in the lunar soil.
Atmosphere: A mix of isolation and hope. The Earth is visible as a blue and white marble in the inky black sky.
Keywords: Astronaut, planting flower, red moon, desolate, Earth in sky, hope, isolated, retro-futuristic. -->
