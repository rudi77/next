# AI Training Pipeline

A distributed system for managing AI model training jobs across multiple GPUs, with integrated support for ms-swift training. This system provides a web interface for submitting training jobs, monitoring GPU usage, and tracking job status.

## Features

- 🚀 Submit ms-swift training jobs with customizable hyperparameters
- 📊 Real-time GPU monitoring and status updates
- 🔄 Automatic job queuing when GPUs are occupied
- 📝 Job history and status tracking
- 🎯 Support for multi-GPU training jobs
- 🔌 Distributed architecture using RabbitMQ
- 🤖 Integrated with ms-swift for LLM and multi-modal model training

## System Architecture

The system consists of three main components:

1. **FastAPI Backend**
   - Handles job submissions
   - Manages GPU allocation
   - Provides REST API endpoints
   - Integrates with RabbitMQ for job queuing

2. **Training Worker**
   - Processes training jobs from the queue
   - Manages GPU resources
   - Executes ms-swift training commands
   - Updates job status
   - Handles training execution

3. **Streamlit Frontend**
   - User-friendly web interface
   - Real-time GPU status display
   - Advanced job submission form with ms-swift parameters
   - Training job monitoring

## Prerequisites

- Python 3.8+
- NVIDIA GPU(s) with CUDA support
- RabbitMQ Server
- NVIDIA System Management Interface (nvidia-smi)
- ms-swift 3.0.3+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-training-pipeline.git
cd ai-training-pipeline
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install and start RabbitMQ:
```bash
# For Ubuntu/Debian
sudo apt-get install rabbitmq-server
sudo systemctl start rabbitmq-server

# For macOS
brew install rabbitmq
brew services start rabbitmq

# Using Docker (recommended for local development)
docker run -d --name rabbitmq \
    -p 5672:5672 -p 15672:15672 \
    -e RABBITMQ_DEFAULT_USER=guest \
    -e RABBITMQ_DEFAULT_PASS=guest \
    rabbitmq:3-management
```

If using Docker, you can manage RabbitMQ through the web interface at `http://localhost:15672` (login with guest/guest).

## Configuration

Create a `.env` file in the project root:
```env
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASSWORD=guest
```

## Running the Application

You can run each component separately or use VS Code's launch configurations.

### Using VS Code

1. Open the project in VS Code
2. Go to the "Run and Debug" view (Ctrl+Shift+D)
3. Select "Full Stack" from the dropdown
4. Press F5 to start all components

### Manual Start

1. Start the FastAPI backend:
```bash
uvicorn app.main:app --reload --port 8080
```

2. Start the training worker:
```bash
python -m app.run_worker
```

3. Start the Streamlit frontend:
```bash
streamlit run streamlit_app.py
```

## Usage

1. Access the web interface at `http://localhost:8501`
2. Submit a training job:
   - Enter model type (e.g., "qwen2-vl-2b-instruct")
   - Enter model path (e.g., "qwen/Qwen2-VL-2B-Instruct")
   - Configure dataset paths
   - Set training parameters (epochs, batch size, etc.)
   - Select number of GPUs
   - Click "Submit Job"
3. Monitor GPU status and job progress in the respective tabs

## API Endpoints

- `POST /submit_job`: Submit a new training job
- `GET /gpu_status`: Get current GPU status
- `GET /jobs`: List all jobs
- `GET /job/{job_id}`: Get specific job status

## Project Structure

```
ai_training_pipeline/
├── app/
│   ├── main.py           # FastAPI application
│   ├── config.py         # Configuration settings
│   ├── gpu/
│   │   └── manager.py    # GPU management
│   ├── job/
│   │   └── store.py      # Job status tracking
│   ├── rabbitmq/
│   │   └── publisher.py  # RabbitMQ integration
│   ├── training/
│   │   └── swift_config.py  # ms-swift configuration
│   ├── worker/
│   │   └── training_worker.py  # Job processing
│   └── utils/
│       └── logger.py     # Logging configuration
├── logs/                 # Log files
├── streamlit_app.py      # Web interface
├── requirements.txt      # Dependencies
└── README.md
```

## Logging

Logs are stored in the `logs/` directory with separate files for each component:
- `backend_YYYYMMDD.log`
- `worker_YYYYMMDD.log`
- `streamlit_YYYYMMDD.log`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
```