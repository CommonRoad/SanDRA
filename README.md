# SanDRA
Safe large language model-based Decision making framework for automated vehicles using Reachability Analysis

## Run locally
To run SanDRA with local models, you need to follow these steps:
1. Download [Ollama](https://ollama.com/download)
2. Install Go (Recommended):
```bash
sudo apt update
sudo apt install golang-go
```
3. Download a model (We recommend to use a model with >8B parameters to avoid problems with structured outputs):
```bash
ollama pull qwen3:8b
```
4. Start the Ollama server
```bash
ollama serve
```