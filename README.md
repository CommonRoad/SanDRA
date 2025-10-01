# SanDRA
Safe large language model-based Decision making framework for automated vehicles using Reachability Analysis

## Setup
For using SanDRA with OpenAI models, you need an OpenAI API-key. Make sure to export it as environment variable named **OPENAI_API_KEY**.
If you'd rather use local models, you can follow the instructions in section **Run with local LLMs**.

For leveraging reachability analysis you need to install
* [commonroad-reach-semantic](https://gitlab.lrz.de/cps/commonroad/commonroad-reach-semantic): branch `feature/repair-new` 
(use `export CXX=/usr/bin/g++-10` before installation to use the correct compiler)

For set predictions, you need to install
* [sonia (spot)](https://gitlab.lrz.de/cps/spot-cpp): branch `master` (`python setup.py install`)

## Main scripts
There are 2 ways to test SanDRA:
1. With a [CommonRoad](https://commonroad.in.tum.de/) scenario.
2. With the [highwayenv](https://highway-env.farama.org/).

*commonroad_run.py* and  *highwayenv_run.py* illustrate how to run SanDRA decision making in either of these cases. Please make sure to prepare the seeds for highwayenv / the scenarios for CommonRoad beforehand.

## Run with local LLMs
To run SanDRA with local models, you need to follow these steps:
1. Download [Ollama](https://ollama.com/download)
2. Install Go (Recommended):
```bash
sudo apt update
sudo apt install golang-go
```
3. Download a model (We recommend to use a model with >=8B parameters to avoid problems with structured outputs):
```bash
ollama pull qwen3:8b
```
4. Start the Ollama server
```bash
ollama serve
```
