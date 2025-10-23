# AI Translation Quality Analysis Framework

This repository contains the scripts and methodology for a thesis project analyzing the quality of translations produced by a local Large Language Model (LLM). The framework uses a multi-agent, iterative workflow to generate, evaluate, and refine translations, providing a rich dataset for qualitative analysis.

## Requirements

- Python 3.11 or higher (Python 3.14 is untested as of writing).
- Python package: `aiohttp`
- llama.cpp (llama-server)
- LLM Model: A GGUF-compatible model. The experiments for the thesis were conducted using `unsloth/gemma-3n-E4B-it-GGUF`.

## Quick Setup (Windows)

These instructions use the [Scoop](https://scoop.sh) package manager for a straightforward setup on Windows.

1. Install Scoop from the official [website](https://scoop.sh).
2. Add `versions` bucket:

```sh
scoop bucket add versions
```

3. Install Python and llama.cpp (Vulkan, or any other variant that your hardware supports):

```sh
scoop install python311 llama.cpp-vulkan
```

4. Install Python dependencies:

```sh
python311 -m pip install aiohttp
```

5. Download the Model: Download the [unsloth/gemma-3n-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-3n-E4B-it-GGUF) model from Hugging Face and place it in the root directory of this project (we use specifically Q4_K_M, though other variants should work fine too).

> [!NOTE] For other platforms (Linux/macOS): You will need to install Python 3.11 and compile llama.cpp from source according to their official documentation.

## Usage

The experiment is run in three stages: preparing the data, running the LLM server, and executing the main Python script.

##### Step 1: Prepare the corpus

Create a JSON file in the corpus directory (e.g., [corpus/literature.json](corpus/literature.json)). It must have `source_lang`, `target_lang`, `type`, and lastly `texts` containing an array of texts.

##### Step 2: Run the LLM Server

Open a terminal in the project's root directory and run the `llama-server`. This will load the model into memory and open an API endpoint for the script to use.

```sh
# Example command to run llama-server
llama-server -m .\gemma-3n-E4B-it.gguf --port 8000 -c 32768 -fa on --cache-ram 4096 --repeat-penalty 1.0 --min-p 0.01 --top-k 64 --top-p 0.95 --no-webui
```

##### Step 3: Run the Experiment Script

Open a second terminal in the project's root directory. Run the `evaluator_optimizer.py` script, pointing it to your input corpus.

```sh
python evaluator_optimizer.py --input "corpus/literature.json" --timeout 0 --simple-evaluator --iterations 5 --refinement-iterations 5 --cache-prompt --omit-roles --preserve-history
```

## Output

The script will generate a .csv file in a newly created `evaluator_optimizer_attempts` directory. This CSV file contains a detailed log of every attempt in the refinement loop, allowing for a thorough analysis of the AI's behavior.
