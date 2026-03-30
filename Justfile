model_url := "https://huggingface.co/unsloth/gemma-3n-E4B-it-GGUF/resolve/main/gemma-3n-E4B-it-Q4_K_M.gguf?download=true"
model_file := "gemma-3n-E4B-it-Q4_K_M.gguf"
port := "8000"
ctx := "32768"

default:
    @just --list

setup:
    @echo "Checking for model..."
    @if [ ! -f {{model_file}} ]; then \
        echo "Downloading {{model_file}}..."; \
        aria2c -x 16 -s 16 -o {{model_file}} "{{model_url}}"; \
    else \
        echo "Model already exists."; \
    fi

serve:
    @echo "Starting llama-server with model: {{model_file}}"
    llama-server -m ./{{model_file}} --port {{port}} -c {{ctx}} -fa on --cache-ram 0 --repeat-penalty 1.0 --min-p 0.01 --top-k 64 --top-p 0.95

run input_file="corpus/literature.json":
    python main.py --input "{{input_file}}" --timeout 0 --iterations 1 --refinement-iterations 3 --preserve-last-n-messages 2

vectorise:
    python vectorise_dictionary.py

# Experimental
aggregate input_csv:
    python aggregator.py --input "{{input_csv}}"