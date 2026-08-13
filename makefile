SCRIPT ?= subgen.py
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

DEPS := faster-whisper huggingface-hub tqdm nvidia-cublas-cu12 nvidia-cudnn-cu12

UV_PATH := $(shell command -v uv 2>/dev/null)

.PHONY: all help run interactive setup clean

all: interactive

help:
	@echo "Usage: make [target] [ARGS=\"...\"]"
	@echo ""
	@echo "Targets:"
	@echo "  run          Run the script (auto-detects 'uv' or '.venv')"
	@echo "  interactive  Launch the script's built-in interactive CLI - [RECOMMENDED]"
# 	@echo "  prompt       Interactively select flags before running"
	@echo "  setup        Create .venv and install fallback dependencies"
	@echo "  clean        Remove the .venv directory"
	@echo ""
	@echo "Common Script Flags (pass via ARGS=\"...\"):"
	@echo "  [target_dir]          (Positional) Directory containing video files"
	@echo "  -m, --model <name>    Model (English): tiny.en / base.en / small.en / medium.en"
	@echo "                        Model (Mix): tiny / base / small / medium / turbo / large"
	@echo "  -d, --device <type>   Compute device: 'cuda' or 'cpu'"
	@echo "  --compute-type <type> Compute type for CTranslate2 (e.g., int8, float16)"
	@echo "  -c, --cooldown <sec>  Cooldown time in seconds between files"
	@echo "  -l, --language <code> Language code: en, ja, fa, es, or auto"
	@echo "  -t, --task <type>     Task to perform: 'transcribe' or 'translate'"
	@echo "                        NOTE: DO NOT use 'turbo' or English models for 'translate'"
	@echo "  -i, --interactive     Launch script's built-in interactive wizard"

# prompt:
# 	@echo "=== Transcribe Script Flag Configurator ==="
# 	@read -p "Target directory [./]: " target_dir; \
# 	target_dir=$${target_dir:-./}; \
# 	read -p "Select model size (e.g., turbo, base, large) [turbo]: " model; \
# 	model=$${model:-turbo}; \
# 	read -p "Select device (cuda/cpu) [cuda]: " device; \
# 	device=$${device:-cuda}; \
# 	read -p "Select compute type (int8/float16/int8_float16) [int8]: " compute; \
# 	compute=$${compute:-int8}; \
# 	read -p "Cooldown between files in seconds [Preset Default]: " cooldown; \
# 	read -p "Language code (e.g., en) or auto [auto]: " language; \
# 	language=$${language:-auto}; \
# 	read -p "Task (transcribe/translate) [transcribe]: " task; \
# 	task=$${task:-transcribe}; \
# 	args="\"$$target_dir\" -m $$model -d $$device --compute-type $$compute -l $$language -t $$task"; \
# 	if [ -n "$$cooldown" ]; then \
# 		args="$$args -c $$cooldown"; \
# 	fi; \
# 	echo ""; \
# 	echo "🚀 Executing with arguments: $$args"; \
# 	$(MAKE) --no-print-directory run ARGS="$$args"

run:
ifdef UV_PATH
	@echo "🚀 'uv' detected at $(UV_PATH). Running with uv..."
	uv run $(SCRIPT) $(ARGS)
else
	@echo "⚠️ 'uv' not found. Falling back to .venv..."
	@$(MAKE) --no-print-directory setup
	@echo "🚀 Running with standard venv..."
	$(PYTHON) $(SCRIPT) $(ARGS)
endif

setup:
ifndef UV_PATH
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "✅ .venv already exists. Skipping setup."; \
	else \
		echo "📦 Creating Python virtual environment..."; \
		python3 -m venv $(VENV_DIR); \
		echo "⬇️ Installing dependencies..."; \
		$(PIP) install --upgrade pip; \
		$(PIP) install $(DEPS); \
		echo "✅ Setup complete."; \
	fi
else
	@echo "✅ 'uv' is installed. No .venv is needed."
endif

interactive:
	@$(MAKE) --no-print-directory run ARGS="-i $(ARGS)"

clean:
	@echo "🧹 Cleaning up..."
	rm -rf $(VENV_DIR)
	@echo "✅ Clean complete."
