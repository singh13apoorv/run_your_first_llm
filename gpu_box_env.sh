mkdir -p logs/rtx4090
nvidia-smi > logs/rtx4090/nvidia-smi.txt
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))" \
  | tee logs/rtx4090/env.txt