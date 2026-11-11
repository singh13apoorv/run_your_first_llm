from common import load_model, DEVICE, memory_gb

def main():
    _, model = load_model()

    params = list(model.parameters())
    print(f"Device:      {DEVICE}")
    print(f"Parameters:  {sum(p.numel() for p in params):,}")
    print(f"Tensors:     {len(model.state_dict())}")
    print(f"Dtype:       {params[0].dtype}")
    print(f"Memory:      {memory_gb():.2f} GB allocated")
    print(f"Weights on:  {params[0].device}")

if __name__ == "__main__":
    main()