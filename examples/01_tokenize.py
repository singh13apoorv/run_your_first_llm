from common import load_model

TEXT = "The capital of France is"


def main():
    tokenizer, _ = load_model(with_model=False)

    ids = tokenizer(TEXT).input_ids
    print(f"Token ids:      {ids}")
    print(f"Tokens:         {tokenizer.convert_ids_to_tokens(ids)}")
    print(f"Decoded:        {tokenizer.decode(ids)}")
    print(f"BOS:            {tokenizer.bos_token} ({tokenizer.bos_token_id})")
    print(f"EOS:            {tokenizer.eos_token} ({tokenizer.eos_token_id})")
    print(f"Vocab size:     {len(tokenizer)}")


if __name__ == "__main__":
    main()