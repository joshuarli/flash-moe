don't have enough disk space lol, just have enough to store repacked experts

on a remote server:
```
uvx hf download mlx-community/Qwen3.5-397B-A17B-4bit --revision 39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3

link ~/ai/hf to ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit
```

locally:

mkdir -p /tmp/hf-remote
sshfs josh@100.110.16.38:/home/josh/ai/hf /tmp/hf-remote

python3 generate_expert_index.py --model /tmp/hf-remote/snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3

# Extract non-expert weights + tokenizer (writes ~5.5GB locally)
cd metal_infer

uv run --with numpy extract_weights.py --model /tmp/hf-remote/snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3/

python export_tokenizer.py /tmp/hf-remote/snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3/tokenizer.json

python export_vocab.py /tmp/hf-remote/snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3/tokenizer.json

# Repack experts locally (~209GB, reads remote, writes local)
cd /Users/josh/flash-moe
python repack_experts.py --output /Users/josh/flash-moe/model

# This writes to /Users/josh/flash-moe/model/packed_experts/.

# Run inference (point at local dir)
export FLASH_MOE_MODEL=/Users/josh/flash-moe/model
cd metal_infer && make && ./infer --serve 8000


