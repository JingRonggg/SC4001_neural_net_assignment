# HDB price baseline

## Run it

Install the exact environment recorded in `uv.lock`:

```bash
uv sync
```

Run the baseline with:

```bash
uv run python partb.py
```


## In NTU CCDS GPU CLUSTER

1. run 
```bash
module load uv
uv sync
```

2. run each job sequentially, ensuring one job has succeeded before running the next
```bash
sbatch run_b.sh
```