# snnsim


## Example

Example run

```text
PS C:\Users\Jonathan\Documents\snnsim> cargo test xor
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.04s
     Running unittests src\lib.rs (target\debug\deps\snnsim-e72de92ac403bfd0.exe)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 0.00s

     Running tests\xor.rs (target\debug\deps\xor-9de280f164f1c62a.exe)

running 1 test
████████████████████████████████████████   10000/  10000 [00:00:20 / 00:00:00] 486.2254/s current: 38.00%, best: 31.50%             test xor ... FAILED

failures:

---- xor stdout ----
weights:
        [2, 3] [0.83714, -1.30811, -1.24219, 0.49648, 0.52080, -1.12557]
        [3, 1] [0.69260, 1.27329, -0.73469]

thread 'xor' panicked at tests\xor.rs:109:5:
assertion failed: false
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace


failures:
    xor

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 20.57s

error: test failed, to rerun pass `--test xor`
PS C:\Users\Jonathan\Documents\snnsim>
```

## TODOs

Neurons|Implemented
---|---
Leaky Integrate and Fire (LIF)|✓
Adaptive LIF|x
Resonate and fire|x
Quadratic LIF|x
Izhikevich|x
Hodgkin-Huxley|x
Sigma Delta (SDNN)|x

Optimizers|Implemented
---|---
Backpropagation through time (BPTT)|✓
SpikeProp|x
SuperSpike|x
SLAYER|x
EventProp|x
Spike Timing Dependant Plasticity (STDP)|x

### Misc

- add a `connectivity` parameter to support sparsely connected spiking layers like snnTorch which supports connecting layers with conv layer connectivity.
- should `backward` take some cost rather than just the target spikes?
- look at using for [cusparse](https://docs.nvidia.com/cuda/cusparse/) for sparse matrix operations. This might require more complex logic to check when it is worth using sparse matrix operations.
- output the EFLOPs metrics for models to give an idea of their foundational performance.
- run a test including sparsity in the cost function
- look at [ml_genn](https://github.com/genn-team/ml_genn).
- I went to a conferece and heard about "audoAdjoint" which is intended to be a version of automatic differentiation that works for event driven models and SNNs. Look into this.
- paper from Matias Barandiaran and James Stovold on "Developmental Graph Cellular Automata Can Grow Reservoirs" and lookup info on plastic reservoirs and how these could be implemented/used.
