# CUDA Kernel-Optimization RL Environment

An original reinforcement learning environment built on [Prime Intellect](https://primeintellect.ai)'s `verifiers` framework, where an agent is given a naive CUDA kernel (elementwise, reduction categories) and must iteratively rewrite it for speed while staying numerically correct against a reference implementation. Built as an independent portfolio project following Prime Intellect's Lab RL training guide, with a focus on designing genuinely hard-to-hack reward and correctness verification for GPU-benchmarked code.

## Environment design

- **Task generation**: parameterized by kernel category, difficulty tier, and seed. Difficulty controls which inefficiency is seeded into the starting kernel (uncoalesced memory access, no shared memory reuse, poor block/grid sizing, unnecessary `__syncthreads()`, no vectorized loads, warp divergence).
- **Correctness**: floating-point epsilon comparison against a reference kernel, not exact equality. Hard gate, zero score for any correctness failure, no partial credit for fast-but-wrong.
- **Scoring**: correctness gate + speedup ratio vs. the naive baseline, normalized per difficulty tier.
- **Reward-hacking defenses**: randomized inputs between correctness and benchmark runs (prevents memorization), output checksums (prevents dead-code-elimination and cached-result exploits), and timing-vs-input-size scaling checks (catches compiled-away kernels).
- **Tools**: `view_kernel`, `edit_kernel`, `compile_and_test`, `benchmark`, `check_score`, via the `StatefulToolEnv` pattern with in-memory kernel/test state.

## Setup

Requires the `prime` CLI and a Prime Intellect account. See [Prime Intellect's Lab guide](https://docs.primeintellect.ai) for base setup (`prime lab setup`, `verifiers` install). Environment-specific setup and reproduction steps are in `PLAN.md`.

## Related work

See [qwen-mtp-rollout-speed](https://github.com/MrSnazMan/qwen-mtp-rollout-speed) for a related experiment measuring speculative-decoding (MTP) effects on RL rollout speed, done as part of the same broader RL training learning project.

## Acknowledgments

- [Prime Intellect](https://primeintellect.ai) — the `prime` CLI, Prime Lab, Hosted Training, and the `verifiers` library this environment is built on
- [Claude Code](https://claude.com/claude-code) (Anthropic) — used extensively throughout this project for environment implementation and design iteration

## License

MIT
