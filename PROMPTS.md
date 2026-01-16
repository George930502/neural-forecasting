# Neural Forecasting Task – End-to-End Development Prompt

You are an AI engineer tasked with building a complete, Codabench-compatible neural forecasting solution for the **HDR Challenge Year 2 – Neural Forecasting** task.

---

## 1. Grounding and Repository Understanding

- **Deeply read and follow the task description** in `task.md`. Treat it as the single source of truth for:
  - prediction targets  
  - input / output formats  
  - evaluation protocol  
  - submission constraints and scoring rules  

- **Thoroughly inspect** the files under `HDRChallenge_y2/NeuralForecasting/` to understand:
  - existing baselines and utilities  
  - dataset loading and preprocessing conventions  
  - expected model interfaces  
  - training, inference, and packaging scripts  
  - submission guidelines

---

## 2. Paper-Driven Improvements (Using MCP Tools)

- Use **paper search MCP** to identify the most relevant papers for:
  - multivariate neural time-series forecasting  
  - μECoG / ECoG neural signal modeling  
  - neural decoding and sequence forecasting  

- Prioritize papers that provide **actionable improvements** for this task, such as:
  - temporal modeling architectures  
  - channel-aware or spatial modeling  
  - frequency-band–aware representations  
  - robust normalization and regularization  
  - handling distribution shift across sessions or subjects  

- **Download the selected papers** and store them under the directory:
  ```
  papers/
  ```

- Use **pdf reader MCP** to read the papers and extract key insights, including:
  - methods compatible with data shaped as `N × T × C × F`
  - recommended model architectures
  - preprocessing and normalization strategies
  - loss functions suitable for forecasting
  - domain-specific insights for μECoG signals

---

## 3. End-to-End Forecasting System

Implement a full neural forecasting pipeline under:
```
develop/
```

This pipeline must include:
- dataset loading and preprocessing
- model definition
- training loop and checkpointing
- inference pipeline compatible with Codabench
- reproducibility controls (random seeds, logging, deterministic behavior when possible)

---

## 4. Codabench Compatibility and Evaluation

- Codabench evaluates submissions using the testing data located at:
  ```
  dataset/test
  ```

- After development:
  - run a **full local evaluation** using the testing data
  - verify that outputs strictly follow the submission format defined in `task.md`
  - ensure the submission can run in a clean environment (no hidden dependencies, correct entry points)

---

**Important:**  
Always prioritize correctness and Codabench compatibility. Performance improvements must not break the required interfaces or evaluation pipeline.
