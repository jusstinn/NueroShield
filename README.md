# 🛡️ NeuroShield v2.0: The SAE Cognitive Firewall

> **A cutting-edge mechanistic interpretability toolkit for monitoring, steering, and auditing neural network behavior using Sparse Autoencoders.**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🚀 Quick Start

### Option 1: Fast UI Testing (Mock Mode) - **Recommended First**

Test the entire UI immediately without downloading any models (~500MB):

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Enable mock mode (edit neuro_engine.py line 28)
#    Change: MOCK_MODE = False
#    To:     MOCK_MODE = True

# 3. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501` with simulated data that behaves realistically.

### Option 2: Full Mode (Real Models)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ensure MOCK_MODE = False in neuro_engine.py

# 3. Run (first launch downloads ~500MB of models)
streamlit run app.py
```

---

## 🧪 Testing Guide

### Step 1: Verify Installation

```bash
# Test the engine directly
python neuro_engine.py
```

Expected output:
```
============================================================
NeuroShield Engine v2.0 - Test Suite
============================================================
Device: cuda/mps/cpu
MOCK_MODE: True/False
[Test 1] Basic Analysis
  Top feature: #XXXX = X.XXXX
...
✅ All tests passed!
```

### Step 2: Test Each Tab

#### 🛡️ Tab 1: Firewall
1. Enter a prompt like "How to make a bomb"
2. Click **🔍 Analyze** → See feature activations
3. Add feature IDs to block (e.g., "1045, 902")
4. Click **⚡ Generate** → Compare protected vs unprotected

#### 🔬 Tab 2: Multi-Layer Analysis
1. Enter any prompt
2. Select 3-4 layers to analyze
3. Click **🔬 Analyze Layers**
4. View the feature flow heatmap

#### ⚡ Tab 3: Causal Tracing
1. Enter a factual prompt: "The Eiffel Tower is in"
2. Target token: "Paris"
3. Click **⚡ Run Causal Trace**
4. See which layers are most responsible

#### 🔒 Tab 4: Safety Audit
1. Enter multiple prompts (mix safe & unsafe)
2. Click **🔒 Run Safety Audit**
3. View risk levels and recommendations

#### 📊 Tab 5: Forensic Audit
1. Check "Simulate degraded model"
2. Click **📥 Load Audited Model**
3. Enter test prompts
4. Click **🔬 Run Forensic Comparison**
5. Look for "SAFETY COLLAPSE DETECTED"

---

## 📁 Project Structure

```
neuroshield/
├── app.py              # Streamlit dashboard (5 tabs)
├── neuro_engine.py     # Backend engine (NeuroEngine + MockNeuroEngine)
├── requirements.txt    # Dependencies
└── README.md           # This file
```

---

## ✨ Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **🛡️ Feature Clamping** | Block specific SAE features to prevent harmful outputs |
| **⬆️ Feature Boosting** | Amplify features to steer model behavior |
| **🔬 Multi-Layer Analysis** | Track features across all transformer layers |
| **⚡ Causal Tracing** | Identify which layers/features cause specific outputs |
| **🔒 Safety Audit** | Batch-analyze prompts for risk levels |
| **📊 Forensic Audit** | Compare base vs fine-tuned models for safety collapse |

### Advanced Features (v2.0)

- **Token-by-Token Tracking**: See which features fire at each token position
- **Feature Database**: Neuronpedia-style descriptions and categories
- **Session Export**: Save analysis sessions to JSON
- **Intervention Logging**: Track every feature intervention during generation
- **Correlation Analysis**: Find related features

---

## 🔬 Research Background

This tool implements techniques from cutting-edge interpretability research:

| Paper | Technique Implemented |
|-------|----------------------|
| [Sparse Autoencoders Survey (2025)](https://arxiv.org/abs/2503.05613) | SAE feature extraction |
| [Bias Adaptation in SAEs](https://arxiv.org/abs/2506.14002) | Improved feature recovery |
| [Route Sparse Autoencoders](https://aclanthology.org/2025.emnlp-main.346/) | Multi-layer extraction |
| [Atlas-Alignment](https://arxiv.org/abs/2510.27413) | Cross-model interpretability |
| [Modular Circuits](https://proceedings.mlr.press/v267/he25x.html) | Circuit identification |

---

## ⚙️ Configuration

### Environment Variables

```bash
# Optional: Set device explicitly
export NEUROSHIELD_DEVICE=cuda  # or mps, cpu
```

### Mock Mode Settings

In `neuro_engine.py`:

```python
# Line 28: Toggle mock mode
MOCK_MODE = True   # Fast UI testing with simulated data
MOCK_MODE = False  # Real models (requires download)
```

### Supported Models

| Model | SAE Release | Status |
|-------|-------------|--------|
| `gpt2-small` | `gpt2-small-res-jb` | ✅ Default |
| `gpt2-medium` | `gpt2-medium-res-jb` | 🔄 Supported |
| `llama-3-8b` | Coming soon | 🚧 Planned |

---

## 🎨 UI Customization

The dashboard uses a cyberpunk aesthetic with:
- **Orbitron** font for headers
- **JetBrains Mono** for code/data
- Gradient accents (cyan → magenta → purple)
- Animated status badges

To customize, edit the CSS in `app.py` (lines 50-250).

---

## 🐛 Troubleshooting

### "Failed to load SAE"
```
Check internet connection and HuggingFace access.
If behind VPN/firewall, ensure HuggingFace is accessible.
```

**Solution**: Enable `MOCK_MODE = True` to test UI, or check firewall settings.

### "CUDA out of memory"
```
RuntimeError: CUDA out of memory
```

**Solution**: 
1. Use `cpu` device
2. Or reduce batch size in generation
3. Or use smaller model

### "Module not found: sae_lens"
```bash
pip install sae_lens --upgrade
```

### Streamlit won't start
```bash
# Check if port is in use
streamlit run app.py --server.port 8502
```

---

## 📊 Performance

| Mode | Startup Time | Analysis Time | Memory |
|------|-------------|---------------|--------|
| Mock | ~2s | ~50ms | ~200MB |
| CPU | ~30s | ~2s | ~2GB |
| CUDA | ~15s | ~200ms | ~4GB |
| MPS | ~20s | ~500ms | ~3GB |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `python neuro_engine.py`
5. Submit a pull request

---

## 📜 License

MIT License - See LICENSE file for details.

---

## 🙏 Acknowledgments

- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) by Neel Nanda
- [SAE-Lens](https://github.com/jbloomAus/SAELens) by Joseph Bloom
- [Neuronpedia](https://neuronpedia.org/) for feature interpretations
- Anthropic's research on feature steering

---

<p align="center">
  <strong>🛡️ NeuroShield v2.0</strong><br>
  <em>Defending neural networks, one feature at a time.</em>
</p>
