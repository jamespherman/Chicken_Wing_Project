# Transformer-Based Video-Pupil Modeling: Research Report

## Executive Summary

This report investigates the feasibility of using transformer-based deep learning to model the relationship between surgical video content and pupil dynamics. The goal is to move beyond simple luminance-based models toward architectures that can capture complex spatiotemporal patterns in the visual scene that drive pupillary responses.

## Current Problem

Our existing pupil-luminance regression achieves:
- **Median R² = 0.007** (with temporal kernel)
- **Mean frame luminance** is a poor predictor because the pupil responds to **foveal illumination**, not whole-scene brightness
- Two outlier subjects drive any apparent correlation with duration

## Research Findings

### 1. Neural Encoding Models with Video + Pupil

**ViV1T (Video Vision Transformer)** [bioRxiv, September 2025](https://www.biorxiv.org/content/10.1101/2025.09.16.676524v1.full)
- Trained on natural movies to predict mouse V1 neural responses
- Uses **two separate transformers** for spatial and temporal patches
- **Key finding**: Performance dropped 11.2% when pupil/behavioral data was withheld
- Architecture processes video patches + pupil size + running speed jointly
- Achieves 26% improvement over convolutional baselines

**Foundation Model for Neural Activity** [Nature, April 2025](https://www.nature.com/articles/s41586-025-08829-y)
- Large-scale transformer trained on neural responses to natural videos
- Generalizes to new mice and new stimulus types
- Demonstrates that video → neural response mapping is learnable with sufficient data

### 2. Remote Photoplethysmography (rPPG) - Analogous Problem

The rPPG field extracts **physiological signals from facial video**, which is analogous to our problem (predicting pupil from scene video).

**Key Architectures:**

| Model | Architecture | Key Innovation |
|-------|-------------|----------------|
| **PhysFormer** | Transformer | Temporal aggregation for physiological signals |
| **Spiking-PhysFormer** | Hybrid SNN+Transformer | Energy-efficient, mobile deployment |
| **AttenHRVNet** | 3D-CNN + Spatiotemporal Attention | Dynamic focus on informative regions/times |
| **VS-Net** | Video magnification + Self-attention | Enhances subtle color changes |
| **GraphPhys** | Graph Neural Network | Models facial region relationships |

[Frontiers Review, 2024](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2024.1420100/full)

**rPPG-Toolbox** [NeurIPS 2023, GitHub](https://github.com/ubicomplab/rPPG-Toolbox)
- Open-source implementation of multiple rPPG methods
- Could be adapted for pupil prediction from surgical video

### 3. Visual Saliency and Pupil Response

**Critical Connection**: Pupil dilation is modulated by visual saliency.

[Journal of Neuroscience](https://www.jneurosci.org/content/34/2/408):
> "Transient pupil dilation is elicited after visual stimulus presentation, and the evoked pupil response is modulated by contrast-based saliency, with faster and larger pupil responses following the presentation of more salient stimuli."

**Visual Saliency Transformer (VST)** [arXiv](https://arxiv.org/abs/2104.12099)
- Pure transformer for saliency prediction
- Models long-range dependencies in images
- Could provide intermediate features for pupil prediction

**TempVST (Temporal Visual Saliency Transformer)** [ResearchGate](https://www.researchgate.net/publication/382807677_The_Visual_Saliency_Transformer_Goes_Temporal_TempVST_for_Video_Saliency_Prediction)
- Extends VST to video domain
- Captures temporal saliency dynamics
- Saliency maps could predict pupil responses

### 4. Vision Transformers and Human-Like Attention

**DINO-trained ViTs** [arXiv, 2024](https://arxiv.org/html/2410.22768v1)
- Self-supervised ViTs develop attention patterns that closely align with human gaze
- Attention maps correlate with where humans look
- These attention maps could predict pupil responses

**Gaze-Informed Vision Transformers** [arXiv](https://arxiv.org/html/2308.13969v2)
- Integrates eye tracking (fixations + pupil dilation) with driving behavior prediction
- Demonstrates joint modeling of gaze and cognitive state

### 5. Cognitive Load Classification from Eye Movements

[MDPI Sensors](https://www.mdpi.com/1424-8220/21/23/8019)
- 92% accuracy classifying cognitive load from eye movement features
- Uses CNN and LSTM architectures
- Features: pupil dilation, fixations, saccades

[Frontiers](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1445697/full)
- Unsupervised machine learning on pupil dynamics
- K-means clustering identifies distinct cognitive load response patterns

---

## Proposed Architectures for Surgical Video → Pupil Prediction

### Architecture 1: Gaze-Guided Video Transformer (Recommended)

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT PROCESSING                         │
├─────────────────────────────────────────────────────────────┤
│  Video frames (T × H × W × 3)    Gaze positions (T × 2)    │
│         ↓                               ↓                   │
│  Patch embedding (ViT)           Gaze-guided ROI extraction │
│         ↓                               ↓                   │
│  [CLS] + spatial patches         Foveal patch sequence      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              SPATIOTEMPORAL TRANSFORMER                     │
├─────────────────────────────────────────────────────────────┤
│  Spatial Self-Attention (within frame)                      │
│         ↓                                                   │
│  Temporal Self-Attention (across frames)                    │
│         ↓                                                   │
│  Cross-Attention: Global features ← Foveal features         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              PUPIL PREDICTION HEAD                          │
├─────────────────────────────────────────────────────────────┤
│  Temporal convolution (for PLR dynamics)                    │
│         ↓                                                   │
│  Linear projection → Pupil diameter (T × 1)                 │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**
- Uses gaze position to extract foveal ROI (what the surgeon actually sees)
- Learns spatial saliency patterns (tools vs tissue)
- Models temporal dynamics (PLR kernel emerges from training)
- Cross-attention integrates scene context with foveal content

### Architecture 2: Two-Stream Fovea-Periphery Model

```
FOVEAL STREAM (high resolution)          PERIPHERAL STREAM (low resolution)
        │                                          │
    Gaze-centered                             Whole frame
    crop (64×64)                              downsampled (128×128)
        │                                          │
    3D-CNN encoder                            3D-CNN encoder
        │                                          │
        └──────────────┬───────────────────────────┘
                       │
              Concatenate + Attention
                       │
              Temporal Transformer
                       │
              Pupil prediction (with PLR kernel)
```

**Rationale:**
- Mirrors retinal organization (fovea vs periphery)
- Foveal stream captures gaze-contingent luminance
- Peripheral stream captures scene context/changes

### Architecture 3: Saliency-Guided Pupil Prediction

```
Video frames → Pre-trained VST → Saliency maps
                                      │
Gaze positions ──────────────────────►│
                                      ↓
                        Gaze-weighted saliency features
                                      │
                        Temporal Transformer
                                      │
                        Pupil prediction
```

**Rationale:**
- Leverages pre-trained saliency models
- Saliency at gaze point predicts pupil response
- Requires less training data (transfer learning)

---

## Data Requirements

### For Training (Estimated)
| Data Type | Minimum | Recommended |
|-----------|---------|-------------|
| Subjects | 30-50 | 100+ |
| Hours of video | 10-20 | 50+ |
| Samples (at 90 Hz) | ~3M | ~15M |

### Current Dataset
- 14 subjects, ~2.7 hours total
- **Insufficient for training a full transformer from scratch**
- **Sufficient for fine-tuning a pre-trained model**

### Recommended Approach
1. Use **pre-trained video backbone** (e.g., VideoMAE, TimeSformer)
2. **Freeze early layers**, train only task-specific heads
3. **Data augmentation**: temporal shifts, brightness jitter, gaze noise

---

## Implementation Roadmap

### Phase 1: Gaze-Contingent Baseline (1-2 weeks)
- Implement simple gaze-contingent luminance extraction
- Compare R² with mean frame luminance
- Validate that foveal luminance improves prediction

### Phase 2: Pre-trained Feature Extraction (2-3 weeks)
- Extract features from pre-trained ViT/VideoMAE
- Use gaze-weighted pooling of spatial features
- Train linear regression: features → pupil
- Compare with luminance-only baseline

### Phase 3: Fine-tuned Transformer (4-6 weeks)
- Fine-tune temporal transformer on surgical video
- Implement gaze-guided attention mechanism
- Train end-to-end with pupil prediction loss
- Evaluate generalization to held-out subjects

### Phase 4: Clinical Validation (Ongoing)
- Correlate model residuals with surgical outcomes
- Compare cognitive load estimates with existing methods
- Validate on prospective data collection

---

## Recommendations

### Immediate (Low Effort, High Value)
**Implement gaze-contingent luminance extraction first.**
- This is a prerequisite for any transformer approach
- Will likely improve R² from 0.007 to 0.05-0.20
- Provides the foveal features needed for deep learning

### Short-term (Medium Effort)
**Extract features from pre-trained video transformers.**
- Use VideoMAE or TimeSformer as frozen backbone
- Pool features at gaze location
- Train simple temporal model on top
- Requires no additional data collection

### Long-term (High Effort, Potentially High Value)
**Train custom gaze-guided video transformer.**
- Requires more data (consider collaborations)
- Could achieve state-of-the-art pupil prediction
- Interpretable attention maps reveal what drives pupil response
- Publishable as methodological contribution

---

## Key References

1. ViV1T: Movie-trained transformer for neural response prediction - [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.09.16.676524v1.full)
2. Foundation model for neural activity - [Nature](https://www.nature.com/articles/s41586-025-08829-y)
3. Deep learning for rPPG - [Frontiers](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2024.1420100/full)
4. rPPG-Toolbox - [GitHub](https://github.com/ubicomplab/rPPG-Toolbox)
5. Visual Saliency Transformer - [arXiv](https://arxiv.org/abs/2104.12099)
6. Pupil and saliency - [J Neuroscience](https://www.jneurosci.org/content/34/2/408)
7. DINO-ViT and human attention - [arXiv](https://arxiv.org/html/2410.22768v1)
8. Cognitive load from eye movements - [MDPI](https://www.mdpi.com/1424-8220/21/23/8019)

---

## Conclusion

A transformer-based approach to video-pupil modeling is **scientifically justified and technically feasible**, drawing on advances in:
- Neural encoding models (ViV1T, foundation models)
- Remote physiological sensing (rPPG with transformers)
- Visual saliency prediction (VST, TempVST)
- Attention mechanisms that mimic human gaze

However, given the current dataset size (14 subjects), I recommend a **staged approach**:
1. **First**: Implement gaze-contingent luminance (immediate improvement)
2. **Then**: Extract pre-trained transformer features (transfer learning)
3. **Finally**: Consider custom training if more data becomes available

The most promising immediate direction is combining **gaze-contingent luminance** with **temporal convolution** - this addresses the core limitation (wrong luminance measure) without requiring deep learning infrastructure.
