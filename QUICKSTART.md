# 🏥 Pharmacy Desert Explorer - Quick Start Guide

## Project Structure

```
pharmacy-deserts/
├── app/
│   └── app.py                  # ← Streamlit UI (run this!)
├── data/
│   ├── finalfinalfinal_training.py  # ← GLM model training
│   ├── features.py             # ← Data preprocessing
│   └── loaders.py              # ← Data loading utilities
├── models/                     # ← Scoring logic
├── viz/                        # ← Map visualization
├── results/                    # ← Model output scores
├── deploy/                     # ← Docker & deployment configs
└── requirements.txt            # ← Dependencies
```

## 🚀 How to Run (2 Easy Steps!)

### Step 1: Generate GLM Scores (One-time setup)
```bash
# Run the GLM training script to generate scores
python data/finalfinalfinal_training.py

# This creates output in: results/national_ifae_rank.csv
```

### Step 2: Launch the Streamlit App
```bash
# Run the Streamlit app
streamlit run app/app.py

# 🎉 That's it! The app will open in your browser at http://localhost:8501
```

## ⚡ What's Different Now?

### ✅ Benefits:
1. **Much cleaner code** - UI separated from data processing
2. **Faster interactions** - Caching prevents re-running on every click
3. **Easier to maintain** - Each file has one clear purpose
4. **Better for your presentation** - Professional structure

### 📁 File Purposes:

**`app/app.py`** (Main Streamlit Application)
- User interface with sliders and maps
- Loads data using cached loaders from `app/state.py`
- Three scoring modes: GLM Only, Math Only, Blended
- Displays interactive Folium map visualizations

**`data/features.py`** (Data Preprocessing)
- Percentile normalization and feature engineering
- Score computation helpers

**`data/finalfinalfinal_training.py`** (GLM Model Training)
- Poisson GLM with OOF cross-validation
- Per-state calibration and neighbor QA
- Outputs ranked results to `results/`
- Run this whenever you want fresh GLM scores

## 🔄 Workflow

```
┌──────────────────────────────────────┐
│ 1. python data/finalfinalfinal_      │
│    training.py                       │
│    (trains GLM, generates scores)    │
└──────────┬───────────────────────────┘
           │
           v
┌──────────────────────────────────────┐
│ results/                             │
│  └── national_ifae_rank.csv          │  ← GLM scores stored here
│  └── qa_expected_vs_observed.csv     │  ← QA metrics
└──────────┬───────────────────────────┘
           │
           v
┌──────────────────────────────────────┐
│ 2. streamlit run app/app.py          │
│    (interactive dashboard)           │
└──────────────────────────────────────┘
           │
           v
   ┌───────────────────┐
   │ App loads:         │
   │ • Data files       │
   │ • GLM scores       │
   │ • Blends scores    │
   │ • Shows results    │
   └───────────────────┘
```

## 🎯 For Your Presentation

### Demo Flow:
1. Show `data/finalfinalfinal_training.py` briefly
   - "This trains our GLM + hybrid model"
   - "It analyzes income, health, pharmacy access, density, and more"
   
2. Run `app/app.py`
   - "This is our interactive tool"
   - "Left sidebar lets you choose scoring modes"
   - "Mathematical weights are adjustable"
   - "GLM scores are pre-computed"

3. Adjust sliders
   - "See how fast it responds!" ⚡
   - "No re-loading data thanks to caching"
   
4. Show results
   - "Final ranking blends both approaches"
   - "Math = transparent, GLM = statistical discovery"

## 🔧 Troubleshooting

**If GLM scores not found:**
```bash
# Run the training script
python data/finalfinalfinal_training.py
```

**If Streamlit won't start:**
```bash
# Make sure you're in the project root directory
cd path/to/pharmacy-deserts

# Activate virtual environment if needed
source venv/bin/activate

# Run app
streamlit run app/app.py
```

**If imports fail:**
```bash
# Install dependencies
pip install -r requirements.txt
```

## 📊 Key Features Now Working

✅ **Hybrid AI + Math Approach** - Best of both worlds  
✅ **Fast Slider Adjustments** - Instant response with caching  
✅ **Clear Code Organization** - Easy to understand and maintain  
✅ **Professional UI** - Metrics, maps, explanations  
✅ **Export Options** - Download full results or top 100  

## 💡 Pro Tips

- **Update GLM scores before presentation**: Run the training script fresh
- **Adjust mathematical weights live**: Show stakeholder priorities
- **Explain the hybrid approach**: Math = transparent, GLM = discovery
- **Use the map view**: Visual impact for presentations

---

**Ready for your presentation! 🎉**

