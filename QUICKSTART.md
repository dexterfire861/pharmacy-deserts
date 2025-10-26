# 🏥 Pharmacy Desert Explorer - Quick Start Guide

## New Structure (Simplified!)

Your codebase is now organized into separate files:

```
pharmacy-deserts/
├── app.py                      # ← Streamlit UI (run this!)
├── data_processing.py           # ← Data processing functions (library)
├── IF_AE_training.ipynb        # ← AI model training (run this first)
├── data/                       # ← Input data files
├── results/                    # ← AI scores output
└── requirements.txt            # ← Dependencies
```

## 🚀 How to Run (2 Easy Steps!)

### Step 1: Generate AI Scores (One-time setup)
```bash
# Run the Isolation Forest notebook to generate AI scores
jupyter execute IF_AE_training.ipynb

# This creates: results/national_ifae_rank.csv
```

### Step 2: Launch the Streamlit App
```bash
# Run the new streamlit app
streamlit run app.py

# 🎉 That's it! The app will open in your browser
```

## ⚡ What's Different Now?

### ✅ Benefits:
1. **Much cleaner code** - UI separated from data processing
2. **Faster interactions** - Caching prevents re-running on every click
3. **Easier to maintain** - Each file has one clear purpose
4. **Better for your presentation** - Professional structure

### 📁 File Purposes:

**`app.py`** (Main Streamlit Application)
- User interface with sliders and maps
- Loads data using functions from `data_processing.py`
- Blends mathematical + AI scores
- Displays interactive visualizations

**`data_processing.py`** (Data Processing Library)
- Pure functions for loading data
- Mathematical scoring model
- Score blending logic
- No UI code - just data operations

**`IF_AE_training.ipynb`** (AI Model Training)
- Trains Isolation Forest on features
- Generates AI anomaly scores
- Outputs `results/national_ifae_rank.csv`
- Run this whenever you want fresh AI scores

## 🔄 Workflow

```
┌─────────────────────────────┐
│ 1. Run IF_AE_training.ipynb │
│    (generates AI scores)    │
└──────────┬──────────────────┘
           │
           v
┌─────────────────────────────┐
│ results/                    │
│  └── national_ifae_rank.csv │  ← AI scores stored here
└──────────┬──────────────────┘
           │
           v
┌─────────────────────────────┐
│ 2. Run: streamlit run app.py│
│    (interactive dashboard)  │
└─────────────────────────────┘
           │
           v
   ┌───────────────────┐
   │ App loads:        │
   │ • Data files      │
   │ • AI scores       │
   │ • Blends scores   │
   │ • Shows results   │
   └───────────────────┘
```

## 🎯 For Your Presentation

### Demo Flow:
1. Show `IF_AE_training.ipynb` briefly
   - "This trains our Isolation Forest AI model"
   - "It analyzes income, health, pharmacy access, heat, air quality"
   
2. Run `app.py`
   - "This is our interactive tool"
   - "Left sidebar shows both models"
   - "Mathematical weights are adjustable"
   - "AI scores are pre-computed"

3. Adjust sliders
   - "See how fast it responds!" ⚡
   - "No re-loading data thanks to caching"
   
4. Show results
   - "Final ranking blends both approaches"
   - "Math = transparent, AI = pattern discovery"

## 🔧 Troubleshooting

**If AI scores not found:**
```bash
# Re-run the notebook
jupyter execute IF_AE_training.ipynb
```

**If Streamlit won't start:**
```bash
# Make sure you're in the right directory
cd /Users/aryaanverma/pharmacy-deserts/pharmacy-deserts

# Activate virtual environment if needed
source venv/bin/activate

# Run app
streamlit run app.py
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

- **Update AI scores before presentation**: Run the notebook fresh
- **Adjust mathematical weights live**: Show stakeholder priorities
- **Explain the hybrid approach**: Math = transparent, AI = discovery
- **Use the map view**: Visual impact for presentations

---

**Ready for your presentation! 🎉**

