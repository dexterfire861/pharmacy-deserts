# Pharmacy Desert Explorer

A Streamlit-based geospatial analysis platform for identifying "pharmacy deserts" - areas with limited pharmacy access. The system combines flexible data ingestion, mathematical scoring algorithms, and machine learning models to analyze pharmaceutical accessibility across geographic regions.

## Quick Start

### Prerequisites
- Python 3.11+
- pip or conda for package management

### Installation
```bash
# Clone the repository
git clone https://github.com/dexterfire861/pharmacy-deserts.git
cd pharmacy-deserts

# Install dependencies
pip install -r requirements.txt

# Set up environment (copy and modify as needed)
cp env.example .env
```

### Running the Application
```bash
# Launch the Streamlit app
streamlit run app/app.py

# The app will open at http://localhost:8501
```

## 📁 Project Structure

```
pharmacy-deserts/
├── app/                    # Streamlit UI application
│   ├── app.py             # Main application entry point
│   ├── state.py           # Cached data loading and state management
│   ├── config.py          # Environment configuration
│   └── pages/             # Additional Streamlit pages
├── models/                # Scoring and ML models
│   ├── schema.py          # Scoring component definitions
│   ├── scoring.py         # Mathematical scoring functions
│   └── ai_scores.py       # AI model integration
├── ingestion/             # Data ingestion pipeline
│   ├── parsers.py         # File parsing (CSV, Excel, JSON, etc.)
│   ├── normalize.py       # ZIP code normalization
│   └── dataset_loader.py  # Configuration-driven data loading
├── storage/               # Dataset versioning and persistence
├── training/              # ML model training pipeline
│   ├── orchestrator.py    # Training workflow management
├── data/                  # Data loading utilities
├── viz/                   # Visualization components
├── raw_data/              # Local data storage (versioned)
├── results/               # Model outputs and rankings
└── deploy/                # Docker and deployment configs
```

## 🧠 Core Concepts

### Configuration-Driven Data Mapping
The system uses flexible schema mapping to work with arbitrary data formats:
- Upload CSV/Excel files with any column names
- Map columns to standardized scoring components
- System automatically handles data normalization and scoring

### Dual-Mode Performance
- **GLM Mode**: Fast startup with pre-computed model results
- **Math/Blended Mode**: Full dataset processing with Census ACS API calls
- Intelligent caching prevents redundant expensive operations

### Dataset Versioning
```
raw_data/datasets/{dataset_id}/versions/{timestamp}/
├── dataset_config.json    # Configuration and mappings
├── files/                 # Raw uploaded files
└── mappings/             # Column mapping configurations
```

## 🔧 Development Workflows

### Data Upload and Processing
1. Navigate to "Upload_Data" page in the Streamlit interface
2. Upload CSV/Excel files → parsed by `ingestion/parsers.py`
3. Configure ZIP code normalization modes
4. Map columns to scoring components via UI
5. System generates versioned dataset configuration

### Model Training
```bash
# Train new model with current dataset
python new_training.py

# Training pipeline outputs:
# - GLM coefficients: results/glm_full_coefficients.csv
# - IFAE rankings: results/national_ifae_rank.csv
# - QA reports: results/qa_expected_vs_observed.csv
```

### Environment Configuration
```bash
# Local development
ENVIRONMENT=development
REQUIRE_AUTH=false

# Production deployment
ENVIRONMENT=production  
AWS_S3_BUCKET=your-bucket-name
REQUIRE_AUTH=true
APP_PASSWORD=your-password
```

## 🏗️ Architecture for ML Integration

### Scoring System
The flexible scoring system (`models/schema.py`) defines standard components:
- **Demographics**: population, density, age distribution  
- **Socioeconomic**: income, education, employment
- **Health**: burden metrics, chronic conditions
- **Access**: pharmacy count, distance metrics
- **Geographic**: rural/urban classification

### Model Integration Points
- **Data Loading**: `app/state.py` provides cached data access
- **Scoring Engine**: `models/scoring.py` handles mathematical calculations
- **Training Pipeline**: `training/orchestrator.py` manages ML workflows
- **Results Storage**: Versioned outputs in `results/` directory

### External Dependencies
- **Census ACS API**: Educational attainment data by ZIP code
- **S3 Storage**: Production dataset versioning (optional)
- **GLM Models**: Statsmodels-based Negative Binomial regression

## 📊 Model Architecture

The system uses a research-grade **GLM + Expected-Access** approach:

1. **Poisson GLM**: Out-of-fold cross-validation for unbiased predictions
2. **Per-State Calibration**: State-level adjustment with national fallback
3. **Spatial Analysis**: Geographic k-nearest neighbors for validation
4. **Anomaly Detection**: Flags suspicious zero-pharmacy areas
5. **Dual Scoring**: Residual-based and deficit-based IFAE rankings

## 🚀 Deployment

### Docker
```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f pharmacy-deserts
```

### AWS EC2
```bash
# Use provided setup script
./deploy/ec2-setup.sh
```

## 📚 Documentation

- **QUICKSTART.md**: Step-by-step getting started guide
- **INTEGRATION_GUIDE.md**: Detailed model integration documentation  
- **.github/copilot-instructions.md**: AI coding assistant guidance

## 🤝 Contributing

This codebase is optimized for AI-assisted development with clear separation of concerns:
- Modular design with single-responsibility components
- Configuration-driven flexibility for different data sources
- Comprehensive error handling and logging
- Extensive caching for performance optimization

For ML model integration work, focus on:
- `models/` directory for scoring algorithms
- `training/` directory for model training workflows
- `data/` directory for feature engineering
- `results/` directory for model outputs and evaluation