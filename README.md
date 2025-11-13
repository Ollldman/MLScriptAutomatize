# ModelForge

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green)
![Pydantic](https://img.shields.io/badge/Pydantic-2.0+-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15+-purple)

## Overview

**ModelForge** is an automated machine learning pipeline framework that follows CRISP-DM methodology to streamline the entire data science workflow. From data loading to model deployment, ModelForge provides a structured, validated, and production-ready approach to building machine learning models.

## ✨ Features

### 🔧 Core Capabilities
- **Multi-source Data Loading**: CSV, Excel, SQL databases, Kaggle, Hugging Face, UCI, and sklearn datasets
- **Automated Data Validation**: Type checking, missing value detection, outlier analysis
- **Smart Data Preprocessing**: Automated cleaning, encoding, and normalization
- **Hyperparameter Optimization**: GridSearchCV with XGBoost
- **Comprehensive Reporting**: PDF/HTML reports with visualizations and metrics
- **Production Ready**: Model persistence, email notifications, and API integration

### 🏗️ Architecture
- **Pydantic Validation**: Type-safe data structures throughout the pipeline
- **Modular Design**: Separated concerns for data loading, validation, analysis, and reporting
- **CRISP-DM Compliance**: Follows industry-standard data mining methodology
- **Extensible Framework**: Easy to add new data sources, models, and visualizations

## 🛠️ Technologies Used

| Technology | Purpose | Version |
|------------|---------|---------|
| **scikit-learn** | Machine learning algorithms and utilities | 1.3+ |
| **XGBoost** | Gradient boosting for classification tasks | Latest |
| **Pydantic** | Data validation and settings management | 2.0+ |
| **FastAPI** | REST API for pipeline execution | 0.100+ |
| **PyQt5** | Desktop GUI interface (in development) | 5.15+ |
| **Pandas** | Data manipulation and analysis | 2.0+ |
| **NumPy** | Numerical computations | 1.24+ |
| **Seaborn** | Statistical data visualization | 0.12+ |
| **Matplotlib** | Plotting and visualization | 3.7+ |

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- poetry package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/ModelForge.git
cd ModelForge
```

2. **Install dependencies**
```bash
# Using poetry
poetry install
```

3. **Configure environment variables**

### Environment Configuration

Create a `.env` file with the following variables:

```env
# Database Configuration
DB_CONNECTION=postgresql://user:pass@localhost/dbname
DB_FOR_MODELS=postgresql://user:pass@localhost/ml_results
AIO_DB_CONNECTION=postgresql+asyncpg://user:pass@localhost/dbname

# Kaggle API (optional)
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key

# Email Configuration (optional)
smtp_server=smtp.gmail.com
smtp_port=587
sender_email=your_email@gmail.com
sender_password=your_app_password

# Project Directories
reports_dir=./reports
```

**For more information on how the modules work, see the module folders <u>docs.md</u> files inside the folder ModelForge/moudles.**

> **Additionally, please pay attention to the file 
ARCHITECTURE.md**

## 📖 Usage

### Manual Execution

Run the main pipeline interactively:

```bash
python ModelForge/main.py
```

The script will guide you through:

1. **Business Understanding** - Define project goals and success metrics
2. **Data Loading** - Choose from multiple data sources
3. **Data Understanding** - Automatic statistics and visualization
4. **Data Preparation** - Cleaning, encoding, and preprocessing
5. **Modeling** - XGBoost with configurable GridSearch
6. **Evaluation** - Comprehensive model metrics
7. **Deployment** - Report generation and optional email delivery

### Example: Iris Dataset

```bash
# When prompted for data source, choose option 7 (sklearn)
# Select 'load_iris' as dataset name
# Choose target column (usually the last one)
# Use default GridSearch parameters or customize
```

### Cron Scheduling (Linux/Mac)

1. **Open crontab**
```bash
crontab -e
```

2. **Add scheduled execution**
```bash
# Run every day at 2 AM
0 2 * * * cd /path/to/ModelForge && python ModelForge/main.py

# Run every Monday at 6 AM
0 6 * * 1 cd /path/to/ModelForge && python ModelForge/main.py
```

3. **For Windows Task Scheduler**
   - Create Basic Task in Task Scheduler
   - Set action: "Start a program"
   - Program: `python`
   - Arguments: `ModelForge/main.py`
   - Start in: `C:\path\to\ModelForge`

## 📊 Sample Output

### Plots
![GidSearchCV plot](docs/grid_search_XGBoostClassifier.png) 
![ScatterMatrix](docs/scatter_matrix_20251113_150509.png) 
*Example of generated plots*

### Automated Report Example

![Sample Report](docs/report_1.png) 
![Sample Report](docs/report_2.png) 
*Example of generated PDF report with metrics and visualizations*

### Pipeline Execution
Start:
![Script Execution](docs/Data_preparation.png)
End:
![Script Execution](docs/Experiment_results.png)
*Interactive command-line interface with step-by-step guidance*

## 🖥️ UI Interface (In Development)

![UI Interface](docs/UI_example.png)
*PyQt5-based desktop application for visual pipeline management (in progress)*

## 📁 Project Structure

```
ModelForge/
├── modules/
│   ├── data_loading_to_dataFrame/    # Data source connectors
│   ├── data_validation/              # Cleaning and preprocessing
│   ├── data_analysis/                # Statistics and modeling
│   └── report/                       # Reporting and email
├── integration/                      # Database and API integration
├── ui/                              # PyQt5 interface (in development)
├── tests/                           # Comprehensive test suite
└── settings.py                      # Pydantic settings management
```

## 🔧 API Integration

Start the FastAPI server:

```bash
uvicorn ModelForge.integration.api_integration:app --reload
```

API endpoints:
- `POST /run-pipeline` - Execute full ML pipeline
- `GET /health` - Service health check

## 🧪 Testing

Run the test suite:

```bash
poetry run pytest tests/ -v
```

Test coverage includes:
- Data loading from all sources
- Validation and preprocessing functions
- Model training and evaluation
- Report generation

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📚 [Documentation](README.md)
- 🐛 [Issue Tracker](https://github.com/Ollldman/ModelForge/issues)

## 🚧 Roadmap

- [ ] Enhanced UI with PyQt5
- [ ] Additional ML models (Random Forest, Neural Networks)
- [ ] Time series analysis capabilities
- [ ] Cloud deployment (AWS, GCP, Azure)
- [ ] Real-time data streaming support
- [ ] Advanced hyperparameter optimization (Bayesian, Optuna)

---

**ModelForge** - Forging better models, automatically. 🔥


![Codewars](https://www.codewars.com/users/Ollldman/badges/micro)
