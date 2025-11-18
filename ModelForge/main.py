import os
import sys
import logging
import pickle
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ModelForge.logger_config import setup_logging
from ModelForge.modules.data_loading_to_dataFrame.load_from_static import load_from_csv, load_from_excel
from ModelForge.modules.data_loading_to_dataFrame.load_from_api import load_dataset
from ModelForge.modules.data_loading_to_dataFrame.acync_load_from_db import async_sql_to_df
from ModelForge.modules.data_validation.handle_missing_values import handle_missing_values, check_dataset
from ModelForge.modules.data_validation.remove_dublicate_rows import remove_duplicate_rows
from ModelForge.modules.data_validation.feature_encoding import encode_categorical_column
from ModelForge.modules.data_validation.normalize_numerical_columns import normalize_numerical_columns
from ModelForge.modules.data_analysis.data_statistics import compute_descriptive_statistics
from ModelForge.modules.data_analysis.show_result_plot import plot_scatter_matrix_with_histograms, plot_grid_search_results
from ModelForge.modules.data_analysis.gboost_models import xgb_run_classification_grid_search
from ModelForge.modules.report.report_generator import generate_automl_report
from ModelForge.modules.report.report_data import ReportData
from ModelForge.modules.report.send_report_by_email import send_report_via_email
from ModelForge.integration.db_storage import save_results_to_db
from ModelForge.settings import settings

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def setup_experiment_directories():
    """Create necessary directories for the experiment"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    directories = {
        'models': f'models/experiment_{timestamp}',
        'plots': f'plots/experiment_{timestamp}', 
        'reports': f'reports/experiment_{timestamp}',
        'logs': f'logs/experiment_{timestamp}'
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories, timestamp

def select_data_source() -> Tuple[Dict[str, Any], str]:
    """Interactive data source selection"""
    print("### DATA LOADING ###")
    print("Select data source:")
    print("1. CSV File")
    print("2. Excel File") 
    print("3. Database (SQL)")
    print("4. Kaggle Dataset")
    print("5. Hugging Face Dataset")
    print("6. UCI Dataset")
    print("7. Sklearn Built-in Dataset")

    dataset_name: str = input("Enter dataset name: ")
    
    choice = input("Enter choice (1-7): ").strip()
    
    if choice == "1":
        file_path = input("Enter CSV file path: ").strip()
        return {"source": "csv", "path": file_path}, dataset_name
    
    elif choice == "2":
        file_path = input("Enter Excel file path: ").strip()
        sheet_name = input("Enter sheet name (optional, press Enter for default): ").strip()
        return {"source": "excel", "path": file_path, "sheet_name": sheet_name or 0}, dataset_name
    
    elif choice == "3":
        connection_string = input("Enter database connection string: ").strip()
        sql_query = input("Enter SQL query: ").strip()
        return {"source": "database", "connection_string": connection_string, "sql_query": sql_query}, dataset_name
    
    elif choice == "4":
        dataset_id = input("Enter Kaggle dataset ID (format: username/dataset-name): ").strip()
        filename = input("Enter filename within dataset: ").strip()
        return {"source": "kaggle", "dataset_id": dataset_id, "filename": filename}, dataset_name
    
    elif choice == "5":
        dataset_id = input("Enter Hugging Face dataset ID: ").strip()
        return {"source": "huggingface", "dataset_id": dataset_id}, dataset_name
    
    elif choice == "6":
        dataset_id = input("Enter UCI dataset ID (number): ").strip()
        return {"source": "uci", "dataset_id": int(dataset_id)}, dataset_name
    
    elif choice == "7":
        dataset_name = input("Enter sklearn dataset name (e.g., load_iris): ").strip()
        return {"source": "sklearn", "name": dataset_name}, dataset_name
    
    else:
        print("Invalid choice. Using Iris dataset as default.")
        return {"source": "sklearn", "name": "load_iris"}, dataset_name

def load_data(source_config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Load data based on source configuration"""
    try:
        source = source_config["source"]
        
        if source == "csv":
            return load_from_csv(source_config["path"])
        elif source == "excel":
            return load_from_excel(source_config["path"], source_config.get("sheet_name", 0)) #type:ignore
        elif source == "database":
            return async_sql_to_df(source_config["connection_string"], source_config["sql_query"]) #type:ignore
        elif source == "kaggle":
            return load_dataset('kaggle', dataset_id=source_config["dataset_id"], 
                              filename=source_config["filename"])
        elif source == "huggingface":
            return load_dataset('huggingface', dataset_id=source_config["dataset_id"])
        elif source == "uci":
            return load_dataset('uci', dataset_id=source_config["dataset_id"])
        elif source == "sklearn":
            return load_dataset('sklearn', name=source_config["name"])
        else:
            logging.error(f"Unknown data source: {source}")
            return None
            
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def select_target_column(df: pd.DataFrame) -> str:
    """Interactive target column selection"""
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns):
        print(f"{i}: {col} (dtype: {df[col].dtype})")
    
    while True:
        try:
            col_index = int(input(f"\nSelect target column index (0-{len(df.columns)-1}): "))
            if 0 <= col_index < len(df.columns):
                return df.columns[col_index]
            else:
                print("Invalid index. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def configure_grid_search() -> Dict[str, Any]:
    """Interactive GridSearch configuration"""
    print("\n### MODEL CONFIGURATION ###")
    print("Configure XGBoost GridSearch parameters (press Enter for defaults):")
    
    param_grid = {}
    
    # n_estimators
    n_estimators = input("n_estimators (comma-separated, default: 100,200): ").strip()
    if n_estimators:
        param_grid["n_estimators"] = [int(x.strip()) for x in n_estimators.split(",")]
    
    # max_depth  
    max_depth = input("max_depth (comma-separated, default: 3,6): ").strip()
    if max_depth:
        param_grid["max_depth"] = [int(x.strip()) for x in max_depth.split(",")]
    
    # learning_rate
    learning_rate = input("learning_rate (comma-separated, default: 0.01,0.1,0.2): ").strip()
    if learning_rate:
        param_grid["learning_rate"] = [float(x.strip()) for x in learning_rate.split(",")]
    
    # subsample
    subsample = input("subsample (comma-separated, default: 0.8,1.0): ").strip()
    if subsample:
        param_grid["subsample"] = [float(x.strip()) for x in subsample.split(",")]
    
    cv = input(f"Cross-validation folds (default: 5): ").strip()
    cv = int(cv) if cv else 5
    
    return {"param_grid": param_grid, "cv": cv}

def create_report_data_structure(directories: Dict[str, str], timestamp: str) -> ReportData:
    """Create and initialize ReportData instance with experiment metadata"""
    return ReportData(
        project_name=f"AutoML_Experiment_{timestamp}",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        business_understanding=None,
        data_understanding=None,
        data_preparation=None,
        modeling=None,
        evaluation=None,
        deployment=None
    )

def setup_logging_with_experiment(log_dir: str, timestamp: str):
    """
    Setup logging with experiment-specific log file
    """
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Create experiment-specific log file
    log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # File handler for experiment log
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    
    # Configure specific loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    logging.info(f"Experiment logging configured. Log file: {log_file}")

def main():
    """Main execution function following CRISP-DM methodology with validated data structure"""
    
    # Setup logging and directories
    setup_logging()
    
    # Set matplotlib to only show warnings and errors
    import matplotlib
    matplotlib.set_loglevel('warning')  # Это специально для matplotlib
    
    # Также через logging
    import logging
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

    directories, timestamp = setup_experiment_directories()
    # Setup logging with experiment-specific log file
    setup_logging_with_experiment(directories['logs'], timestamp)

    logger = logging.getLogger(__name__)
    
    # Initialize validated report data structure
    report_data = create_report_data_structure(directories, timestamp)

    # Log experiment start
    logger.info(f"=== Starting AutoML Experiment {timestamp} ===")
    logger.info(f"Experiment directories: {directories}")
    
    try:
        ### STEP 1: BUSINESS UNDERSTANDING ###
        clear_screen()
        print("### BUSINESS UNDERSTANDING ###")
        report_data.business_understanding = {
            "goal": input("Enter business goal: ").strip() or "Automated classification task",
            "success_metric": input("Enter success metric: ").strip() or "Accuracy score"
        }
        logger.info(f"Business understanding - Goal: {report_data.business_understanding['goal']}")
        logger.info(f"Business understanding - Success metric: {report_data.business_understanding['success_metric']}")
        
        ### STEP 2: DATA LOADING ###
        source_config, dataset_name = select_data_source()
        df = load_data(source_config)
        
        if df is None or df.empty:
            logging.error("Failed to load data. Exiting.")
            return
        
        print(f"Data loaded successfully. Shape: {df.shape}")
        logger.info(f"Data loaded successfully from {source_config['source']}. Shape: {df.shape}")
        
        ### STEP 3: DATA UNDERSTANDING ###
        print("### DATA UNDERSTANDING ###")
        
        # Show basic info
        print("\nDataset Info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Compute statistics
        stats = compute_descriptive_statistics(df)
        
        print("\nDescriptive Statistics:")
        for col, col_stats in stats.items():
            print(f"\n{col}:")
            print(f"  Count: {col_stats['count']}")
            print(f"  Mean: {col_stats['mean']:.4f}")
            print(f"  Std: {col_stats['std']:.4f}")
            print(f"  Min: {col_stats['min']:.4f}")
            print(f"  Max: {col_stats['max']:.4f}")
        
        # Check for missing values
        missing_report = check_dataset(df)
        print(f"\nMissing Values Report: {missing_report}")
        
        # Create scatter plot matrix
        plot_paths = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            plot_path = os.path.join(directories['plots'], f"scatter_matrix_{timestamp}.png")
            try:
                plot_scatter_matrix_with_histograms(
                    df, 
                    numeric_columns=numeric_cols[:5],  # Limit to first 5 numeric columns
                    output_path=plot_path
                )
                plot_paths.append(plot_path)
                print(f"Scatter matrix saved to: {plot_path}")
            except Exception as e:
                logging.warning(f"Could not create scatter matrix: {e}")
        
        # Update report data with data understanding
        report_data.data_understanding = {
            "num_rows": len(df),
            "num_cols": len(df.columns),
            "columns": df.columns.tolist(),
            "statistics_summary": {col: f"mean={stats[col]['mean']:.2f}, std={stats[col]['std']:.2f}" 
                                 for col in stats.keys()},
            "missing_report": missing_report,
            "plot_paths": plot_paths
        }
        logger.info(f"Descriptive statistics computed for {len(stats)} columns")
        logger.info(f"Missing values report: {missing_report}")
        
        ### STEP 4: DATA PREPARATION ###
        clear_screen()
        print("### DATA PREPARATION ###")
        
        # Select target column
        target_column = select_target_column(df)
        
        # Handle missing values
        print("\nHandling missing values...")
        df_clean = handle_missing_values(
            df,
            numeric_strategy="median",
            categorical_strategy="most_frequent"
        )
        
        # Remove duplicates
        print("Removing duplicates...")
        dedup_result = remove_duplicate_rows(df_clean)
        if dedup_result.status == "success":
            df_clean = pd.DataFrame(dedup_result.cleaned_dataframe)
            print(f"Removed {len(dedup_result.removed_duplicates)} duplicate rows") # type:ignore
        
        # Encode categorical features
        print("Encoding categorical features...")
        categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
        encoding_steps = []
        for col in categorical_cols:
            if col != target_column:
                df_clean = encode_categorical_column(df_clean, col, method="one_hot")
                encoding_steps.append(f"One-hot encoded: {col}")
        
        # Encode target if categorical
        if df_clean[target_column].dtype in ['object', 'category']:
            df_clean = encode_categorical_column(df_clean, target_column, method="label")
            encoding_steps.append(f"Label encoded target: {target_column}")
        
        # Update report data with preparation steps
        report_data.data_preparation = {
            "target_column": target_column,
            "steps": [
                "Missing value imputation (median for numeric, most_frequent for categorical)",
                f"Duplicate removal ({dedup_result.status})"
            ] + encoding_steps,
            "final_shape": df_clean.shape,
            "plot_paths": []  # Could add preprocessing visualization paths here
        }
        
        print(f"Data preparation completed. Final shape: {df_clean.shape}")
        logger.info(f"Data preparation completed. Target column: {target_column}")
        logger.info(f"Final dataset shape: {df_clean.shape}")
        
        ### STEP 5: MODELING ###
        clear_screen()
        print("### MODELING ###")
        
        # Prepare features and target
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Configure and run GridSearch
        grid_config = configure_grid_search()
        model_results = xgb_run_classification_grid_search(
            X_train.values, y_train.values, 
            X_test.values, y_test.values,
            param_grid=grid_config["param_grid"] or None,  # Use None to get defaults
            cv=grid_config["cv"]
        )
        
        # Save model
        model_path = os.path.join(directories['models'], f"xgb_model_{timestamp}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_results["best_estimator"], f)
        
        # Create GridSearch plot
        plot_path = os.path.join(directories['plots'], f"grid_search_{timestamp}.png")
        plot_grid_search_results(
            model_results["plot_data"],
            "XGBoostClassifier",
            output_dir=directories['plots']
        )
        
        # Update report data with modeling results
        report_data.modeling = {
            "model_name": "XGBoostClassifier",
            "best_params": model_results["best_params"],
            "best_score": float(model_results["best_score"]),
            "plot_paths": [plot_path],
            "model_path": model_path,
            "grid_search_config": {
                "cv_folds": grid_config["cv"],
                "param_grid_used": grid_config["param_grid"] or "default"
            }
        }
        
        print(f"\nBest parameters: {model_results['best_params']}")
        print(f"Best CV score: {model_results['best_score']:.4f}")
        print(f"Test metrics: {model_results['test_metrics']}")
        print(f"Model saved to: {model_path}")
        logger.info(f"Model training completed. Best score: {model_results['best_score']:.4f}")
        logger.info(f"Best parameters: {model_results['best_params']}")
        logger.info(f"Test metrics: {model_results['test_metrics']}")
        
        ### STEP 6: EVALUATION ###
        clear_screen()
        print("### EVALUATION ###")
        
        report_data.evaluation = {
            "metrics": model_results["test_metrics"],
            "interpretation": f"Model achieved {model_results['test_metrics']['accuracy']:.4f} accuracy on test set"
        }
        
        print("Model Evaluation Results:")
        for metric, value in model_results["test_metrics"].items():
            print(f"{metric}: {value:.4f}")
        
        logger.info("Model evaluation completed successfully")
        
        ### STEP 7: DEPLOYMENT & REPORTING ###
        clear_screen()
        print("### DEPLOYMENT & REPORTING ###")
        
        # Generate report using the validated ReportData structure
        report_path = generate_automl_report(
            report_data.dict(),  # Convert Pydantic model to dict for compatibility
            output_dir=directories['reports'],
            project_name=f"AutoML_Experiment_{timestamp}",
            save_pdf=True
        )
        
        deployment_info = {
            "report_path": report_path,
            "timestamp": timestamp,
            "model_deployment_path": model_path,
            "directories": str(directories),
            "recommendations": "Model is ready for production use. Monitor performance regularly."
        }
        
        if report_path:
            report_data.deployment = deployment_info
            print(f"Report generated: {report_path}")
            
            # Email report
            send_email = input("\nSend report via email? (y/n): ").strip().lower()
            if send_email == 'y':
                recipient = input("Enter recipient email: ").strip()
                if recipient:
                    success = send_report_via_email(
                        recipient_emails=[recipient],
                        subject=f"AutoML Report - Experiment {timestamp}",
                        body="Please find attached the AutoML experiment report.",
                        attachment_paths=[report_path]
                    )
                    if success:
                        print("Report sent via email successfully.")
                        deployment_info["email_sent_to"] = recipient
                    else:
                        print("Failed to send email.")
                        deployment_info["email_status"] = "failed"
        
        # Final update of deployment info
        report_data.deployment = deployment_info
        
        print("\n### EXPERIMENT COMPLETED SUCCESSFULLY ###")
        print(f"Model saved: {model_path}")
        print(f"Plots saved: {directories['plots']}")
        print(f"Report saved: {directories['reports']}")
        print(f"Logs saved: {directories['logs']}")

        logger.info(f"Report generated: {report_path}")
        logger.info("Experiment completed successfully")
        
        # Validate final report data
        if report_data.is_complete():
            print("✓ Report data validation: COMPLETE")
        else:
            print("⚠ Report data validation: INCOMPLETE (missing required sections)")

        # save to db
        logger.info("Step 9: Save to DB (to be implemented)")
        logger.info("Saving results to DB...")
        save_results_to_db(
            model_name=model_results["best_estimator"].__class__.__name__,
            dataset_name=dataset_name,
            metrics=model_results["test_metrics"],
            best_params=model_results["best_params"]
        )

        logger.info("Pipeline completed successfully.")
        
    except Exception as e:
        logging.error(f"Experiment failed: {e}", exc_info=True)
        print(f"Experiment failed with error: {e}")
        
        # Even on failure, attempt to generate partial report
        try:
            if report_data:
                partial_report_path = generate_automl_report(
                    report_data.model_dump(),
                    output_dir=directories['reports'],
                    project_name=f"Partial_Report_{timestamp}",
                    save_pdf=True
                )
                if partial_report_path:
                    logger.info(f"Partial report generated: {partial_report_path}")
                    print(f"Partial report generated: {partial_report_path}")
        except Exception as report_error:
            logging.error(f"Failed to generate partial report: {report_error}")
        
        return

if __name__ == "__main__":
    main()