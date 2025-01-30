import streamlit as st
import requests
import json
import time
from datetime import datetime
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.logger import setup_logger

# Setup logger
logger = setup_logger("streamlit")

# API endpoint
API_URL = "http://localhost:8080"

def format_timestamp(timestamp_str):
    if not timestamp_str:
        return "N/A"
    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def submit_job():
    st.header("Submit Training Job")
    
    with st.form("job_submission"):
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        # First column: Basic job settings
        with col1:
            st.subheader("Basic Settings")
            model_name = st.text_input("Model Name")
            dataset_path = st.text_input("Dataset Path")
            gpu_count = st.number_input("Number of GPUs", min_value=1, max_value=4, value=1)
            
            # Move submit button to bottom of first column
            submitted = st.form_submit_button("Submit Job", use_container_width=True)
        
        # Second column: Hyperparameters
        with col2:
            st.subheader("Hyperparameters")
            st.caption("Configure training parameters in JSON format")
            hyperparameters = st.text_area(
                "",  # Remove label since we have a subheader
                value=json.dumps({
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "epochs": 10,
                    "optimizer": "adam",
                    "loss": "categorical_crossentropy"
                }, indent=2),
                height=300  # Make text area taller
            )
        
        if submitted:
            try:
                logger.info(f"Submitting job with model: {model_name}")
                # Validate hyperparameters JSON
                hyperparams = json.loads(hyperparameters)
                
                # Prepare job data
                job_data = {
                    "model_name": model_name,
                    "dataset_path": dataset_path,
                    "hyperparameters": hyperparams,
                    "gpu_count": gpu_count
                }
                
                # Submit job
                response = requests.post(f"{API_URL}/submit_job", json=job_data)
                response.raise_for_status()
                
                logger.info(f"Job submitted successfully: {response.json()['job_id']}")
                st.success(f"Job submitted successfully! Job ID: {response.json()['job_id']}")
            except json.JSONDecodeError:
                st.error("Invalid JSON format in hyperparameters")
            except Exception as e:
                logger.error(f"Error submitting job: {str(e)}", exc_info=True)
                st.error(f"Error submitting job: {str(e)}")

def display_gpu_status():
    st.header("GPU Status")
    
    try:
        logger.debug("Fetching GPU status")
        response = requests.get(f"{API_URL}/gpu_status")

        st.write(f"GPU status response: {response.json()}")

        gpu_status = response.json()
        
        if not gpu_status["gpu_info"]:
            st.warning("No GPUs detected or nvidia-smi not accessible")
            return
            
        # Display GPU information in a table
        gpu_data = []
        for idx, gpu in enumerate(gpu_status["gpu_info"]):
            memory_used = float(gpu['memory_used'])
            memory_total = float(gpu['memory_total'])
            memory_free = memory_total - memory_used
            utilization = float(gpu['utilization'])
            
            # Add color coding based on utilization
            if utilization > 80:
                status = "🔴 High Load"
            elif utilization > 30:
                status = "🟡 Medium Load"
            else:
                status = "🟢 Available"
                
            gpu_data.append({
                "GPU": idx,
                "Status": status,
                "Memory Used": f"{memory_used/1024:.1f} GB",
                "Memory Free": f"{memory_free/1024:.1f} GB",
                "Memory Total": f"{memory_total/1024:.1f} GB",
                "Utilization": f"{utilization:.1f}%"
            })
        
        st.dataframe(
            gpu_data,
            hide_index=True,
            use_container_width=True
        )
        
        # Display free GPUs with better formatting
        free_count = len(gpu_status['free_gpus'])
        total_count = gpu_status['total_gpus']
        
        if free_count == 0:
            st.error(f"No GPUs available (0 of {total_count})")
        else:
            st.success(
                f"Available GPUs: {', '.join(map(str, gpu_status['free_gpus']))} "
                f"({free_count} of {total_count})"
            )
    
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend service. Is it running?")
    except Exception as e:
        logger.error(f"Error fetching GPU status: {str(e)}", exc_info=True)
        st.error(f"Error fetching GPU status: {str(e)}")

def display_jobs():
    st.header("Training Jobs")
    
    try:
        # Fetch all jobs
        response = requests.get(f"{API_URL}/jobs")
        jobs = response.json()
        
        # Group jobs by status
        running_jobs = []
        queued_jobs = []
        completed_jobs = []
        failed_jobs = []
        
        for job_id, job in jobs.items():
            job_info = {
                "Job ID": job_id[:8],
                "Model": job["data"]["model_name"],
                "GPUs": job["data"]["gpu_count"],
                "Created": format_timestamp(job["created_at"]),
                "Status": job["status"]
            }
            
            if job["status"] == "running":
                running_jobs.append(job_info)
            elif job["status"] == "queued":
                queued_jobs.append(job_info)
            elif job["status"] == "completed":
                completed_jobs.append(job_info)
            elif job["status"] == "failed":
                failed_jobs.append(job_info)
        
        # Display running jobs
        if running_jobs:
            st.subheader("Running Jobs")
            st.table(running_jobs)
        
        # Display queued jobs
        if queued_jobs:
            st.subheader("Queued Jobs")
            st.table(queued_jobs)
        
        # Display completed jobs
        if completed_jobs:
            st.subheader("Completed Jobs")
            st.table(completed_jobs)
        
        # Display failed jobs
        if failed_jobs:
            st.subheader("Failed Jobs")
            st.table(failed_jobs)
            
    except Exception as e:
        logger.error(f"Error fetching jobs: {str(e)}", exc_info=True)
        st.error(f"Error fetching jobs: {str(e)}")

def main():
    st.set_page_config(
        page_title="AI Training Pipeline",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("AI Training Pipeline")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Submit Job", "GPU Status", "Jobs"])
    
    with tab1:
        submit_job()
    
    with tab2:
        # Static header
        st.header("GPU Status")
        # Create containers for dynamic content
        gpu_table_container = st.empty()
        gpu_status_container = st.empty()
        gpu_timestamp = st.empty()
    
    with tab3:
        # Static header
        st.header("Training Jobs")
        jobs_container = st.empty()
    
    # Continuous refresh of dynamic content
    while True:
        # Update GPU status
        try:
            response = requests.get(f"{API_URL}/gpu_status")
            gpu_status = response.json()
            
            if not gpu_status["gpu_info"]:
                with gpu_status_container:
                    st.warning("No GPUs detected or nvidia-smi not accessible")
            else:
                # Prepare GPU data
                gpu_data = []
                for idx, gpu in enumerate(gpu_status["gpu_info"]):
                    memory_used = float(gpu['memory_used'])
                    memory_total = float(gpu['memory_total'])
                    memory_free = memory_total - memory_used
                    utilization = float(gpu['utilization'])
                    
                    status = ("🔴 High Load" if utilization > 80 else 
                             "🟡 Medium Load" if utilization > 30 else 
                             "🟢 Available")
                    
                    gpu_data.append({
                        "GPU": idx,
                        "Status": status,
                        "Memory Used": f"{memory_used/1024:.1f} GB",
                        "Memory Free": f"{memory_free/1024:.1f} GB",
                        "Memory Total": f"{memory_total/1024:.1f} GB",
                        "Utilization": f"{utilization:.1f}%"
                    })
                
                # Update table
                with gpu_table_container:
                    st.dataframe(
                        gpu_data,
                        hide_index=True,
                        use_container_width=True
                    )
                
                # Update status message
                free_count = len(gpu_status['free_gpus'])
                total_count = gpu_status['total_gpus']
                with gpu_status_container:
                    if free_count == 0:
                        st.error(f"No GPUs available (0 of {total_count})")
                    else:
                        st.success(
                            f"Available GPUs: {', '.join(map(str, gpu_status['free_gpus']))} "
                            f"({free_count} of {total_count})"
                        )
        except requests.exceptions.ConnectionError:
            with gpu_status_container:
                st.error("Cannot connect to backend service. Is it running?")
        except Exception as e:
            with gpu_status_container:
                st.error(f"Error fetching GPU status: {str(e)}")
        
        # Update jobs list
        with jobs_container:
            display_jobs()
        
        # Update timestamp
        with gpu_timestamp:
            st.write(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
        
        # Wait before next update
        time.sleep(10)

if __name__ == "__main__":
    main() 