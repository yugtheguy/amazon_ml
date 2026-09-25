import sys
import platform
import subprocess
import json
import os

def get_nvidia_smi():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version,pstate", "--format=csv,noheader"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "N/A"

def check_env():
    env_info = {
        "os": platform.platform(),
        "python": sys.version,
        "nvidia-smi": get_nvidia_smi()
    }
    try:
        import cuml
        env_info["cuml"] = cuml.__version__
    except ImportError:
        env_info["cuml"] = "Not Installed"

    try:
        import cupy
        env_info["cupy"] = cupy.__version__
    except ImportError:
        env_info["cupy"] = "Not Installed"
        
    try:
        import cudf
        env_info["cudf"] = cudf.__version__
    except ImportError:
        env_info["cudf"] = "Not Installed"

    print(json.dumps(env_info, indent=4))
    
    # Save to artifacts if in Kaggle
    artifact_dir = os.environ.get("KAGGLE_ARTIFACT_DIR", "artifacts/retrieval/E001-C")
    os.makedirs(artifact_dir, exist_ok=True)
    
    with open(os.path.join(artifact_dir, "environment.json"), "w") as f:
        json.dump(env_info, f, indent=4)
        
if __name__ == "__main__":
    check_env()
