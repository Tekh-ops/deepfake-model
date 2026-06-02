import os
import shutil
import glob

# Mappings of original file -> new file
MOVES = {
    "models/stream_a.py": "modules/image_analysis/spatial_detector.py",
    "models/efficientnet_detector.py": "modules/image_analysis/efficientnet_detector.py",
    "models/vit_detector.py": "modules/image_analysis/vit_detector.py",
    "models/xception_detector.py": "modules/image_analysis/xception_detector.py",
    "utils/gradcam.py": "modules/image_analysis/gradcam.py",
    
    "models/stream_b.py": "modules/video_analysis/frequency_detector.py",
    "models/stream_c.py": "modules/video_analysis/temporal_detector.py",
    "models/rppg_detector.py": "modules/video_analysis/rppg_detector.py",
    
    "models/audio_detector.py": "modules/audio_analysis/audio_detector.py",
    
    "utils/metadata.py": "modules/metadata_analysis/metadata.py",
    "utils/hashing.py": "modules/metadata_analysis/hashing.py",
    
    "models/detector.py": "modules/core/detector.py",
    "models/ensemble.py": "modules/core/ensemble.py",
    
    "utils/video_utils.py": "modules/core/video_utils.py",
    "utils/preprocessor.py": "modules/core/preprocessor.py",
    "utils/legal.py": "modules/core/legal.py", # Just in case
}

# Import replacements to perform on all python files
REPLACEMENTS = {
    "from modules.image_analysis.spatial_detector": "from modules.image_analysis.spatial_detector",
    "import modules.image_analysis.spatial_detector": "import modules.image_analysis.spatial_detector",
    
    "from modules.video_analysis.frequency_detector": "from modules.video_analysis.frequency_detector",
    "from modules.video_analysis.temporal_detector": "from modules.video_analysis.temporal_detector",
    "from modules.video_analysis.rppg_detector": "from modules.video_analysis.rppg_detector",
    
    "from modules.audio_analysis.audio_detector": "from modules.audio_analysis.audio_detector",
    
    "from modules.core.detector": "from modules.core.detector",
    "import modules.core.detector": "import modules.core.detector",
    
    "from modules.core.ensemble": "from modules.core.ensemble",
    
    "from modules.image_analysis.efficientnet_detector": "from modules.image_analysis.efficientnet_detector",
    "from modules.image_analysis.vit_detector": "from modules.image_analysis.vit_detector",
    "from modules.image_analysis.xception_detector": "from modules.image_analysis.xception_detector",
    
    "from modules.image_analysis.gradcam": "from modules.image_analysis.gradcam",
    "from modules.metadata_analysis.metadata": "from modules.metadata_analysis.metadata",
    "from modules.metadata_analysis.hashing": "from modules.metadata_analysis.hashing",
    "from modules.core.video_utils": "from modules.core.video_utils",
    "from modules.core.preprocessor": "from modules.core.preprocessor",
    "from modules.core.legal": "from modules.core.legal",
    
    "from modules.factcheck": "from modules.factcheck",
    "import modules.factcheck": "import modules.factcheck",
}

def create_dirs():
    dirs = [
        "modules/image_analysis",
        "modules/video_analysis",
        "modules/audio_analysis",
        "modules/metadata_analysis",
        "modules/factcheck",
        "modules/core"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        init_file = os.path.join(d, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write("")

def move_files():
    for src, dst in MOVES.items():
        if os.path.exists(src):
            print(f"Moving {src} -> {dst}")
            shutil.move(src, dst)
        else:
            print(f"Warning: {src} not found")
            
    # Move factcheck dir contents
    if os.path.exists("factcheck") and os.path.isdir("factcheck"):
        for item in os.listdir("factcheck"):
            src = os.path.join("factcheck", item)
            dst = os.path.join("modules", "factcheck", item)
            if not os.path.exists(dst):
                print(f"Moving {src} -> {dst}")
                shutil.move(src, dst)
        # Try to remove empty dir
        try:
            os.rmdir("factcheck")
        except:
            pass

def update_imports():
    # Find all python files
    py_files = glob.glob("**/*.py", recursive=True)
    
    # Exclude venv and cache
    py_files = [f for f in py_files if ".venv" not in f and "__pycache__" not in f]
    
    for fpath in py_files:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
            
        new_content = content
        for old_str, new_str in REPLACEMENTS.items():
            new_content = new_content.replace(old_str, new_str)
            
        if new_content != content:
            print(f"Updating imports in {fpath}")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(new_content)

if __name__ == "__main__":
    print("Starting refactor...")
    create_dirs()
    move_files()
    update_imports()
    print("Refactor complete.")
