import os
import shutil
from pathlib import Path

def organize_files():
    # Source directory containing all face subdirectories
    source_dir = Path("data/data/faces")
    # Target emotion directories
    target_base = Path("data/emotions")
    emotions = ["neutral", "happy", "sad", "angry"]
    
    # Ensure target directories exist
    for emotion in emotions:
        (target_base / emotion).mkdir(parents=True, exist_ok=True)
    
    # Process each person's directory
    for person_dir in source_dir.iterdir():
        if not person_dir.is_dir():
            continue
            
        # Process each .pgm file
        for pgm_file in person_dir.glob("*.pgm"):
            # Extract emotion from filename
            filename = pgm_file.name.lower()
            
            # Determine which emotion directory this file belongs to
            target_emotion = None
            for emotion in emotions:
                if emotion in filename:
                    target_emotion = emotion
                    break
            
            if target_emotion:
                # Create new filename to avoid conflicts
                new_filename = f"{person_dir.name}_{pgm_file.name}"
                target_path = target_base / target_emotion / new_filename
                
                # Copy file to appropriate emotion directory
                shutil.copy2(pgm_file, target_path)
                print(f"Copied {pgm_file} to {target_path}")

if __name__ == "__main__":
    organize_files() 