import json
import argparse
from pathlib import Path

# Threshold to filter out noisy low-confidence object detections
CONFIDENCE_THRESHOLD = 0.4

# Max times a single object class can be repeated in the text
# (prevents BM25 spamming, e.g., 40 "Tree"s)
MAX_REPETITIONS = 2

def merge_objects(metadata_path: str, objects_dir: str):
    meta_file = Path(metadata_path)
    if not meta_file.exists():
        print(f"Error: {meta_file} not found. Please run extractor.py first.")
        return

    print(f"Loading metadata from {meta_file}...")
    with open(meta_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    obj_dir_path = Path(objects_dir)
    if not obj_dir_path.exists():
        print(f"Error: Object directory {obj_dir_path} not found.")
        return

    processed_count = 0
    missing_count = 0

    for record in records:
        video_name = record.get("video_name")
        frame_index = record.get("frame_index")
        
        if not video_name or frame_index is None:
            continue

        # Format: objects/L21_V001/017.json
        obj_file = obj_dir_path / video_name / f"{frame_index:03d}.json"
        
        if not obj_file.exists():
            missing_count += 1
            record["objects"] = ""
            continue

        try:
            with open(obj_file, "r", encoding="utf-8") as f:
                obj_data = json.load(f)
            
            scores = obj_data.get("detection_scores", [])
            entities = obj_data.get("detection_class_entities", [])
            
            # Count occurrences to avoid spamming
            entity_counts = {}
            filtered_entities = []
            
            for score_str, entity in zip(scores, entities):
                score = float(score_str)
                if score >= CONFIDENCE_THRESHOLD:
                    count = entity_counts.get(entity, 0)
                    if count < MAX_REPETITIONS:
                        filtered_entities.append(entity)
                        entity_counts[entity] = count + 1
                        
            # Join into a single string for BM25
            record["objects"] = " ".join(filtered_entities)
            processed_count += 1
            
        except Exception as e:
            print(f"Error reading {obj_file}: {e}")
            record["objects"] = ""
            missing_count += 1

    print(f"Processed {processed_count} frames with objects.")
    if missing_count > 0:
        print(f"Warning: {missing_count} frames were missing object JSON files.")

    print(f"Saving merged metadata back to {meta_file}...")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    
    print("Merge complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge object detection JSONs into metadata.json")
    parser.add_argument("--metadata", default="data/metadata/metadata.json", help="Path to metadata.json")
    parser.add_argument("--objects-dir", default="data/objects", help="Path to objects root directory")
    
    args = parser.parse_args()
    merge_objects(args.metadata, args.objects_dir)
