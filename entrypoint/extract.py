import argparse
import sys
import os
import json
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingest.loader import get_loader
from src.ingest.metadata import MetadataExtractor

def main():
    parser = argparse.ArgumentParser(description="RAG ChatBot: Metadata Extraction")
    parser.add_argument("--file", type=str, help="Path to a single file to process")
    parser.add_argument("--dir", type=str, default="data/files", help="Directory to process all files from")
    parser.add_argument("--all", action="store_true", help="Process all files in the directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process")
    parser.add_argument("--output", type=str, default="data/metadata.json", help="Output JSON file path")
    
    args = parser.parse_args()
    
    extractor = MetadataExtractor()
    results = []

    files_to_process = []
    if args.file:
        files_to_process.append(args.file)
    elif args.all:
        if not os.path.exists(args.dir):
            print(f"Directory not found: {args.dir}")
            return
        files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if os.path.isfile(os.path.join(args.dir, f))]
        files_to_process.extend(files)
    else:
        print("Please specify --file or --all")
        return

    if args.limit:
        files_to_process = files_to_process[:args.limit]
        print(f"Limiting to specific {args.limit} files.")

    print(f"Processing {len(files_to_process)} files...")
    
    for file_path in tqdm(files_to_process):
        try:
            # Load text (use first 4000 chars roughly)
            loader = get_loader(file_path)
            docs = loader.load()
            
            if not docs:
                continue
                
            full_text = "\n".join([d.page_content for d in docs])
            
            # Extract
            metadata = extractor.extract(full_text)
            
            # Convert to dict
            meta_dict = metadata.model_dump()
            meta_dict["source_file"] = os.path.basename(file_path)
            
            results.append(meta_dict)
            
            # Print for single file mode
            if args.file:
                print(json.dumps(meta_dict, ensure_ascii=False, indent=2))
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Save results
    if args.all or len(results) > 1:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved extracted metadata to {args.output}")

if __name__ == "__main__":
    main()
