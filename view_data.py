import json
from database import Database, WEBSITES

def view_database_stats():
    """View statistics from the SQLite database."""
    db = Database(WEBSITES)
    stats = db.get_traces_collected()
    
    print("\n=== Database Statistics ===")
    print("Traces collected per website:")
    for website, count in stats.items():
        print(f"- {website}: {count} traces")

def view_json_data():
    """View data from the JSON file."""
    try:
        with open('dataset.json', 'r') as f:
            data = json.load(f)
            
        print("\n=== JSON Data Statistics ===")
        print(f"Total traces: {len(data)}")
        
        # Group by website
        website_counts = {}
        for item in data:
            website = item['website']
            website_counts[website] = website_counts.get(website, 0) + 1
        
        print("\nTraces per website:")
        for website, count in website_counts.items():
            print(f"- {website}: {count} traces")
            
        # Show sample trace data
        if data:
            print("\nSample trace data (first trace):")
            sample = data[0]
            print(f"Website: {sample['website']}")
            print(f"Trace length: {len(sample['trace_data'])}")
            print(f"First few values: {sample['trace_data'][:5]}")
            
    except FileNotFoundError:
        print("\nNo dataset.json file found. Run collect.py first to collect data.")
    except json.JSONDecodeError:
        print("\nError reading dataset.json. File might be corrupted.")

if __name__ == "__main__":
    print("Website Fingerprinting Data Viewer")
    print("=================================")
    
    view_database_stats()
    view_json_data() 