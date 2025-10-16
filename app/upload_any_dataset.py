# upload_any_dataset.py
"""
Upload ANY dataset from your computer via simple drag-and-drop or file picker.
No hardcoded data. Works with ANY CSV or Excel file you have.
"""

import requests
import sys
from pathlib import Path

def upload_dataset(file_path: str, table_name: str = None):
    """
    Upload ANY dataset to the API.
    
    Args:
        file_path: Path to your CSV/Excel file (e.g., 'downloads/my_data.csv')
        table_name: Optional custom table name (auto-generated from filename if not provided)
    """
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"❌ Error: File not found: {file_path}")
        print("\nUsage:")
        print("  python upload_any_dataset.py /path/to/your/file.csv")
        print("  python upload_any_dataset.py ~/Downloads/sales_2025.xlsx")
        return
    
    # API endpoint
    api_url = "http://localhost:8000/upload-file"
    
    print(f"\n{'='*70}")
    print(f"📤 Uploading file: {file_path}")
    print(f"{'='*70}\n")
    
    # Open and upload file
    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {}
        if table_name:
            data['table_name'] = table_name
        
        try:
            response = requests.post(api_url, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                
                print("✅ Upload successful!\n")
                print(f"📊 File Info:")
                print(f"   - Original: {result['file_info']['original_filename']}")
                print(f"   - Type: {result['file_info']['file_type']}")
                print(f"   - Table: {result['file_info']['table_name']}")
                
                print(f"\n📈 Data Info:")
                print(f"   - Original rows: {result['data_info']['original_rows']}")
                print(f"   - Cleaned rows: {result['data_info']['cleaned_rows']}")
                print(f"   - Columns: {result['data_info']['cleaned_columns']}")
                print(f"   - Rows removed: {result['data_info']['rows_removed']}")
                
                print(f"\n🧹 Cleaning Operations Performed:")
                for op in result['cleaning_report']['operations']:
                    print(f"   {op}")
                
                print(f"\n💾 Files Saved:")
                print(f"   - Raw: {result['file_info']['raw_file_path']}")
                print(f"   - Cleaned: {result['file_info']['cleaned_file_path']}")
                
                print(f"\n📥 Download cleaned file:")
                print(f"   curl -O http://localhost:8000{result['download_links']['cleaned_csv']}")
                
                print(f"\n🌐 View in Dashboard:")
                print(f"   Open: http://localhost:8501")
                print(f"   Select table: {result['file_info']['table_name']}")
                
                print(f"\n{'='*70}")
                print("🎉 Success! Your dataset is ready to use.")
                print(f"{'='*70}\n")
                
            else:
                print(f"❌ Upload failed!")
                print(f"Status code: {response.status_code}")
                print(f"Error: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Error: Cannot connect to API server.")
            print("\nMake sure the FastAPI server is running:")
            print("  uvicorn app.main:app --reload")
            print("\nThen try again.")
        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n📁 Usage: Upload ANY dataset from your computer")
        print("="*70)
        print("\nPython:")
        print("  python upload_any_dataset.py /path/to/your/file.csv")
        print("  python upload_any_dataset.py ~/Downloads/sales_data.xlsx")
        print("  python upload_any_dataset.py C:\\Users\\You\\data.csv")
        print("\nOr use curl:")
        print('  curl -X POST "http://localhost:8000/upload-file" \\')
        print('    -F "file=@/path/to/your/file.csv"')
        print("\nOr use browser:")
        print("  Open: http://localhost:8000/docs")
        print("  Click: POST /upload-file")
        print("  Upload your file")
        print("\n" + "="*70)
        sys.exit(1)
    
    file_path = sys.argv[1]
    table_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    upload_dataset(file_path, table_name)