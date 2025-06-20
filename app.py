import torch  




from flask import Flask, send_from_directory, request, jsonify
# additional imports
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib




# ////////////////////////////////////////////////////////////////////////
from train import FingerprintClassifier, ComplexFingerprintClassifier, INPUT_SIZE  


_sites = ["CseBuet", "Google", "ProthomAlo"]
# _sites_name=["CseBuet", "Google", "ProthomAlo"]
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

# instantiate & load models
_model_simple = FingerprintClassifier(INPUT_SIZE, hidden_size=128, num_classes=len(_sites))  
_model_complex = ComplexFingerprintClassifier(INPUT_SIZE, hidden_size=128, num_classes=len(_sites))  
_model_simple.load_state_dict(torch.load(os.path.join("saved_models", "simple_model.pth"), map_location=_device))  
_model_complex.load_state_dict(torch.load(os.path.join("saved_models", "complex_model.pth"), map_location=_device))  
_model_simple.to(_device).eval()  
_model_complex.to(_device).eval()







matplotlib.use('Agg')

app = Flask(__name__)

stored_traces = []
stored_heatmaps = []

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/collect_trace', methods=['POST'])
def collect_trace():
    try:
        data = request.get_json()
        trace = data.get('trace')
        website = data.get('website', 'unknown')  # Get website from request data
        
        if not trace:
            return jsonify({'error': 'No trace data provided'}), 400

        # Calculate stats
        arr = np.array(trace)
        min_val = int(np.min(arr))
        max_val = int(np.max(arr))
        range_val = int(max_val - min_val)
        samples = int(len(arr))

        # Generate heatmap
        fig, ax = plt.subplots(figsize=(8, 2))
        heatmap = arr.reshape(1, -1)
        im = ax.imshow(heatmap, aspect='auto', cmap='hot')
        ax.axis('off')

        # Save heatmap image with website name
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        # Clean website name for filename
        website_name = website.replace('https://', '').replace('http://', '').replace('/', '_').replace('.', '_')
        filename = f'heatmap_{website_name}_{timestamp}.png'
        heatmap_path = os.path.join('static', filename)
        plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Store trace and heatmap
        stored_traces.append(trace)
        stored_heatmaps.append(filename)

        # Return the heatmap image URL and stats
        # return jsonify({
        #     'heatmap_url': f'/{filename}',
        #     'min': min_val,
        #     'max': max_val,
        #     'range': range_val,
        #     'samples': samples
        # })




                # ——— prepare and run prediction ———  
        # pad/trim to INPUT_SIZE  
        arr_p = np.zeros((INPUT_SIZE,), dtype=np.float32)  
        arr_p[: len(arr)] = arr[:INPUT_SIZE]  
        x = torch.from_numpy(arr_p).unsqueeze(0).to(_device)  
        with torch.no_grad():  
            ps = _model_simple(x).argmax(dim=1).item()  
            pc = _model_complex(x).argmax(dim=1).item()  
        pred_simple = _sites[ps]  
        pred_complex = _sites[pc]  

        # print(pred_simple)
        # print(pred_complex)
        # Return the heatmap image URL, stats, AND predictions  
        return jsonify({  
            'heatmap_url': f'/{filename}',  
            'min': min_val,  
            'max': max_val,  
            'range': range_val,  
            'samples': samples,  
            'prediction_simple': pred_simple,  
            'prediction_complex': pred_complex  
        })  



    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/download_traces')
def download_traces():
    """Download all collected traces."""
    try:
        # Return the stored traces
        formatted_traces = []
        for i, trace in enumerate(stored_traces):
            formatted_trace = {
                'trace_data': trace,
                'heatmap_url': f'/{stored_heatmaps[i]}' if i < len(stored_heatmaps) else None
            }
            formatted_traces.append(formatted_trace)
        
        return jsonify(formatted_traces)
    except Exception as e:
        print(f"Error downloading traces: {e}")
        return jsonify([])

@app.route('/api/clear_results', methods=['POST'])
def clear_results():
    """Clear all collected traces."""
    try:
        # Clear stored traces and heatmaps
        stored_traces.clear()
        stored_heatmaps.clear()
        return jsonify({'success': True, 'message': 'All traces cleared successfully'})
    except Exception as e:
        print(f"Error clearing traces: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# Additional endpoints can be implemented here as needed.

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)