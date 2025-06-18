from flask import Flask, send_from_directory, request, jsonify
# additional imports
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib
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
        # plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
        # plt.close(fig)

        # Store trace and heatmap
        stored_traces.append(trace)
        stored_heatmaps.append(filename)

        # Return the heatmap image URL and stats
        return jsonify({
            'heatmap_url': f'/{filename}',
            'min': min_val,
            'max': max_val,
            'range': range_val,
            'samples': samples
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