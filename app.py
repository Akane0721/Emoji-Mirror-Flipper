import os
from flask import Flask, render_template, request, send_from_directory
from datetime import datetime
from mirror import mirror_flip_image, mirror_flip_gif

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = 'static'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            time_stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
            filename = f"{time_stamp}.{file.filename.split('.')[-1]}"
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.mkdir(app.config['UPLOAD_FOLDER'])
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            return render_template('mirror.html', filename=filename)
        
    return render_template('index.html')
     

@app.route('/mirror', methods=['POST'])
def mirror():
    filename = request.form['filename']
    l2r = request.form.get('l2r') == 'l2r'
    input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists('output'):
                os.mkdir('output')
    output_filename = 'flipped_' + request.form.get('l2r') + '_' + filename
    output_filepath = os.path.join('output', output_filename)
    if filename.endswith('.jpg') or filename.endswith('.png'):
        mirror_flip_image(input_filepath, output_filepath, l2r)
    elif filename.endswith('.gif'):
        mirror_flip_gif(input_filepath, output_filepath, l2r)
    else:
        return render_template('upload_failed.html')
    return render_template('mirror.html', filename=filename, result_filename=output_filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/output/<filename>')
def generated_file(filename):
    return send_from_directory('output', filename)

if __name__ == "__main__":
    print("Visit http://localhost:5000/ to upload your image.")
    app.run(debug=True)