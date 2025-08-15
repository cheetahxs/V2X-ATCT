from flask import Flask, request, jsonify, render_template,abort   ,send_file
import subprocess  
import os
# import datetime
from datetime import datetime  
from urllib.parse import quote, unquote


app = Flask(__name__, static_folder='templates/assets')  




@app.route('/')
def index():

    return render_template('gen_data.html')



ROOT_DIR_vis = '/path/to/your/generate/data/directory' 
ROOT_DIR_vis = os.path.abspath(ROOT_DIR_vis)


@app.route('/resultvis')
def resultvis(subpath=''):
    os.chdir(ROOT_DIR_vis)

    current_path = request.args.get('path', ROOT_DIR_vis)
    current_path = os.path.abspath(current_path)

    if not current_path.startswith(ROOT_DIR_vis):
        return "Access denied.", 403
    if not os.path.exists(current_path):
        return "Path not found.", 404
    if not os.path.isdir(current_path):
        return "Not a directory.", 400

    try:
        items = os.listdir(current_path)
    except PermissionError:
        return "Permission denied.", 403

    items_data = []
    for item in sorted(items):
        item_path = os.path.join(current_path, item)
        rel_path = os.path.relpath(item_path, ROOT_DIR_vis)
        url_path = rel_path.replace(os.sep, '/')

        items_data.append({
            'name': item,
            'is_dir': os.path.isdir(item_path),
            'path': url_path
        })

    parent_path = None
    if current_path != ROOT_DIR_vis:
        parent = os.path.dirname(current_path)
        parent_rel = os.path.relpath(parent, ROOT_DIR_vis)
        parent_path = parent_rel.replace(os.sep, '/')

    current_rel_path = os.path.relpath(current_path, ROOT_DIR_vis)
    if current_rel_path == '.':
        current_rel_path = 'Root'

    return render_template(
        'result_and_vis.html',
        items=items_data,
        current_path=current_rel_path,
        parent_path=parent_path
    )




@app.route('/gendata')
def gendata():
    return render_template('gen_data.html')


@app.route('/rq1')
def rq1():
    return render_template('RQ1.html')

@app.route('/rq2')
def rq2():
    return render_template('RQ2.html')



ROOT_DIR_exp = '/path/to/your/experiment/result/directory'

ROOT_DIR_exp = os.path.abspath(ROOT_DIR_exp)

@app.route('/experimentres')
def experimentres():
    os.chdir(ROOT_DIR_exp)

    current_path = request.args.get('path', ROOT_DIR_exp)
    current_path = os.path.abspath(current_path)

    if not current_path.startswith(ROOT_DIR_exp):
        return "Access denied.", 403
    if not os.path.exists(current_path):
        return "Path not found.", 404
    if not os.path.isdir(current_path):
        return "Not a directory.", 400

    try:
        items = os.listdir(current_path)
    except PermissionError:
        return "Permission denied.", 403

    items_data = []
    for item in sorted(items):
        item_path = os.path.join(current_path, item)
        rel_path = os.path.relpath(item_path, ROOT_DIR_vis)
        url_path = rel_path.replace(os.sep, '/')

        items_data.append({
            'name': item,
            'is_dir': os.path.isdir(item_path),
            'path': url_path
        })

    parent_path = None
    if current_path != ROOT_DIR_exp:
        parent = os.path.dirname(current_path)
        parent_rel = os.path.relpath(parent, ROOT_DIR_exp)
        parent_path = parent_rel.replace(os.sep, '/')

    current_rel_path = os.path.relpath(current_path, ROOT_DIR_exp)
    if current_rel_path == '.':
        current_rel_path = 'Root'

    return render_template(
        'experiment_result.html',
        items=items_data,
        current_path=current_rel_path,
        parent_path=parent_path
    )

    # return render_template('experiment_result.html')


@app.route('/run_script', methods=['POST'])
def run_script():
    try:
  
        result = subprocess.run(
            ['python', 'demo.py'],  
            capture_output=True,  
            text=True,            
            check=True              
        )
        
        return jsonify({
            'status': 'success',
            'output': result.stdout
        })
    except subprocess.CalledProcessError as e:
        
        return jsonify({
            'status': 'error',
            'output': e.stderr
        })
    except Exception as e:
       
        return jsonify({
            'status': 'error',
            'output': str(e)
        })






@app.route('/submit_gendata', methods=['POST'])
def submit_config():
    try:
        
        scene_number = request.form.get('scene', '').strip()
        driving_behaviour = request.form.get('drivingbehaviour', '').strip()
        tracknum = request.form.get('tracknum', '').strip()
        vehicle_speed = request.form.get('speed', '').strip()
        # roadnum = request.form.get('roadnum', '').strip()
        carnum = request.form.get('carnum', '').strip()
        save_path = request.form.get('savepath', '').strip()

     
        print(f"scene_number:{scene_number}")
        print(f"driving_behaviour:{driving_behaviour}")
        print(f"vehicle_speed:{vehicle_speed}")
        print(f"save_path:{save_path}")
        print(f"tracknum:{tracknum}")
        # print(f"roadnum:{roadnum}")
        print(f"carnum:{carnum}")


        subprocess.run(
            ['python', './V2X-ATCT/target_tracking/Generate_scenes.py',
             '--scene_num',scene_number,
             '--save_path',save_path,
             '--driving_behaviour',driving_behaviour,
             '--tracknum',tracknum,
             '--vehicle_speed',vehicle_speed,
            #  '--road_num',roadnum,
             '--carnum',carnum
             ], 
            capture_output=True,    
            text=True,             
            check=True           
        )



        return jsonify({
            'success': True,
            'message': f"Scenario Generation Input Successful!"
        })

    except Exception as e:

        return jsonify({
            'success': False,
            'message': str(e)
        })





@app.route('/submit_genrq1', methods=['POST'])
def submit_rq1():
    try:
        
        system = request.form.get('system', '').strip()
        drivingbehaviour = request.form.get('drivingbehaviour', '').strip()
        tracknum = request.form.get('tracknum', '').strip()
        seednum = request.form.get('seednum', '').strip()
        selectseeds = request.form.get('selectseeds', '').strip()
        speed = request.form.get('speed', '').strip()
        carnum = request.form.get('carnum', '').strip()
        savepath = request.form.get('savepath', '').strip()

   
        print(f"system:{system}")
        print(f"drivingbehaviour:{drivingbehaviour}")
        print(f"seednum:{seednum}")
        print(f"selectseeds:{selectseeds}")
        print(f"tracknum:{tracknum}")
        print(f"speed:{speed}")
        print(f"carnum:{carnum}")
        print(f"savepath:{savepath}")

        subprocess.run(
            ['python', './V2X-ATCT/target_tracking/rq1/RQ1.py',
             '--system',system,
             '--gen_seed_num',seednum,
             '--select_seed_num',selectseeds ,
             '--save_path_dir',savepath,
             '--driving_behaviour',drivingbehaviour,
             '--insert_time',tracknum,
             '--speed',speed,
             '--carnum',carnum
             ],  
            capture_output=True,  
            text=True,            
            check=True            
        )




        return jsonify({
            'success': True,
            'message': f"Data Input for Experiment 1 Successful!"
        })

    except Exception as e:

        return jsonify({
            'success': False,
            'message': str(e)
        })




@app.route('/submit_genrq2', methods=['POST'])
def submit_rq2():
    try:
       
        system = request.form.get('system', '').strip()
        drivingbehaviour = request.form.get('drivingbehaviour', '').strip()
        tracknum = request.form.get('tracknum', '').strip()
        seednum = request.form.get('seednum', '').strip()
        selectseeds = request.form.get('selectseeds', '').strip()
        speed = request.form.get('speed', '').strip()
        carnum = request.form.get('carnum', '').strip()
        savepath = request.form.get('savepath', '').strip()
        # spawnsnum = request.form.get('spawnsnum', '').strip()

     
        print(f"system:{system}")
        print(f"drivingbehaviour:{drivingbehaviour}")
        print(f"seednum:{seednum}")
        print(f"selectseeds:{selectseeds}")
        print(f"tracknum:{tracknum}")
        print(f"speed:{speed}")
        print(f"carnum:{carnum}")
        print(f"savepath:{savepath}")
        # print(f"spawnsnum:{spawnsnum}")


        subprocess.run(
            ['python', './V2X-ATCT/target_tracking/rq2/RQ2.py',
             '--system',system,
             '--seed_num',seednum,
             '--select_seed_num',selectseeds ,
             '--save_path',savepath,
             '--driving_behaviour',drivingbehaviour,
             '--insert_time',tracknum,
             '--speed',speed,
             '--carnum',carnum,
            #  '--spawnsnum',spawnsnum
             ], 
            capture_output=True,  
            text=True,              
            check=True             
        )


      
        return jsonify({
            'success': True,
            'message': f"Data Input for Experiment 2 Successful!"
        })

    except Exception as e:
      
        return jsonify({
            'success': False,
            'message': str(e)
        })





@app.route('/view_file')
def view_file():
    file_path = request.args.get('path')
    if not file_path:
        return "No path provided.", 400

    
    full_path = os.path.abspath(os.path.join(ROOT_DIR_vis, file_path))
    if not full_path.startswith(ROOT_DIR_vis):
        return "Access denied.", 403
    if not os.path.exists(full_path):
        return "File not found.", 404
    if not os.path.isfile(full_path):
        return "Not a file.", 400

  
    _, ext = os.path.splitext(file_path.lower())

    try:
        with open(full_path, 'rb') as f:
            content = f.read()

    
        if ext == '.txt':
            text_content = content.decode('utf-8', errors='replace')
            return render_template('view_txt.html', content=text_content, filename=os.path.basename(file_path))
        elif ext == '.mp4':
            # return render_template('playvideos.html', path=file_path, filename=os.path.basename(file_path))
            return send_file(full_path, mimetype='video/mp4')
        elif ext in ['.jpg', '.jpeg', '.png', '.gif']:
            return send_file(full_path, mimetype=f'image/{ext[1:]}')
        else:
            return f"Unsupported file type: {ext}. <br><a href='/' class='btn btn-md rounded font-sm hover-up mr-5'>Return to Homepage</a>"
    except Exception as e:
        return f"Error reading file: {str(e)}", 500





if __name__ == '__main__':
    app.run(debug=True,port=5000)  # http://127.0.0.1:5000