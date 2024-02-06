from flask import Flask, jsonify, request, send_file
import subprocess
import docker
import json
import os
import pickle as pkl

app = Flask(__name__)


def write_filenames_to_file(file_list, file_path):
    
    file_list = [i.split('.')[0] for i in file_list]
    with open(file_path, 'w') as file:
        for filename in file_list:
            file.write(filename + '\n')



@app.route('/run-docker-script', methods=['POST'])
def run_docker_script():

    MAIN_DIR = '/home/ahmed/workspace/00_active_3Ddet/'
    args = request.json
    mode = args['mode']

    if (mode == 'inference'):

        op = args['op']
        filenames = ' '.join(args['frame_ids'])

        client = docker.from_env()
        container = client.containers.get('active3D')
        command = f"/opt/conda/envs/ahmed/bin/python ./tools/proannoV2/inference.py "\
                    f"--op {op} --filenames {filenames}"
        
        exit_code, output = container.exec_run(command)
        output_decoded = output.decode().strip()

        detections_dir = '/home/ahmed/workspace/00_active_3Ddet/data/tumtraf/proannoV2/OpenLABEL/currently_annotating/detections'
        json_files = [f for f in os.listdir(detections_dir) if f.endswith('.json')]
        json_data = []
        for json_file in json_files:
            if json_file.split('.')[0] in args['frame_ids']:
                file_path = os.path.join(detections_dir, json_file)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    json_data.append(data)


        return jsonify({
        "exit_code": exit_code, # if 0 then success
        "preds": json_data
        })
    

    elif (mode == 'evaluation'):

        op = args['op']
        filenames = ' '.join(args['frame_ids'])
        
        client = docker.from_env()
        container = client.containers.get('active3D')
        command = f"/opt/conda/envs/ahmed/bin/python /tools/proannoV2/evaluation.py"\
                    f"--op {op} --filenames {filenames}"
        
        exit_code, output = container.exec_run(command)
        output_decoded = output.decode().strip()
        array_str = output_decoded.split('\n')[-1]
        output_array = json.loads(array_str)

        return jsonify({
        "exit_code": exit_code, # if 0 then success
        "evals": output_array
        })


    elif (mode == 'AL'):
        query_map =  {
            'CRB': 'crb',
            'tCRB': 'tcrb'
        }
        
        op = args['op']

        query = query_map[args['query']]
        n_select = args['N_select']
        # cfg_file = args['cfg_file']

        client = docker.from_env()
        container = client.containers.get('active3D')
        command = f"/opt/conda/envs/ahmed/bin/python ./tools/proannoV2/active.py " \
                    f"--op {op} --query {query} --n_select {n_select}"

        exit_code, output = container.exec_run(command)

        return jsonify({
        "exit_code": exit_code
        })


    else:
        return jsonify({
            'error': 'NotImplemented'
        })



@app.route('/run-docker-script-old', methods=['POST'])
def run_docker_script_old():

    AL_DIR = '/home/ahmed/workspace/00_active_3Ddet/'
    data = request.json
    # print("we are here: data loaded")

    filename = data.get('fileName', '')
    file_list = data.get('fileList', [])
    ckpt = data.get('modelCkpt', 'Oracle_80.pth')
    cfg_file = data.get('cfgFile', '/ahmed/tools/cfgs/tumtraf_models/pv_rcnn.yaml')
    op = data.get('op', 'inference')
    train = data.get('activeTrain', False)
    train_scheme = data.get('trainScheme', 'tunedAll')

    if (len(filename.split('.')) > 1):
        filename = filename.split('.')[0]

    file_list_name = 'None'
    if len(file_list) > 0:
        file_list_name = 'latest_filenames.txt'
        write_filenames_to_file(file_list=file_list, 
                                file_path=os.path.join(AL_DIR, file_list_name))
    client = docker.from_env()
    container = client.containers.get('active3D')

    # print("we are here: data extracted ")

    command = f"/opt/conda/envs/ahmed/bin/python /tools/proannoV2/main.py --filename {filename}"\
                f"--op {op} --active_train {train} --train_scheme {train_scheme}" \
                f"--file_list_name {file_list_name} --ckpt {ckpt} --cfg_file {cfg_file}"

    # print("we are here: command-line generated")

    # exec_run captures only the output in stdout --> the output that is printed
    exit_code, output = container.exec_run(command)
    # print("we are here: recieving output from docker")
    
    output_decoded = output.decode().strip()
    # print("decoded output: ", output_decoded)

    array_str = output_decoded.split('\n')[-1]  # Get the last line, assuming it is JSON
    output_array = json.loads(array_str)
    
    return jsonify({
        "exit_code": exit_code, # if 0 then success
        "preds": output_array
        })


if __name__ == "__main__":
    #app.run(host='0.0.0.0', port=5001, debug=True)
    app.run(host='127.0.0.1', port=6000, debug=True)
    # '0.0.0.0' tells the operating system to listen to all public IPs.
    # it makes our Flask server accessible from other machines on the same network.