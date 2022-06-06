# Driving Environment Detection (Locality Calssification)

# Setting up Docker Environment and Dependencies
<ul>
    <li>Step 1: Clone the repository to local machine 
        <pre>git clone https://gitlab.vtti.vt.edu/ctbs/fhwa-cv/driving-environment-detection.git</pre>
    </li>
    <li>Step 2: cd to downloaded repository 
        <pre>cd [repo-name]</pre>
    </li>
    <li>Step 3: Build the docker image using Dockerfile.ML
    <pre>docker build -f Dockerfile.ML -t driving_env .</pre>
    </li>
    <li>Step 4: Run container from image and mount data volumes
        <pre>docker run -it --rm -p 9999:8888 -v $(pwd):/opt/app -v [path to data]:/opt/app/data --shm-size=20G driving_env</pre>
    example: <pre>docker run -it --rm -p 9999:8888 --user=12764:10001 -v $(pwd):/opt/app -v /vtti:/vtti --gpus all --shm-size=20G driving_env</pre>
    </li>You may get an error <pre>failed: port is already allocated</pre>
    If so, expose a different port number on the server, e.g. '9898:8888'
    <li>If you wish to run the jupyter notebook, type 'jupyter' on the container's terminal</li>
    <li>On your local machine perform port forwarding using
        <pre>ssh -N -f -L 9999:localhost:9999 host@server.xyz </pre>
    </li>
</ul>

# Dataset Information

Organize the data as follows in the repository
<pre>
./
 |__ data
        |__ Interstate
        |__ Urban
	|__ Residential
        
</pre>

# Model 1: Intersection Detection

To run the model

<pre>
cd /opt/app
python main.py \
--config [optional:path to config file] \
--mode ['train', 'test', 'test_single'] \
--comment [optional:any comment while training] \
--weight [optional:custom path to weight] \
--device [optional:set device number if you have multiple GPUs]
</pre>

## Training & Testing

