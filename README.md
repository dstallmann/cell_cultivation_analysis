# Towards an Automatic Analysis of CHO-K1 Suspension Growth in Microfluidic Single-cell Cultivation
These is the sourcecode for the paper “Towards an Automatic Analysis of CHO-K1 Suspension Growth in Microfluidic Single-cell Cultivation”, for the Bioinformatics journal.

## Usage instructions
Use Python 3 (perferably 3.7.3) and the requirements.txt and your favorite module management tool to make sure all required modules are installed. CUDA is required to run the project. We recommend to use CUDA 11.0, but 10.1 or 10.2 should work too. Should you have any trouble to install torch and torchvision with CUDA enabled, you can append the following to the module installation command to get the packages directly from pytorch.org.

```
-f https://download.pytorch.org/whl/torch_stable.html
```

Make sure the data directory contains the folders 128p_bf etc. and has the same parental path as the vae folder. It should look like this:

 .<br/>
    ├── ...<br/>
    ├── data<br/>
	│	├── 128p_bf<br/>
	│	└── ...<br/>
    ├── ...<br/>
    ├── vae<br/>
    └── ...

Call cli.py with adjusted parameters, if desired.

```
python cli.py 
```

should work, but can be specified to e.g.

```
python cli.py --epochs 10000 --lr 2e-4
```

If the images can not be found, adjust dataloader.py

```
prefix = os.getcwd().replace("\\", "/")[:-4]  # gets the current path up to /vae and removes the /vae to get to the data directory
```

to point to the data folder containing the images.

### Notes
The bayes_opt package as well as radam.py have only been marginally edited for this project. Their structure, mandatory content, documentation etc. remain the same. Credits are given within the code.
