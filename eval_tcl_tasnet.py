import torch
import numpy as np
import argparse
import soundfile as sf
#from datasets.lr_musdb import musdb
import museval
#import norbert
from pathlib import Path
import scipy.signal
#import resampy
from asteroid.complex_nn import torch_complex_from_magphase
import os
import warnings
import sys
import pandas as pd
import torch.nn.functional as F
import torchaudio

import os
import glob
import tqdm
import subprocess

#separator = torch.hub.load('sigsep/open-unmix-pytorch', 'umxl')

from eval import inference_args

#this uses the folder structure in my split samples
def load_from_tcl_sample(root_dir):
    ls_contents = os.listdir(root_dir)
    all_files = []
    directories = [] #parallel array with allfiles just to keep track of directory names
    for item in ls_contents:
        for root, dirs, files in os.walk(os.path.join(root_dir, item)):
            for test_file in files:
                all_files.append(os.path.join(test_file))
                directories.append(item)
    return all_files, directories


#input: 1x2xsamples, in a torch tensor
#output:  (with zero padding if needed), and number of padded samples
def frame_cutter(audio_tensor, frame_len_s, sample_rate):
    frame_size = frame_len_s * sample_rate
    
    #padding:
    remainder = frame_size - (audio_tensor.shape[2] % frame_size)
    if remainder !=0:
        audio_tensor = F.pad(input=audio_tensor, pad=(0, remainder, 0, 0, 0, 0), value=0)
        
    split_tensor = torch.split(audio_tensor, frame_size, dim=2)
    #split_tensor = torch.cat(split_tensor, dim=0)
    
    return split_tensor, remainder

#input nx4x2xsamples, start padding
#output 1x2xsamples, without the previously applied padding
def frame_gluer(prediction, remainder):
    #maybe no need to remove padding since it is appleid at the end, I can just cut the audio like i did in the data loaders after merging
    #torch_seq = torch.split(prediction, 1, dim=0) #commented as now we pass ready tuples
    return torch.cat(prediction, dim=3)


def eval_main(
    root,
    samplerate=44100,
    niter=1,
    alpha=1.0,
    softmask=False,
    residual_model=False,
    model_path='.',
    outdir=None,
    start=0.0,
    duration=-1.0,
    no_cuda=False,
    eval_data_path=None, 
    instrument='drums',
    variant='no_concat',
):


    if os.path.exists(outdir):
        print("Results of previous run saved in your chosen outdir: {}, please choose another location".format(outdir), file=sys.stderr)
    else:
        outdir = os.path.abspath(outdir)

    Path(outdir).mkdir(exist_ok=True, parents=True)
    print("Evaluated results will be saved in:\n {}".format(outdir), file=sys.stderr)

    if not eval_data_path:
        print("No location given for test data, please set one in cfg/eval.yml", file=sys.stderr)
        exit()

    torch.cuda.empty_cache()
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    test_dataset, directories  = load_from_tcl_sample(eval_data_path)
   
    for track, directory in zip(test_dataset, directories):
        output_path = Path(os.path.join(outdir, directory))
        output_path.mkdir(exist_ok=True, parents=True)

        track_path = os.path.join(eval_data_path, directory, track)
        #p_out = subprocess.run(["python3", "-m", "demucs.separate", "-n", 
        #    "tasnet","-o", output_path, "-f", track[:-4], "--mp3", track_path], capture_output=True)

        subprocess.run(["python3", "-m", "demucs.separate", "-n", "demucs", "-o", 
            os.path.join(output_path, track[:-4]), "--mp3", track_path])

        #print(p_out.returncode)
        #print(p_out.stdout.decode())
        #print(p_out.stderr.decode())
        # for the version used to generate the metrics, read the files and put
        # them in these dicts.
        '''estimates = {}
        estimates['vocals'] = prediction[0][0] 
        estimates['drums'] = prediction[0][1]
        estimates['bass'] = prediction[0][2]
        estimates['other'] = prediction[0][3]

        #adapt the output of openunmix: it outputs 22050, our gt loaded from the musdb package is
        #in 44100. so, we resample
        if instrument == 'bass':
            estimates['degraded_instrument_track'] = estimates['bass']
            estimates['degraded_backing_track'] = estimates['vocals'] + estimates['other'] + estimates['drums']

        elif instrument == 'drums':
            estimates['degraded_instrument_track'] = estimates['drums']
            estimates['degraded_backing_track'] = estimates['vocals'] + estimates['other'] + estimates['bass']

        #for key, val in estimates.items():
        #    estimates[key] = resampy.resample(estimates[val], 22050, 44100, axis=0)

        output_path = Path(os.path.join(outdir, directory, track))
        output_path.mkdir(exist_ok=True, parents=True)

        for target, estimate in estimates.items():
            sf.write(str(output_path / Path(target).with_suffix(".wav")), estimate.T, samplerate)
       '''
        
if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="OSU Inference", add_help=False)

    parser.add_argument("--root", type=str, help="The path to the MUSDB18 dataset")
    
    parser.add_argument(
        "--outdir",
        type=str,
        default="./results_using_pre-trained",
        help="Results path where " "best_model.pth" " is stored",
    )

    parser.add_argument("--start", type=float, default=0.0, help="Audio chunk start in seconds")

    parser.add_argument(
        "--duration",
        type=float,
        default=-1.0,
        help="Audio chunk duration in seconds, negative values load full track",
    )

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA inference"
    )

    args, _ = parser.parse_known_args()
    args = inference_args(parser, args)
    # Somehow these are not getting called at all.

    import yaml
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    with open("cfg/eval_tcl_tasnet.yml") as f:
        eval_conf = yaml.safe_load(f)
    eval_parser = prepare_parser_from_dict(eval_conf, parser=parser)

    arg_dic, plain_args = parse_args_as_dict(eval_parser, return_plain_args=True)

    model = os.path.join(plain_args.model_path, plain_args.model_name)
    
    #model = os.path.join("test.pth")

    eval_main(
        root='', #musdb.__path__[0],
        samplerate=plain_args.samplerate,
        alpha=args.alpha,
        softmask=args.softmask,
        niter=args.niter,
        residual_model=args.residual_model,
        model_path=plain_args.model_path, 
        outdir=plain_args.output_path,
        start=args.start,
        duration=args.duration,
        no_cuda=args.no_cuda,
        eval_data_path = plain_args.test_data_path,
        instrument=plain_args.instrument,
        variant=plain_args.model
    )
