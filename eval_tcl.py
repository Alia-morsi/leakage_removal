import torch
import torchaudio
import numpy as np
import argparse
import soundfile as sf
import musdb
#import museval
import norbert
from pathlib import Path
import scipy.signal
import resampy
from asteroid.complex_nn import torch_complex_from_magphase
import os
import warnings
import sys
import librosa

from eval import load_model, separate

#where the model itself is stored
model_output_path = './exp_outputs/x-umx_outputs_exp2_bass'

#test file output
test_output_files = 'results_using_pre-trained/EvaluateResults_leakage_tcl_sample_exp2_bass_tcl'

#test file input:
eval_data_path = '/home/alia/Documents/leakage_removal/TCL_Sample/performance_sample_explorer/split_samples/Bass'

#insert path of model to load
model_path = 'best_model.pth'

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

def eval_main(root,
    samplerate=44100,
    niter=1,
    alpha=1.0,
    softmask=False,
    residual_model=False,
    model_name="leakage_xumx",
    outdir='',
    start=0.0,
    duration=-1.0,
    no_cuda=False,
):

    outdir = os.path.join(os.path.abspath(outdir), test_output_files)
    
    Path(outdir).mkdir(exist_ok=True, parents=True)
    print("Evaluated results will be saved in:\n {}".format(outdir), file=sys.stderr)

    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model, instruments = load_model(model_name, device)

    test_path = '/home/alia/Documents/leakage_removal/TCL_Sample/performance_sample_explorer/split_samples/Bass'

    test_dataset, directories  = load_from_tcl_sample(test_path)
 
    for track, directory in zip(test_dataset, directories):
        audio, rate = torchaudio.load(os.path.join(test_path, directory, track))
        # reshape to the expected format (the sf read output)

        if rate != samplerate:
            # resample to model samplerate if needed
            # we changed this from resampy to torchaudio
            audio = torchaudio.functional.resample(audio, rate, samplerate)

        # reshaping to match the output produced by soundfile.read
        audio = audio.T
        # handling an input audio path. we already assume it's 44100

        if audio.shape[1] > 2:
            warnings.warn("Channel count > 2! " "Only the first two channels will be processed!")
            audio = audio[:, :2]

        if audio.shape[1] == 1:
            # if we have mono, let's duplicate it
            # as the input of OpenUnmix is always stereo
            audio = np.repeat(audio, 2, axis=1)


        estimates = separate(
            audio,
            model,
            instruments,
            niter=niter,
            alpha=alpha,
            softmask=softmask,
            residual_model=residual_model,
            device=device,
        )

        output_path = Path(os.path.join(outdir, directory, track))
        output_path.mkdir(exist_ok=True, parents=True)
        
        for target, estimate in estimates.items():
            sf.write(str(output_path / Path(target).with_suffix(".wav")), estimate, samplerate)
        

if __name__ == "__main__":
    
    eval_main("", 
        model_name=os.path.join(model_output_path, model_path),
        outdir='',
    )

