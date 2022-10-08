import torch
import torchaudio
import numpy as np
import argparse
import soundfile as sf
import musdb
import museval
import norbert
from pathlib import Path
import scipy.signal
import resampy
from asteroid.complex_nn import torch_complex_from_magphase
import os
import warnings
import sys
import librosa

from eval import load_model, separate, inference_args

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

#to test the concat model on tcl, we need the tcl backing tracks too!
def load_backing_track(filename):
    return 0

def eval_main(root,
    samplerate=44100,
    niter=1,
    alpha=1.0,
    softmask=False,
    residual_model=False,
    model_path='.',
    model_name="leakage_xumx",
    outdir='',
    start=0.0,
    duration=-1.0,
    no_cuda=False,
    instrument='drums',
    eval_data_path=None,
    variant='no_concat'
):

#    outdir = os.path.join(os.path.abspath(outdir), test_output_files)

    model_name = os.path.join(model_path, model_name)

    if not (os.path.exists(model_name)):
        print("model does not exist: {}. Please update path in cnf/eval.yml".format(model_name), file=sys.stderr)
        quit()

    if os.path.exists(outdir):
        print("Results of previous run saved in your chosen outdir: {}, please choose another location".format(outdir), file=sys.stderr)
    else:
        outdir = os.path.abspath(outdir)

    Path(outdir).mkdir(exist_ok=True, parents=True)
    print("Evaluated results will be saved in:\n {}".format(outdir), file=sys.stderr)

    if not eval_data_path:
        print("No location given for test data, please set one in cfg/eval.yml", file=sys.stderr)
        exit()


    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model, instruments = load_model(variant, model_name, device)


    test_dataset, directories  = load_from_tcl_sample(eval_data_path)
    results = museval.EvalStore()
    txtout = os.path.join(outdir, "results.txt")
 
    for track, directory in zip(test_dataset, directories):
        audio, rate = torchaudio.load(os.path.join(eval_data_path, directory, track))
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
    import yaml
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    parser = argparse.ArgumentParser(description="OSU Inference", add_help=False)

    with open("cfg/eval_tcl.yml") as f:
        eval_conf = yaml.safe_load(f)
    eval_parser = prepare_parser_from_dict(eval_conf, parser=parser)

    arg_dic, plain_args = parse_args_as_dict(eval_parser, return_plain_args=True)

    model = os.path.join(plain_args.model_path, plain_args.model_name)

    eval_main("", 
        model_name=plain_args.model_name,
        outdir=plain_args.output_path,
        model_path=plain_args.model_path,
        eval_data_path=plain_args.test_data_path,
        instrument=plain_args.instrument,
        variant=plain_args.model
    )

