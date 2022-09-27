import torch
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

from eval import load_model, separate

#insert path of model to load
model_path = 'test.pth'

def eval_main(root,
    samplerate=44100,
    niter=1,
    alpha=1.0,
    softmask=False,
    residual_model=False,
    model_name="leakage_xumx",
    outdir='dummy_test_outputs',
    start=0.0,
    duration=-1.0,
    no_cuda=False,
):

    model_name = os.path.abspath(model_name)
    if not (os.path.exists(model_name)):
        outdir = os.path.abspath("./checkpoint_results")
        model_name = "leakage_xumx_checkpoint"
    else:
        outdir = os.path.join(
            os.path.abspath(outdir),
            "EvaluateResults_musdb18_testdata",
        )
    
    Path(outdir).mkdir(exist_ok=True, parents=True)
    print("Evaluated results will be saved in:\n {}".format(outdir), file=sys.stderr)

    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model, instruments = load_model(model_name, device)

    # test_dataset = musdb.DB(root=root, subsets="test", is_wav=True)
    # TODO: write a class to travers the entire dataset (train/test, and file number)
    # results = museval.EvalStore()

    #Path(outdir).mkdir(exist_ok=True, parents=True)
    #txtout = os.path.join(outdir, "results.txt")
    #fp = open(txtout, "w")

    test_path = '/media/data/alia/Documents/datasets/leakage_removal/test/drums/Al James - Schoolboy Facination/4'

    test_dataset = [test_path]
 
    for track in test_dataset:
        #input_file = os.path.join(root, "test", track.name, "mixture.wav")
        input_file = os.path.join(test_path, 'degraded_audio_mix.wav')

        # handling an input audio path
        info = sf.info(input_file)
        start = int(start * info.samplerate)
        # check if dur is none
        if duration > 0:
            # stop in soundfile is calc in samples, not seconds
            stop = start + int(duration * info.samplerate)
        else:
            # set to None for reading complete file
            stop = None

        audio, rate = sf.read(input_file, always_2d=True, start=start, stop=stop)

        if audio.shape[1] > 2:
            warnings.warn("Channel count > 2! " "Only the first two channels will be processed!")
            audio = audio[:, :2]

        if rate != samplerate:
            # resample to model samplerate if needed
            audio = resampy.resample(audio, rate, samplerate, axis=0)

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

        output_path = Path(os.path.join(outdir, 'first_track_test'))
        output_path.mkdir(exist_ok=True, parents=True)
        
        #print("Processing... {}".format(track.name), file=sys.stderr)
        #print(track.name, file=fp)
        for target, estimate in estimates.items():
            sf.write(str(output_path / Path(target).with_suffix(".wav")), estimate, samplerate)
        
        #track_scores = museval.eval_mus_track(track, estimates)
        #results.add_track(track_scores.df)
        #print(track_scores, file=sys.stderr)
        #print(track_scores, file=fp)
        #print(results, file=sys.stderr)
        #print(results, file=fp)
        #results.save(os.path.join(outdir, "results.pandas"))
        #results.frames_agg = "mean"
        #print(results, file=sys.stderr)
        #print(results, file=fp)
        #fp.close()        

if __name__ == "__main__":
    # Training settings
    #parser = argparse.ArgumentParser(description="OSU Inference", add_help=False)

    #parser.add_argument("--root", type=str, help="The path to the MUSDB18 dataset")

    #parser.add_argument(
    #    "--outdir",
    #    type=str,
    #    default="./results_using_pre-trained",
    #    help="Results path where " "best_model.pth" " is stored",
    #)

    #parser.add_argument("--start", type=float, default=0.0, help="Audio chunk start in seconds")

    #parser.add_argument(
    #    "--duration",
    #    type=float,
    #    default=-1.0,
    #    help="Audio chunk duration in seconds, negative values load full track",
    #)

    #parser.add_argument(
    #    "--no-cuda", action="store_true", default=False, help="disables CUDA inference"
    #)

    #args, _ = parser.parse_known_args()
    #args = inference_args(parser, args)

    #model = os.path.join(args.outdir, "test.pth")
    model = "test.pth"

    eval_main("", 
        model_name=model,
        outdir='dummy_test',
    )

