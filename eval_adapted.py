#Todo: have a different eval.py for each model variant.
# not sure if train.py will have the same fate
import torch
import numpy as np
import argparse
import soundfile as sf
from datasets.lr_musdb import musdb
import museval
import norbert
from pathlib import Path
import scipy.signal
import resampy
from asteroid.complex_nn import torch_complex_from_magphase
import os
import warnings
import sys

from eval import load_model, separate, inference_args

#This expects the test files to be stored in the same file structure of the leakage dataset (variants from 1-10, etc). For running test files kept in other structures, please build on eval_one instead. 

#where the model itself is stored
model_output_path = './exp_outputs/x-umx_outputs_exp1'

#test file output
test_output_files = 'EvaluateResults_leakage_musdb18_exp1_drums_testdata'

#test file input:
eval_data_path = '/media/alia/2aa686b1-fdb5-4cc9-b12d-c1fcdc0782cc/datasets/leakage_removal/'

def eval_main(
    root,
    samplerate=44100,
    niter=1,
    alpha=1.0,
    softmask=False,
    residual_model=False,
    model_name="leakage_xumx",
    outdir=None,
    start=0.0,
    duration=-1.0,
    no_cuda=False,
    eval_data_path=None, 
    instrument='drums'
):

    model_name = os.path.abspath(model_name)
    if not (os.path.exists(model_name)):
        outdir = os.path.abspath(model_output_path)
        #model_name = "leakage_xumx_checkpoint" #i think this is not needed
    else:
        outdir = os.path.join(
            os.path.abspath(model_output_path),
            test_output_files,
        )
    Path(outdir).mkdir(exist_ok=True, parents=True)
    print("Evaluated results will be saved in:\n {}".format(outdir), file=sys.stderr)

    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model, instruments = load_model(model_name, device)
    
    #import pdb
    #pdb.set_trace()
    test_dataset = musdb.DB(root=root, subsets="test", is_wav=True, instrument=instrument, data_path=eval_data_path)
    results = museval.EvalStore()
    Path(outdir).mkdir(exist_ok=True, parents=True)
    txtout = os.path.join(outdir, "results.txt")
    fp = open(txtout, "w")
    for track in test_dataset:
        input_file =  os.path.join(os.path.dirname(track.path), "degraded_audio_mix.wav")

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

        variant_number = os.path.basename(os.path.split(track.path)[0])

        output_path = Path(os.path.join(outdir, instrument, track.name, variant_number))
        output_path.mkdir(exist_ok=True, parents=True)

        print("Processing... {}".format(track.name), file=sys.stderr)
        print(track.name, file=fp)
        for target, estimate in estimates.items():
            sf.write(str(output_path / Path(target).with_suffix(".wav")), estimate, samplerate)
            
        track_scores = museval.eval_mus_track(track, estimates)
        track_scores.df.to_csv(os.path.join(output_path, 'result.csv'))
        results.add_track(track_scores.df)
        print(track_scores, file=sys.stderr)
        print(track_scores, file=fp)
    print(results, file=sys.stderr)
    print(results, file=fp)
    results.save(os.path.join(outdir, "results.pandas"))
    results.frames_agg = "mean"
    print(results, file=sys.stderr)
    print(results, file=fp)
    fp.close()


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
    
    #model = os.path.join("test.pth")
    model = os.path.join(model_output_path, 'best_model.pth')
    eval_main(
        root=musdb.__path__[0],
        samplerate=args.samplerate,
        alpha=args.alpha,
        softmask=args.softmask,
        niter=args.niter,
        residual_model=args.residual_model,
        model_name=model,
        outdir=args.outdir,
        start=args.start,
        duration=args.duration,
        no_cuda=args.no_cuda,
        eval_data_path = eval_data_path,
        instrument="drums"
    )
