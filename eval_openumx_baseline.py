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
import pandas as pd
import torch.nn.functional as F


#separator = torch.hub.load('sigsep/open-unmix-pytorch', 'umxl')

from eval import inference_args

#input: 1x2xsamples, in a torch tensor
#output:  (with zero padding if needed), and number of padded samples
def frame_cutter(audio_tensor, frame_len_s, sample_rate):
    frame_size = frame_len_s * sample_rate
    
    #padding:
    remainder = frame_size - (audio_tensor.shape[2] % frame_size)
    if remainder !=0:
        audio_tensor = F.pad(input=audio_tensor, pad=(0, remainder, 0, 0, 0, 0), value=0)
        
    split_tensor = torch.split(audio_tensor, frame_size, dim=2)
    split_tensor = torch.cat(split_tensor, dim=0)
    
    return split_tensor, remainder

#input nx4x2xsamples, start padding
#output 1x2xsamples, without the previously applied padding
def frame_gluer(prediction, remainder):
    #maybe no need to remove padding since it is appleid at the end, I can just cut the audio like i did in the data loaders after merging
    torch_seq = torch.split(prediction, 1, dim=0)
    merged = torch.cat(torch_seq, dim=3)

    return merged


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

    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #separator = torch.hub.load('sigsep/open-unmix-pytorch', 'umxhq', device=device)
    separator = torch.hub.load('sigsep/open-unmix-pytorch', 'umxhq')    

    test_dataset = musdb.DB(root=root, subsets="test", is_wav=True, instrument=instrument, data_path=eval_data_path)
    results = museval.EvalStore()
    results_df = pd.DataFrame(columns=['target', 'metric', 'mean_values', 'median_values', 'variant', 'track_name'])
    txtout = os.path.join(outdir, "results.txt")
    fp = open(txtout, "w")
    for track in test_dataset:
        input_file =  os.path.join(os.path.dirname(track.path), "degraded_audio_mix.wav")
        clean_backing_track = os.path.join(os.path.dirname(track.path), "clean_backing_track.wav")

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
     
        #audio_torch = torch.tensor(audio.T[None, ...]).float().to(device)
        audio_torch = torch.tensor(audio.T[None, ...]).float()

        import pdb
        pdb.set_trace()

        split_audio, padding = frame_cutter(audio_torch, 10, 44100)

        prediction = separator(split_audio)

        prediction = frame_gluer(prediction, 0)
        #squeeze the prediction to the length of the input audio to remove the initial padding

        prediction = prediction.detach()

        estimates = {}
        estimates['vocal'] = prediction[0][0] 
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

        variant_number = os.path.basename(os.path.split(track.path)[0])
        output_path = Path(os.path.join(outdir, instrument, track.name, variant_number))
        output_path.mkdir(exist_ok=True, parents=True)

        print("Processing... {}".format(track.name), file=sys.stderr)
        print(track.name, file=fp)
        for target, estimate in estimates.items():
            sf.write(str(output_path / Path(target).with_suffix(".wav")), estimate, samplerate)
    
        #just for debugging, overwrite the tracks to be a subset from 10 - 40 seconds
        
        
        track_scores = museval.eval_mus_track(track, estimates)
        track_scores.df.to_csv(os.path.join(output_path, 'frame_result.csv'))
        
        summary_target = ['degraded_backing_track', 'degraded_instrument_track']
        summary_metrics = ['SDR', 'SIR', 'SAR']
        summary_target_col = []
        summary_metric_col = []
        summary_metric_median = []
        summary_metric_mean = []
        track_variant = []
        track_name = []
        for t in summary_target:
            for m in summary_metrics:
                summary_target_col.append(t)
                summary_metric_col.append(m) 
                rel_cols = track_scores.df[(track_scores.df['metric'] == m) & (track_scores.df['target'] == t)]
                summary_metric_median.append(track_scores.frames_agg(rel_cols['score']))
                summary_metric_mean.append(np.nanmean(rel_cols['score']))
                track_variant.append(os.path.basename(os.path.dirname(track.path)))
                track_name.append(track_scores.track_name)
                
        summary_df = pd.DataFrame({ 'target': summary_target_col,
                                    'metric': summary_metric_col, 
                                    'mean_values': summary_metric_mean,
                                    'median_values': summary_metric_median,
                                    'variant': track_variant,
                                    'track_name': track_name
                        })
       
        summary_df.to_csv(os.path.join(output_path, 'results_summary.csv'))
        results.add_track(track_scores.df)
        results_df = results_df.append(summary_df)
        print(track_scores, file=sys.stderr)
        results_df.to_csv(os.path.join(outdir, instrument, 'all_result_summaries.csv'))

    #aggregate results of all runs
    print(results, file=sys.stderr)
    results_df.to_csv(os.path.join(outdir, instrument, 'all_result_summaries.csv'))
    results.frames_agg = "mean"
    print(results, file=sys.stderr)
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

    import yaml
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    with open("cfg/eval_baseline.yml") as f:
        eval_conf = yaml.safe_load(f)
    eval_parser = prepare_parser_from_dict(eval_conf, parser=parser)

    arg_dic, plain_args = parse_args_as_dict(eval_parser, return_plain_args=True)

    model = os.path.join(plain_args.model_path, plain_args.model_name)
    
    #model = os.path.join("test.pth")

    eval_main(
        root=musdb.__path__[0],
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
