from pathlib import Path
import torch.utils.data
import torchaudio
import random
import torch
import tqdm
import soundfile as sf
import copy
from scipy import signal

#additions for the ir convolutions
import pandas as pd
import librosa
import numpy as np
import os

class Leakage_MUSDB18Dataset(torch.utils.data.Dataset):
    """MUSDB18 music separation dataset

    The dataset consists of 150 full lengths music tracks (~10h duration) of
    different genres along with their isolated stems:
        `drums`, `bass`, `vocals` and `others`.

    Out-of-the-box, asteroid does only support MUSDB18-HQ which comes as
    uncompressed WAV files. To use the MUSDB18, please convert it to WAV first:

    - MUSDB18 HQ: https://zenodo.org/record/3338373
    - MUSDB18     https://zenodo.org/record/1117372

    .. note::
        The datasets are hosted on Zenodo and require that users
        request access, since the tracks can only be used for academic purposes.
        We manually check this requests.

    This dataset asssumes music tracks in (sub)folders where each folder
    has a fixed number of sources (defaults to 4). For each track, a list
    of `sources` and a common `suffix` can be specified.
    A linear mix is performed on the fly by summing up the sources

    Due to the fact that all tracks comprise the exact same set
    of sources, random track mixing can be used can be used,
    where sources from different tracks are mixed together.

    Folder Structure:
        >>> #train/<songname>/<variantnum>/clean_backing_track.wav ---------|
        >>> #train/<songname>/<variantnum>/degraded_audio_mix.wav ---------|
        >>> #train/<songname>/<variantnum>/degraded_backing_track.wav ---------|
        >>> #train/<songname>/<variantnum>/degraded_instrument_track.wav ---------|
        >>> #train/<songname>/<variantnum>/<source>.wav ---------|

#Folder hierarchy should be changed to include the source we are currently considering. rn it's all for drums only.

    Args:
        root (str): Root path of dataset
        sources (:obj:`list` of :obj:`str`, optional): List of source names
            that composes the mixture.
            Defaults to MUSDB18 4 stem scenario: `vocals`, `drums`, `bass`, `other`.
        targets (list or None, optional): List of source names to be used as
            targets. If None, a dict with the 4 stems is returned.
             If e.g [`vocals`, `drums`], a tensor with stacked `vocals` and
             `drums` is returned instead of a dict. Defaults to None.
        suffix (str, optional): Filename suffix, defaults to `.wav`.
        split (str, optional): Dataset subfolder, defaults to `train`.
        subset (:obj:`list` of :obj:`str`, optional): Selects a specific of
            list of tracks to be loaded, defaults to `None` (loads all tracks).
        segment (float, optional): Duration of segments in seconds,
            defaults to ``None`` which loads the full-length audio tracks.
        samples_per_track (int, optional):
            Number of samples yielded from each track, can be used to increase
            dataset size, defaults to `1`.
        random_segments (boolean, optional): Enables random offset for track segments.
        random_track_mix boolean: enables mixing of random sources from
            different tracks to assemble mix.
        source_augmentations (:obj:`list` of :obj:`callable`): list of augmentation
            function names, defaults to no-op augmentations (input = output)
        sample_rate (int, optional): Samplerate of files in dataset.

    Attributes:
        root (str): Root path of dataset
        sources (:obj:`list` of :obj:`str`, optional): List of source names.
            Defaults to MUSDB18 4 stem scenario: `vocals`, `drums`, `bass`, `other`.
        suffix (str, optional): Filename suffix, defaults to `.wav`.
        split (str, optional): Dataset subfolder, defaults to `train`.
        subset (:obj:`list` of :obj:`str`, optional): Selects a specific of
            list of tracks to be loaded, defaults to `None` (loads all tracks).
        segment (float, optional): Duration of segments in seconds,
            defaults to ``None`` which loads the full-length audio tracks.
        samples_per_track (int, optional):
            Number of samples yielded from each track, can be used to increase
            dataset size, defaults to `1`.
        random_segments (boolean, optional): Enables random offset for track segments.
        random_track_mix boolean: enables mixing of random sources from
            different tracks to assemble mix.
        source_augmentations (:obj:`list` of :obj:`callable`): list of augmentation
            function names, defaults to no-op augmentations (input = output)
        sample_rate (int, optional): Samplerate of files in dataset.
        tracks (:obj:`list` of :obj:`Dict`): List of track metadata

    References
        "The 2018 Signal Separation Evaluation Campaign" Stoter et al. 2018.
    """

    dataset_name = "Leakage_MUSDB18"

    def __init__(
        self,
        root,
        inputs=["clean_backing_track", "degraded_audio_mix"],
        outputs=['degraded_backing_track', 'degraded_instrument_track'], #another option could be to return the clean instrument track (the source) instead of the degraded instrument track.
        suffix=".wav",
        split="train",
        target="bass",
        #variants_per_track=10, 
        subset=None,
        segment=None,
        samples_per_track=1,
        random_segments=False,
        random_track_mix=False,
        source_augmentations=lambda audio: audio,
        sample_rate=44100,
        backing_track_input = False

    ):
        self.root = root
        self.split = split
        self.target = target
        self.sample_rate = sample_rate
        self.segment = segment
        self.random_track_mix = random_track_mix
        self.random_segments = random_segments
        self.source_augmentations = source_augmentations
        self.inputs = inputs
        self.outputs = outputs
        self.suffix = suffix
        self.subset = subset
        self.samples_per_track = samples_per_track
        self.tracks = list(self.get_tracks())
        self.backing_track_input = backing_track_input
        #self.variants_per_track=variants_per_track
 
        if not self.tracks:
            raise RuntimeError("No tracks found.")

    def __getitem__(self, index):
        model_inputs = {}
        model_outputs = {}

        import pdb
        pdb.set_trace()

        # get track_id
        track_id = index // self.samples_per_track

        if self.segment:
            start = random.uniform(0, self.tracks[track_id]["duration"] - self.segment)
        else:
            start = 0

        # load sources
        for model_input in self.inputs:
            # optionally select a random track for each source
            #track_id = random.choice(range(len(self.tracks))) WHY DID I DO THIS.............
            if self.random_segments:
                start = random.uniform(0, self.tracks[track_id]["duration"] - self.segment)

            # loads the full track duration
            start_sample = int(start * self.sample_rate)
            # check if dur is none
            if self.segment:
                # stop in soundfile is calc in samples, not seconds
                stop_sample = start_sample + int(self.segment * self.sample_rate)
            else:
                # set to None for reading complete file
                stop_sample = None

            # load actual audio
            # model input must hold the exact name as the files in the directory
            audio, _ = sf.read(
                Path(self.tracks[track_id]["path"] / model_input).with_suffix(self.suffix),
                always_2d=True,
                start=start_sample,
                stop=stop_sample,
            )

            # convert to torch tensor
            audio = torch.tensor(audio.T, dtype=torch.float)
            # apply source-wise augmentations
            audio = self.source_augmentations(audio)
            model_inputs[model_input] = audio


        for model_output in self.outputs:
            audio, _ = sf.read(
                Path(self.tracks[track_id]["path"] / model_output).with_suffix(self.suffix),
                always_2d=True,
                start=start_sample,
                stop=stop_sample,
            )
            audio = torch.tensor(audio.T, dtype=torch.float)
            model_outputs[model_output] = audio

        #just for now, remove the clean backing track from the model inputs
        #model_inputs.pop('clean_backing_track')

        #cut all the audio files to the same length, since unf. I forgot to do that when making the dataset..
        all_audios = list(model_inputs.values()) + list(model_outputs.values())
 
        shortest = np.min([t.shape[1] for t in all_audios]) #since we are assuming stereo audio, so the audio length is on the second dim.

        for key, val in model_inputs.items():
            model_inputs[key] = torch.narrow(val, 1, 0, shortest) 

        for key, val in model_outputs.items():
            model_outputs[key] = torch.narrow(val, 1, 0, shortest)

        #Since the first attempt will be without the clean backing track, we will just remove the extra input..
        # clean backing track, then degraded audio mix
        concat_inputs = torch.concat(list(model_inputs.values()), axis=0)
        # degraded instrument track, degraded backing track
        stacked_outputs = torch.stack(list(model_outputs.values()), dim=0)
    
        #import pdb
        #pdb.set_trace()

        if self.backing_track_input:
            return concat_inputs, stacked_outputs
        else:
            # for now, just return the input as is without stacking 
            return model_inputs['degraded_audio_mix'], stacked_outputs

    def __len__(self):
        return len(self.tracks) * self.samples_per_track

    def get_tracks(self):
        """Loads input and output tracks"""
        p = Path(self.root, self.split, self.target) #how do I make this tqdm thing work with the variant structure i have

        all_dirs = []
        for d in p.iterdir():
            if d.is_dir():
                all_dirs.extend([Path(d, num) for num in os.listdir(d)])
 
        for track_path in tqdm.tqdm(all_dirs): 
            if track_path.is_dir():
                if self.subset and track_path.parent.stem not in self.subset:
                    continue #skip track

                #check the inputs and outputs to the model
                ios = ['degraded_audio_mix', 'clean_backing_track', 'degraded_backing_track', 'degraded_instrument_track']
                source_paths = [track_path / (s + self.suffix) for s in ios]
                if not all(sp.exists() for sp in source_paths):
                    print("Exclude track due to missing input or output file")

                #check metadata
                infos = list(map(sf.info, source_paths))
                if not all(i.samplerate == self.sample_rate for i in infos):
                    print("Exclude track due to a different sample rate", track_path)
                    continue
               
                duration = min(i.duration for i in infos)
                #if self.segment is not None:
                #    min_duration = 10.0 #shit hardcoding for now
                #    if min_duration > self.segment:
                #        yield({"path": track_path, "duration": min_duration})
                #else:
                #    yield({"path": track_path, "duration": duration})
                yield({"path": track_path, "duration": duration})
                 

    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task"] = "enhancement"
        infos["licenses"] = [musdb_license]
        return infos


musdb_license = dict()
