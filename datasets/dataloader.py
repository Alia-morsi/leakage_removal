from .leakage_musdb18_dataset import Leakage_MUSDB18Dataset
import torch
from pathlib import Path
import random
import numpy as np

toy_train_tracks = [
    "Johnny Lokke - Promises & Lies",
    "Patrick Talbot - A Reason To Leave",
    "Triviul - Angelsaint",
    "Alexander Ross - Goodbye Bolero",
    "Fergessen - Nos Palpitants",
    "Leaf - Summerghost"
    ]


'''toy_train_tracks = [
    'A Classic Education - NightOwl',
    "Actions - Devil's Words",
    #'Actions - One Minute Smile',
    'Actions - South Of The Water',
    'Aimee Norwich - Child',
    #'Alexander Ross - Goodbye Bolero',
    'Alexander Ross - Velvet Curtain',
    'Angela Thomas Wade - Milk Cow Blues',
    'ANiMAL - Clinic A',
    #'ANiMAL - Easy Tiger',
    #'ANiMAL - Rockshow',
    'Atlantis Bound - It Was My Fault For Waiting',
    'Auctioneer - Our Future Faces',
    'AvaLuna - Waterduct',
    'BigTroubles - Phantom',
    #'Bill Chudziak - Children Of No-one',
    'Black Bloc - If You Want Success',
    'Celestial Shore - Die For Us',
    'Chris Durban - Celebrate',
    #'Clara Berry And Wooldog - Air Traffic',
    #'Clara Berry And Wooldog - Stella',
    'Clara Berry And Wooldog - Waltz For My Victims',
    'Cnoc An Tursa - Bannockburn',
    'Creepoid - OldTree',
    #'Dark Ride - Burning Bridges',
    'Dreamers Of The Ghetto - Heavy Love',
    'Drumtracks - Ghost Bitch',
    'Faces On Film - Waiting For Ga',
    #'Fergessen - Back From The Start',
    #'Fergessen - Nos Palpitants',
    'Fergessen - The Wind',
    'Flags - 54',
    'Giselle - Moss',
    'Grants - PunchDrunk',
    'Helado Negro - Mitad Del Mundo',
    #'Hezekiah Jones - Borrowed Heart',
    'Hollow Ground - Left Blind',
    'Hop Along - Sister Cities',
    'Invisible Familiars - Disturbing Wildlife',
    #'James May - All Souls Moon',
    'James May - Dont Let Go',
    #'James May - If You Say',
    'James May - On The Line',
    'Jay Menon - Through My Eyes',
    'Johnny Lokke - Promises & Lies',
    #'Johnny Lokke - Whisper To A Scream',
    'Jokers, Jacks & Kings - Sea Of Leaves',
    'Leaf - Come Around',
    #'Leaf - Summerghost',
    #'Leaf - Wicked',
    'Lushlife - Toynbee Suite',
    'Matthew Entwistle - Dont You Ever',
   'Meaxic - Take A Step',
    'Meaxic - You Listen',
    'Music Delta - 80s Rock',
    #'Music Delta - Beatles',
    'Music Delta - Britpop',
   # 'Music Delta - Country1',
    'Music Delta - Country2',
    'Music Delta - Disco',
    'Music Delta - Gospel',
    'Music Delta - Grunge',
    'Music Delta - Hendrix',
    'Music Delta - Punk',
   # 'Music Delta - Reggae',
    'Music Delta - Rock',
    'Music Delta - Rockabilly',
    'Night Panther - Fire',
    'North To Alaska - All The Same',
    'Patrick Talbot - A Reason To Leave',
   # 'Patrick Talbot - Set Me Free',
    "Phre The Eon - Everybody's Falling Apart",
    'Port St Willow - Stay Even',
    'Remember December - C U Next Time',
   # 'Skelpolu - Human Mistakes',
    'Skelpolu - Together Alone',
    'Snowmine - Curfews',
    "Spike Mullings - Mike's Sulking",
    'Steven Clark - Bounty',
    'Strand Of Oaks - Spacestation',
    'St Vitus - Word Gets Around',
    'Sweet Lights - You Let Me Down',
    'Swinging Steaks - Lost My Way',
    'The Districts - Vermont',
    'The Long Wait - Back Home To Blue',
    'The Scarlet Brand - Les Fleurs Du Mal',
   # 'The So So Glos - Emergency',
    "The Wrong'Uns - Rothko",
    'Tim Taler - Stalker',
   # 'Titanium - Haunted Age',
    'Traffic Experiment - Once More (With Feeling)',
    'Traffic Experiment - Sirens',
   # 'Triviul - Angelsaint',
   # 'Triviul - Dorothy',
    'Voelund - Comfort Lives In Belief',
    'Wall Of Death - Femme',
   # 'Young Griffo - Blood To Bone',
   # 'Young Griffo - Facade',
    'Young Griffo - Pennies',
        ]'''

toy_valid_tracks = toy_train_tracks[0:2]

#The validation tracks will be 10 random tracks from the toy train set. 
#toy_valid_tracks = np.array(toy_train_tracks)[random.sample(range(0, len(toy_train_tracks)), 10)] 
#toy_valid_tracks = list(toy_valid_tracks)

#validation_tracks = [
#    "Actions - One Minute Smile",
#    "Clara Berry And Wooldog - Waltz For My Victims",
#    "Johnny Lokke - Promises & Lies",
#    "Patrick Talbot - A Reason To Leave",
#    "Triviul - Angelsaint",
#    "Alexander Ross - Goodbye Bolero",
#    "Fergessen - Nos Palpitants",
#    "Leaf - Summerghost",
#    "Skelpolu - Human Mistakes",
#    "Young Griffo - Pennies",
#    "ANiMAL - Rockshow",
#    "James May - On The Line",
#    "Meaxic - Take A Step",
#    "Traffic Experiment - Sirens",
#]

'''toy_validation_tracks = [
    'Actions - South Of The Water',
    'Chris Durban - Celebrate',
    'Dreamers Of The Ghetto - Heavy Love',
    'Invisible Familiars - Disturbing Wildlife',
    'Lushlife - Toynbee Suite',
    'Music Delta - Grunge',
    'North To Alaska - All The Same',
    "Spike Mullings - Mike's Sulking",
    'St Vitus - Word Gets Around',
    'The Scarlet Brand - Les Fleurs Du Mal',
    'Strand Of Oaks - Spacestation',
    'Leaf - Come Around'
]'''


def load_datasets(parser, args):
    """Loads the specified dataset from commandline arguments

    Returns:
        train_dataset, validation_dataset
    """

    args = parser.parse_args()

    dataset_kwargs = {
        "root": Path(args.root),
        "backing_track_input": args.backing_track_input,
        "target":args.target
    }

    #if args.leakage_removal:
    #    dataset_kwargs['ir_paths'] = {'irs_metadata': Path(args.irs_metadata), args.irs_1: Path(args.irs_1_dir), args.irs_2: Path(args.irs_2_dir)}
    #    dataset_kwargs['leakage_removal'] = True

    source_augmentations = Compose(
        [globals()["_augment_" + aug] for aug in args.source_augmentations]
    )

    if args.toy_run == True:
        train_dataset = Leakage_MUSDB18Dataset(
            #root=args.root,
            split="train",
            subset=toy_train_tracks, 
            inputs=args.inputs,
            outputs=args.outputs,
            source_augmentations=source_augmentations,
            random_track_mix=True,
            segment=args.seq_dur,
            random_segments=args.random_segments,
            sample_rate=args.sample_rate,
            samples_per_track=args.samples_per_track,
            **dataset_kwargs,
        )
        train_dataset = filtering_out_valid_toy(train_dataset)

        valid_dataset = Leakage_MUSDB18Dataset(
            #root=args.root,
            split="train",
            subset=toy_valid_tracks,
            inputs=args.inputs,
            outputs=args.outputs,
            segment=None,
            **dataset_kwargs,
        )
 

    else:
        train_dataset = Leakage_MUSDB18Dataset(
            #root=args.root,
            split="train",
            inputs=args.inputs,
            outputs=args.outputs,
            source_augmentations=source_augmentations,
            random_track_mix=True,
            segment=args.seq_dur,
            random_segments=args.random_segments,
            sample_rate=args.sample_rate,
            samples_per_track=args.samples_per_track,
            **dataset_kwargs,
        )
        train_dataset = filtering_out_valid(train_dataset)

        valid_dataset = Leakage_MUSDB18Dataset(
            #root=args.root,
            split="train",
            subset=validation_tracks,
            inputs=args.inputs,
            outputs=args.outputs,
            segment=None,
            **dataset_kwargs,
        )

    return train_dataset, valid_dataset


def filtering_out_valid(input_dataset):
    """Filtering out validation tracks from input dataset.

    Return:
        input_dataset (w/o validation tracks)
    """
    input_dataset.tracks = [
        tmp
        for tmp in input_dataset.tracks
        if not (str(tmp["path"]).split("/")[-1] in validation_tracks)
    ]

    return input_dataset

def filtering_out_valid_toy(input_dataset):
    input_dataset.tracks = [
        tmp
        for tmp in input_dataset.tracks
        if not (str(tmp["path"]).split("/")[-1] in toy_valid_tracks)
    ]

    return input_dataset

class Compose(object):
    """Composes several augmentation transforms.
    Args:
        augmentations: list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for transform in self.transforms:
            audio = transform(audio)
        return audio

#does augment Snr also make sense to be added here

def _augment_gain(audio, low=0.25, high=1.25):
    """Applies a random gain to each source between `low` and `high`"""
    gain = low + torch.rand(1) * (high - low)
    return audio * gain


def _augment_channelswap(audio):
    """Randomly swap channels of stereo sources"""
    if audio.shape[0] == 2 and torch.FloatTensor(1).uniform_() < 0.5:
        return torch.flip(audio, [0])

    return audio

#here, we must add the augmentations we want, which are the noise additions. 
#def _augment_ir_conv(audio, ir_info='', purpose='train'):
#    """ Apply noise convolutions to the audio depending on the purpose. ir_info leads to a csv with the irs to be used for train vs test vs validation, and we filter for a relevant ir depending on the purpose"""

#    return audio
