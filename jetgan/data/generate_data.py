import argparse
import logging
import os
from pathlib import Path
import fastjet
import pythia8
import numpy as np

from generators import PythiaGenerator
import image


def main(args):
    """ Generates raw jet images using pythia and fastjet
    """
    logger = logging.getLogger(__name__)
    logger.info('initializing pythia')

    # initialize generator
    g = PythiaGenerator(pt_hat_min=args.pt_hat_min, pt_hat_max=args.pt_hat_max,
                        jet_radius=args.jet_radius, gauss_smear=args.gaussian_smear,
                        tpc_sim=args.tpc_sim, star_sim=args.star_sim,
                        jet_pt_min=args.jet_pt_min, gauss_mean=args.gaussian_mean,
                        gauss_sigma=args.gaussian_width)
    g.init()

    # initialize image parameter dictionary
    image_params = {}
    image_params['x_bins'] = args.pixels_eta
    image_params['x_min'] = - args.image_width_eta / 2.0
    image_params['x_max'] = args.image_width_eta / 2.0
    image_params['y_bins'] = args.pixels_phi
    image_params['y_min'] = - args.image_width_phi / 2.0
    image_params['y_max'] = args.image_width_phi / 2.0
    image_params['z_bins'] = 2 if args.split_charge else 1

    # create output file names
    detector_jet_file_name = 'det_jet.txt'
    detector_image_file_name = 'det_image.txt'
    generator_jet_file_name = 'gen_jet.txt'
    generator_image_file_name = 'gen_image.txt'

    if args.detailed_file_name:
        detector_jet_file_name = 'detector_jet_pthat_{}_{}_ptmin_{}_R_{}.txt'.format(
            args.pt_hat_min, args.pt_hat_max, args.jet_pt_min, args.jet_radius)
        detector_image_file_name = 'detector_image_pthat_{}_{}_ptmin_{}_R_{}.txt'.format(
            args.pt_hat_min, args.pt_hat_max, args.jet_pt_min, args.jet_radius)
        generator_jet_file_name = 'generator_jet_pthat_{}_{}_ptmin_{}_R_{}.txt'.format(
            args.pt_hat_min, args.pt_hat_max, args.jet_pt_min, args.jet_radius)
        generator_image_file_name = 'generator_image_pthat_{}_{}_ptmin_{}_R_{}.txt'.format(
            args.pt_hat_min, args.pt_hat_max, args.jet_pt_min, args.jet_radius)

    # open files to write to
    os.makedirs(args.output_dir, exist_ok=True)
    det_jet_file_name = os.path.join(args.output_dir, detector_jet_file_name)
    det_image_file_name = os.path.join(args.output_dir, detector_image_file_name)
    det_jet_file = open(det_jet_file_name, 'w')
    det_image_file = open(det_image_file_name, 'w')
    gen_jet_file_name = os.path.join(args.output_dir, generator_jet_file_name)
    gen_image_file_name = os.path.join(args.output_dir, generator_image_file_name)
    gen_jet_file = open(gen_jet_file_name, 'w')
    gen_image_file = open(gen_image_file_name, 'w')

    # start loop
    for _ in range(args.events):
        # generate event
        g.next()

        # get jets and match geometrically
        det_jets = g.detector_jets()
        gen_jets = g.generator_jets()

        if len(det_jets) == 0 or len(gen_jets) == 0:
            continue

        # we will match on the highest pt generator jet
        truth_jet = gen_jets[0]
        circle_selector = fastjet.SelectorCircle(args.jet_radius)
        circle_selector.set_reference(truth_jet)
        candidates = fastjet.sorted_by_pt(circle_selector(det_jets))

        # if there are any matches, the highest pT jet will be selected
        if len(candidates) == 0:
            continue
        detector_jet = candidates[0]

        # now, generate images for leading and subleading jet
        det_jet_image = image.pseudojet_to_image(detector_jet, image_params)
        gen_jet_image = image.pseudojet_to_image(truth_jet, image_params)

        # write jets to file
        image.write_image_to_file(det_jet_file, det_image_file, detector_jet, det_jet_image)
        image.write_image_to_file(gen_jet_file, gen_image_file, truth_jet, gen_jet_image)


if __name__ == '__main__':
    # setup logger
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # setup command line flags
    parser = argparse.ArgumentParser(
        description='Generates Pythia events, generates a smeared detector event using various user-selected methods, prints matched truth and smeared jets to csv')
    parser.add_argument('--events', type=int, default=5000,
                        help=' number of events to generate')
    parser.add_argument('--output-dir', default="data/raw",
                        help="directory to write output to")
    parser.add_argument('--jet-radius', type=float, default=0.4,
                        help='radius for jetfinding')
    parser.add_argument('--gaussian-smear', default=True,
                        dest='gaussian_smear', action='store_true')
    parser.add_argument('--no-gaussian-smear',
                        dest='gaussian_smear', action='store_false')
    parser.add_argument('--gaussian-mean', type=float, default=1.3,
                        help='mean energy shift when using gaussian smearing')
    parser.add_argument('--gaussian-width', type=float, default=0.2,
                        help='energy smearing width when using gaussian smearing')
    parser.add_argument('--tpc-sim', default=False,
                        dest='tpc_sim', action='store_true')
    parser.add_argument('--no-tpc-sim',
                        dest='tpc_sim', action='store_false')
    parser.add_argument('--tpc-efficiency', type=float, default=0.9,
                        help='TPC efficiency when simulating TPC')
    parser.add_argument('--star-sim', default=False,
                        dest='star_sim', action='store_true')
    parser.add_argument('--no-star-sim',
                        dest='star_sim', action='store_false')
    parser.add_argument('--image-width-eta', type=float, default=1.0,
                        help='width of jet image')
    parser.add_argument('--image-width-phi', type=float, default=1.0,
                        help='height of jet image')
    parser.add_argument('--pixels-eta', type=int, default=32,
                        help='pixels in eta')
    parser.add_argument('--pixels-phi', type=int, default=32,
                        help='pixels in phi direction')
    parser.add_argument('--split-charge', type=bool, default=True,
                        help='if true, split charged and neutral into two layers')
    parser.add_argument('--pt-hat-min', type=float, default=20.0,
                        help='minimum pt-hat')
    parser.add_argument('--pt-hat-max', type=float, default=30.0,
                        help='maximum pt-hat')
    parser.add_argument('--track-eta-max', type=float, default=1.0,
                        help='maximum absolute track eta')
    parser.add_argument('--jet-pt-min', type=float, default=10.0,
                        help='minimum reconstructed jet pt')
    parser.set_defaults(detailed_file_name=False)
    parser.add_argument('--detailed-file-name', dest='detailed_file_name',
                        action='store_true')
    parser.add_argument('--no-detailed-file-name', dest='detailed_file_name',
                        action='store_false')

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main(parser.parse_args())
