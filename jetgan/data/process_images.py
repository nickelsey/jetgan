import argparse
import logging
import os
import time
from pathlib import Path
from shutil import copyfile

from generators import PythiaGenerator
import image

def main(args):
  """
  Preprocessing of raw jet images before training.
  """
  logger = logging.getLogger(__name__)
  logger.info('processing jet image files:')
  logger.info('detector images: {}'.format(args.detector_images))
  logger.info('generator images: {}'.format(args.generator_images))
  
  # currently, only support copy from temporary location (data/raw, for instance)
  # to final location. 
  # build output file names
  det_file_name = args.detector_output_name
  gen_file_name = args.generator_output_name
  time_string = time.strftime("%Y%m%d_%H%M%S")

  if args.tag_with_mods:
    pass
  if args.tag_with_date:
    det_file_name = det_file_name + '_' + time_string
    gen_file_name = gen_file_name + '_' + time_string
  
  det_file_name = det_file_name + '.txt'
  gen_file_name = gen_file_name + '.txt'

  # now create the output files
  os.makedirs(args.output_directory, exist_ok=True)
  det_file_path = os.path.join(args.output_directory, det_file_name)
  gen_file_path = os.path.join(args.output_directory, gen_file_name)
  
  # currently, we do not do any preprocessing, so simply copy the input
  # file to the output file location
  copyfile(args.detector_images, det_file_path)
  copyfile(args.generator_images, gen_file_path)

  logger.info('detector images: {} copied to {}'.format(args.detector_images, det_file_path))
  logger.info('generator images: {} copied to {}'.format(args.generator_images, gen_file_path))




if __name__ == '__main__':
    # setup logger
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # setup command line flags
    parser = argparse.ArgumentParser(
        description='Preprocessing for raw jet images before training')
    parser.add_argument('--detector-images', default='data/raw/detector.txt',
                        help='file containing detector jets and jet images')
    parser.add_argument('--generator-images', default='data/raw/generator.txt',
                        help='file containing generator jets and jet images')
    parser.add_argument('--output-directory', default='data/processed',
                        help='output directory')
    parser.add_argument('--detector-output-name', default='detector',
                        help='output file name for detector jets')
    parser.add_argument('--generator-output-name', default='generator',
                        help='output file name for generator jets')
    parser.set_defaults(tag_with_mods=False)
    parser.add_argument('--tag-with-mods', dest='tag_with_mods', action='store_true',
                        help='append modifications to output filename')
    parser.add_argument('--no-tag-with-mods', dest='tag_with_mods',
                        action='store_false')
    parser.set_defaults(tag_with_date=False)
    parser.add_argument('--tag-with-date', dest='tag_with_date', action='store_true',
                        help='append date/time to output filename')
    parser.add_argument('--no-tag-with-date', dest='tag_with_date',
                        action='store_false')
    

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    

    main(parser.parse_args())