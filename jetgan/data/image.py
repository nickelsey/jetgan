"""
Methods for converting 4-vectors (such as PseudoJets) into
images that can be saved to csv or used directly as input 
for training or validation.
"""

import numpy as np
from pandas import read_csv

from math import pi

def find_bin(val, nbins, min_val, delta):
  if val < min_val:
    return -1
  elif val > min_val + (nbins * delta):
    return -1
  else:
    return int((val - min_val) / delta)
  

def pseudojet_to_image(jet, params):
  """
  creates a jet image from the jet constituents. X-axis=eta, 
  Y-axis=phi, Z-axis=|charge|
  """
  
  x_bins = params['x_bins']
  x_min  = params['x_min']
  x_max  = params['x_max']
  y_bins = params['y_bins']
  y_min  = params['y_min']
  y_max  = params['y_max']
  z_bins = params['z_bins']

  x_delta = (x_max - x_min) / x_bins
  y_delta = (y_max - y_min) / y_bins

  # if x_min >= x_max:
  #   raise('x axis must have positive, nonzero length')

  # if y_min >= y_max:
  #   raise('y axis must have positive, nonzero length')

  # if z_bins != 1 or z_bins != 2:
  #   raise('z can only have 1 or 2 bins (charge)')

  image = np.zeros(shape=(x_bins, y_bins, z_bins))

  for constituent in jet.constituents():
    deta = constituent.eta() - jet.eta()
    dphi = jet.delta_phi_to(constituent)
    charge = constituent.user_index()
    pt = constituent.pt()

    x_bin = find_bin(deta, x_bins, x_min, x_delta)
    y_bin = find_bin(dphi, y_bins, y_min, y_delta)
    z_bin = abs(charge) if z_bins == 2 else 0

    if x_bin == -1 or y_bin == -1:
      continue

    image[x_bin, y_bin, z_bin] += pt
  
  return image

def write_image_to_file(jet_file, image_file, jet, image):
  jet_txt = '{}, {}, {}, {}\n'.format(jet.pt(), jet.eta(), jet.phi(), jet.m())
  jet_file.write(jet_txt)

  flattened_image = image.flatten()
  image_file.write(','.join(['%.6f' % f for f in flattened_image]))
  image_file.write('\n')


def load_jet_images(jet_file, image_file, image_width=32, image_height=32,
                    image_channels=2, delimiter=','):
  image_size = image_width * image_height * image_channels
  df_images = read_csv(image_file, sep=delimiter, index_col=False, names=range(
              image_size), header=None, dtype=np.float32, engine='python')
  df_jets = read_csv(jet_file, sep=delimiter, index_col=False, names=range(
              image_size), header=None, dtype=np.float32, engine='python')
  
  images = df_images.values
  images = images.reshape((images.shape[0], image_width, image_height, image_channels))

  jets = df_jets.values

  assert images.shape[0] == jets.shape[0]

  return jets, images
  
