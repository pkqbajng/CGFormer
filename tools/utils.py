import yaml
import numpy as np

def get_inv_map():
  '''
  remap_lut to remap classes of semantic kitti for training...
  :return:
  '''
  config_path = "./tools/SemanticKITTI.yaml"
  dataset_config = yaml.safe_load(open(config_path, 'r'))
  # make lookup table for mapping
  inv_map = np.zeros(20, dtype=np.int32)
  inv_map[list(dataset_config['learning_map_inv'].keys())] = list(dataset_config['learning_map_inv'].values())

  return inv_map