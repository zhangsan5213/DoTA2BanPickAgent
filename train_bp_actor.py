import os
import re
import json
import torch
import random

import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from utils.raw_data import HERO_ID_FEATURE_MAP, HERO_ID_SEMANTIC_MAP
from utils.get_data_cm_bp import fetch_high_mmr_matches
from policy.bp_policy_module import *

torch.random.manual_seed(42)