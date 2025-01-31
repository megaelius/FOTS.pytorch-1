'''
Created on Aug 31, 2017

@author: Michal.Busta at gmail.com
'''
import torch

def np_to_variable(x, dtype=torch.float32):
  v = torch.from_numpy(x).type(dtype)
  return v

def load_net(fname, net,device, optimizer=None):
  sp = torch.load(fname,map_location=device)
  step = sp['step']
  try:
    learning_rate = sp['learning_rate']
  except:
    import traceback
    traceback.print_exc()
    learning_rate = 0.001
  opt_state = sp['optimizer']
  sp = sp['state_dict']
  for k, v in net.state_dict().items():
    try:
      param = sp[k]
      v.copy_(param)
    except:
      import traceback
      traceback.print_exc()

  if optimizer is not None:
    try:
      optimizer.load_state_dict(opt_state)
    except:
      import traceback
      traceback.print_exc()

  print(fname)
  return step, learning_rate
