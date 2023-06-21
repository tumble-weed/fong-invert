#!/usr/bin/env python3

import pdb
import sys
import ipdb.__main__

def foo(x):
  return x+1

class RunningTrace():
  def set_running_trace(self):
    frame = sys._getframe().f_back
    self.botframe = None
    self.setup(frame, None)
    while frame:
      frame.f_trace = self.trace_dispatch
      self.botframe = frame
      frame = frame.f_back
    self.set_continue()
    self.quitting = False
    sys.settrace(self.trace_dispatch)

class ProgrammaticPdb(pdb.Pdb, RunningTrace):
  pass

class ProgrammaticIpdb(ipdb.__main__._get_debugger_cls(), RunningTrace):
  pass
if __name__ == '__main__':
    p = ProgrammaticPdb()
    # p = ProgrammaticIpdb(ipdb.__main__.def_colors)

    p.onecmd('b bar.py:38') # Works before set_running_trace
    p.set_running_trace()
    p.onecmd('b foo')       # only works after calling set_running_trace
    p.onecmd('l')

    x=-1
    x=2
    x=0
    x=foo(x)

    print(x)