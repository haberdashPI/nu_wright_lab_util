import pandas as pd
import glob
import os
import datetime
from progressbar import ProgressBar


def load_files(track_file,source,clean_up_data=lambda d,f: d,
               timestamp='timestamp',load_from_source=True,
               reload_all=False,patterns=['*.dat']):

  def isnew_track(old_tracks):
    def fn(f):
      return (datetime.datetime.fromtimestamp(os.path.getmtime(f)) >
              old_tracks[timestamp].max())
    return fn

  tracks = None
  if load_from_source or (not os.path.isfile(track_file)):
      loaded_tracks = []
      old_tracks = None

      files = reduce(lambda x,y: x+y,
                     map(lambda p: glob.glob(source + p),patterns))

      if not reload_all and os.path.isfile(track_file):
          old_tracks = pd.read_hdf(track_file,'df')
          files = filter(isnew_track(old_tracks),files)

      if len(files) > 0:
          progress = ProgressBar(maxval=len(files)).start()

          for i,f in enumerate(files):
              progress.update(i)
              loaded_tracks.append(clean_up_data(pd.read_csv(f),f))

          tracks = pd.concat(filter(lambda x: x is not None,loaded_tracks)) 
          if old_tracks is not None:
              new_tracks = tracks[tracks[timestamp] > old_tracks[timestamp].max()]
              tracks = pd.concat([old_tracks,new_tracks])

          tracks.to_hdf(track_file,'df',complevel=9,complip='bzip2')
          tracks = tracks.reset_index()
          progress.finish()

      elif os.path.isfile(track_file):
          tracks = old_tracks
          tracks = tracks.reset_index()
  else:
      tracks = pd.read_hdf(track_file,'df')

  return tracks