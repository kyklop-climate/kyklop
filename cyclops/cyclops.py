#!/usr/bin/python
# -*- coding: utf-8 -*-

import datetime
import glob
import matplotlib
from matplotlib import cm, colors, pyplot
from matplotlib.animation import FuncAnimation
from mpl_toolkits.basemap import Basemap, maskoceans
from netCDF4 import Dataset
import numpy
from numpy import logical_or, logical_and
import pytz
from scipy.interpolate import interp1d, interp2d
from skimage.morphology import remove_small_objects, binary_opening, disk, binary_closing
from skimage.measure import label, regionprops
import sys

REFERENCE_TIME = datetime.datetime(1949, 12, 1, tzinfo=pytz.utc)


def hrs_to_date(hrs):
    dt = datetime.timedelta(hours=int(hrs))
    return (REFERENCE_TIME+dt).date()


def date_to_hrs(date):
    if isinstance(date, datetime.date):
        date = datetime.datetime.combine(date, datetime.time(tzinfo=pytz.utc))
    dt = date-REFERENCE_TIME
    return int(dt.total_seconds()/3600)


class CycloneDetector(object):
    def __init__(self, filename):
        nc = Dataset(filename)
        self.debug_mode = 1.
        print self.debug_mode
        self.dt = nc.variables["time"][1]-nc.variables["time"][0]
        self.xlon = nc.variables["xlon"][:]
        self.xlat = nc.variables["xlat"][:]
        N_lon = self.xlon.shape[1]
        N_lat = self.xlat.shape[0]
        #self.idx_to_lon = interp1d(numpy.arange(N_lon), self.xlon[0,:])
        #self.idx_to_lat = interp1d(numpy.arange(N_lat), self.xlat[:,0])
        self.idx_to_lon = interp2d(numpy.arange(N_lon), numpy.arange(N_lat), self.xlon)
        self.idx_to_lat = interp2d(numpy.arange(N_lon), numpy.arange(N_lat), self.xlat)
        self.left = self.xlon.min()
        self.right = self.xlon.max()
        self.bottom = self.xlat.min()+18
        self.top = self.xlat.max()-8
        self.m = Basemap(llcrnrlon=self.left,
                         llcrnrlat=self.bottom,
                         urcrnrlon=self.right,
                         urcrnrlat=self.top,
                         projection='cyl', resolution='l')
        self.ocean_mask = maskoceans(self.xlon, self.xlat, self.xlon).mask
        self.time_min = 0.
        self.time_max = 0.
        self.no_ingested = 0
        self.masks = numpy.empty((0,)+self.ocean_mask.shape)
        self.tc_table = numpy.empty((0,5))
        self.time = numpy.empty((0,))

    def detect_tc_in_step(self, nc, i, ecc_th=0.75):
        mask = self.ocean_mask.copy()
        uas = nc.variables["uas"][i].squeeze()
        vas = nc.variables["vas"][i].squeeze()
        wind_speed = numpy.sqrt(uas**2+vas**2)
        wind_mask = logical_and(self.ocean_mask, wind_speed > 20.)
        temp = nc.variables["ts"][i].squeeze()
        temp_mask = logical_and(self.ocean_mask, temp > 298.15)
        ps = nc.variables["ps"][i].squeeze()
        ps_mask = logical_and(self.ocean_mask, ps < 1005)
        mask = logical_or(wind_mask, logical_and(temp_mask, ps_mask))
        mask = remove_small_objects(mask, 20)
        lbl = label(mask)
        props_windspeed = regionprops(lbl, wind_speed)
        props_pressure = regionprops(lbl, ps)
        centroids = []
        for windspeed, pressure in zip(props_windspeed, props_pressure):
            max_wind_speed = windspeed["max_intensity"]
            min_pressure = pressure["min_intensity"]
            if windspeed["eccentricity"] > ecc_th or max_wind_speed<20.:
                lbl[lbl == windspeed["label"]]=0
            else:
                y, x = windspeed["centroid"]
                lon = float(self.idx_to_lon(x, y))
                lat = float(self.idx_to_lat(x, y))
                centroids.append([lon, lat, max_wind_speed, min_pressure])
        mask = lbl>0
        return mask, centroids

    def ingest_netcdf(self, filename):
        nc = Dataset(filename)
        time = nc.variables["time"][:]
        time_min = time.min()
        time_max = time.max()
        if time_min < self.time_max:
            raise RuntimeError("Non increasing times.")
        self.time = numpy.hstack([self.time, time])
        tc_table = []
        masks = numpy.empty(nc.variables["ps"].shape)
        for i in xrange(len(time)):
            masks[i,:,:], tcs = self.detect_tc_in_step(nc, i)
            for tc in tcs:
                row = [time[i]]
                row.extend(tc)
                tc_table.append(row)
        tc_table = numpy.array(tc_table)
        if self.no_ingested==0:
            self.time_min = time_min
        if self.debug_mode:
            self.masks = numpy.vstack([self.masks, masks])
        if len(tc_table)>0:
            self.tc_table = numpy.vstack([self.tc_table, tc_table])
        self.time_max = time_max
        self.no_ingested += 1

    def tracking(self, max_inactive_time=24., max_distance=6., min_length=4.):
        latest_track = 0
        active_tracks = []
        tracked_tcs = numpy.empty((self.tc_table.shape[0], self.tc_table.shape[1]+1))
        if len(self.tc_table)==0:
            self.tracked_tcs = tracked_tcs
        tracked_tcs[:,:-1] = self.tc_table
        tracked_tcs[:,-1] = 0.
        for i, (time, lon, lat, ws, mp) in enumerate(self.tc_table):
            new_active_tracks = []
            for at in active_tracks:
                lt, llon, llat, lws, lmp, ltno = tracked_tcs[tracked_tcs[:,-1]==at][-1]
                dt = time-lt
                if dt < max_inactive_time:
                    dist = numpy.sqrt((lon-llon)**2 + (lat-llat)**2)
                    if dist <= max_distance:
                        tracked_tcs[i,-1] = at
                    new_active_tracks.append(at)
            active_tracks = new_active_tracks
            if tracked_tcs[i,-1]==0:
                latest_track = tracked_tcs[i,-1] = latest_track + 1
                active_tracks.append(latest_track)
        lengths = numpy.bincount(tracked_tcs[:,-1].astype(int))
        mask = numpy.ones((len(tracked_tcs),), dtype='bool')
        for no, l in enumerate(lengths):
            if l < min_length:
                mask[tracked_tcs[:,-1]==no] = False
        tracked_tcs = tracked_tcs[mask]
        old_track_nos = set(tracked_tcs[:,-1])
        old_track_nos.discard(0)
        old_track_nos = sorted(old_track_nos)
        for i, o in enumerate(old_track_nos):
            tracked_tcs[tracked_tcs[:,-1]==o,-1] = i+1
        self.tracked_tcs = tracked_tcs

    def show_tcs(self, tracked_tcs=None):
        fig = pyplot.figure()
        if self.debug_mode:
           #xs, ys = self.m(self.xlon, self.xlat)
           quad = self.m.pcolormesh(self.xlon, self.xlat, self.masks[0], vmin=0., vmax=1.)
        self.m.drawcoastlines()
        self.m.drawparallels(numpy.arange(-14, 49, 10), labels=[1,0,0,0])
        self.m.drawmeridians(numpy.arange(-140, 20, 10), labels=[0,0,0,1])
        texts = []
        if self.tracked_tcs != None and len(self.tracked_tcs)>0:
            min_speed = self.tracked_tcs[:,3].min()
            max_speed = self.tracked_tcs[:,3].max()
            wind_speed_norm = colors.Normalize(vmin=min_speed, vmax=max_speed)
            wind_speed_mapper = cm.ScalarMappable(norm=wind_speed_norm)
            for at in xrange(1, int(self.tracked_tcs[:,-1].max())+1):
                inds = self.tracked_tcs[:,-1]==at
                lons = self.tracked_tcs[inds][:,1]
                lats = self.tracked_tcs[inds][:,2]
                ws = self.tracked_tcs[inds][:,3]
                xs, ys = self.m(lons, lats)
                h = len(xs)/2
                self.m.plot(xs, ys, label="{}".format(at))
                text = pyplot.text(xs[0], ys[0], "{:.2f}".format(ws[0]))
                text.set_visible(False)
                texts.append(text)
                #pyplot.text(xs[h], ys[h], "{:.2f}".format(numpy.average(ws)))
                #pyplot.text(xs[-1], ys[-1], "{:.2f}".format(ws[-1]))
        #pyplot.legend(loc=3, bbox_to_anchor=(0., 1.02, 1., .102), ncol=6)
        #if self.debug_mode:
           #pyplot.colorbar()
        ttl = pyplot.title("{}/({}-{})".format(hrs_to_date(self.time_min),
                                                    hrs_to_date(self.time_min),
                                                    hrs_to_date(self.time_max)))
        def animate(i):
            ttl.set_text("{} in ({} to {}) ({})".format(hrs_to_date(self.time[i]),
                                                        hrs_to_date(self.time_min),
                                                        hrs_to_date(self.time_max), i))
            time = self.time[i]
            ind = self.tracked_tcs[:,0]==time
            for text in texts:
                text.set_visible(False)
            for tc in self.tracked_tcs[ind]:
                text = texts[int(tc[-1])-1]
                text.set_text("{:.2f} m/s".format(tc[3]))
                text.set_visible(True)
                text.set_position((tc[1], tc[2]))
                text.set_backgroundcolor(wind_speed_mapper.to_rgba(tc[3]))
            quad.set_array(self.masks[i,:-1,:-1].ravel())
            return quad,
        if self.debug_mode:
            ani = FuncAnimation(fig, animate, interval=10., frames=len(self.masks), repeat_delay=1000)
        #ani.save("ani_ca_atmonly.mp4", "avconv", fps=20, extra_args=['-vcodec', 'libx264'])
        pyplot.show()


def build_filename_list():
    filenames = []
    for g in sys.argv[1:]:
        filenames.extend(glob.glob(g))
    return sorted(filenames)


def main():
    matplotlib.verbose.level = 'debug'
    filenames = build_filename_list()
    detector = CycloneDetector(filenames[0])
    for fn in filenames:
        detector.ingest_netcdf(fn)
    detector.tracking()
    tracked_tcs = detector.tracked_tcs
    numpy.savetxt("tcs.dat", tracked_tcs)
    detector.show_tcs()


if __name__ == "__main__":
    main()
