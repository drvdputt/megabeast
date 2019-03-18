import math

import argparse
import numpy as np
import h5py
import itertools as it
from astropy import wcs
from astropy.io import fits
from astropy.table import Table, vstack


def setup_spatial_regions(cat,
                          pix_size=10.0):
    """
    The spatial regions are setup via a WCS object

    Parameters
    ----------
    cat : astropy Table
       catalog of BEAST results
    pix_size : float
       size of pixels/regions in arcsec
    Returns
    -------
    wcs_info: astropy WCS object
    """
    # min/max ra
    min_ra = cat['RA'].min()
    max_ra = cat['RA'].max()
    min_dec = cat['DEC'].min()
    max_dec = cat['DEC'].max()

    # ra/dec delta values
    dec_delt = pix_size/3600.
    ra_delt_physical = dec_delt
    ra_delt = ra_delt_physical / math.cos(math.radians(0.5 * (max_dec + min_dec)))

    # compute the number of pixels and
    n_y = int(np.rint((max_dec - min_dec)/dec_delt) + 1)
    n_x = int(np.rint((max_ra-min_ra)/ra_delt) + 1)
    print('# of x & y pixels = ', n_x, n_y)

    # The grid should be slightly wider than the data range
    if (n_x * ra_delt < max_ra - min_ra):
        raise 'grid does not cover RA range!'
    if (n_y * dec_delt < max_dec - min_dec):
        raise 'grid does not cover DEC range!'

    # use the data center as the grid center. Since the grid is slightly
    # larger than the data, the bins at the edges will have somewhat
    # less sources
    data_center_dec = (min_dec + max_dec) / 2
    data_center_ra = (min_ra + max_ra) / 2

    w = wcs.WCS(naxis=2)
    w.wcs.crpix = np.asarray([n_x, n_y], dtype=float) / 2.
    w.wcs.cdelt = [-ra_delt_physical, dec_delt]
    w.wcs.crval = [data_center_ra, data_center_dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    return (w, n_x, n_y)


def regions_for_objects(ra,
                        dec,
                        wcs_info):
    """
    Generate the x,y coordinates for each object based on the input
    ra/dec and already created WCS information.

    Parameters
    ----------
    ra : array of float
       right ascension of the objects
    dec : array of float
       declination of the objects
    wcs_info: astropy WCS object
       previously generated WCS object based on the full catalog
    Returns
    -------
    dictonary of:
    x : int array
      x values of regions
    y : int array
      y values of regions
    name : str array
      string array composed of x_y
    """

    # generate the array needed for fast conversion
    world = np.empty((len(ra), 2), float)
    world[:, 0] = ra
    world[:, 1] = dec

    # convert
    pixcrd = wcs_info.wcs_world2pix(world, 0)

    # get the arrays to return
    x = np.rint(pixcrd[:, 0]).astype(int)
    y = np.rint(pixcrd[:, 1]).astype(int)
    xy_name = [None]*len(ra)

    for k in range(len(x)):
        xy_name[k] = str(x[k]) + '_' + str(y[k])

    # return the results as a dictonary
    #   values are truncated to provide the ids for the subregions
    return {'x': x, 'y': y, 'name': xy_name}


if __name__ == '__main__':

    # command line params to specify the run directory
    #   and any other needed parameters

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument('stats_filename', metavar='fname',
                        type=str, nargs='+',
                        help="Filename(s) of the stats")
    parser.add_argument("--pix_size", default=10., type=float,
                        help="pixel scale [arcsec]")
    args = parser.parse_args()

    stats_filename = args.stats_filename

    # type of statistic (make a commandline parameter later)
    #   remember to add to output filenames
    stat_type = 'Exp'

    # read in the full brick catalog
    cat = Table.read(stats_filename[0])
    if len(stats_filename) > 0:
        for fname in stats_filename[1:]:
            tcat = Table.read(fname)
            cat = vstack([cat, tcat])

    # generate the wcs info for the output FITS files
    #    also provides the mapping info from ra,dec to x,y
    wcs_info, n_x, n_y = setup_spatial_regions(cat,
                                               pix_size=args.pix_size)

    # get the pixel coordinates for each source
    xy_vals = regions_for_objects(cat['RA'],
                                  cat['DEC'],
                                  wcs_info)

    x = xy_vals['x']
    y = xy_vals['y']

    # setup arrary to store summary stats per pixel
    n_sum = 2
    sum_stats = ['Av', 'Rv', 'f_A']
    n_sum = len(sum_stats)
    summary_stats = np.zeros((n_y, n_x, n_sum+1), dtype=float)
    summary_sigmas = np.zeros((n_y, n_x, n_sum), dtype=float)
    values_foreach_pixel = {cur_stat: {(i, j): [] for i in range(n_x) for j in range(n_y)}
                            for cur_stat in sum_stats}

    # loop through the pixels and generate the summary stats
    for i in range(n_x):
        for j in range(n_y):
            tindxs, = np.where((x == i) & (y == j))
            # tindxs, = np.where((x == i) & (y == j) & (cat['chi2min'] < 10.))
            if len(tindxs) > 0:
                summary_stats[j, i, n_sum] = len(tindxs)
                print(i, j, len(tindxs))
                for k, cur_stat in enumerate(sum_stats):
                    values = cat[cur_stat + '_' + stat_type][tindxs]
                    values_foreach_pixel[cur_stat][i, j] = values
                    summary_stats[j, i, k] = np.average(values)
                    summary_sigmas[j, i, k] = np.std(values, ddof=1) / math.sqrt(len(values))

    master_header = wcs_info.to_header()
    # Now, write the maps to disk
    for k, cur_stat in enumerate(sum_stats):
        map_name = stats_filename[0].replace('stats', 'map' + cur_stat)
        hdu = fits.PrimaryHDU(summary_stats[:, :, k],
                              header=master_header)
        hdu.writeto(map_name, overwrite=True)

        sigma_name = map_name.replace('map', 'map_sigma')
        hdu_sigma = fits.PrimaryHDU(summary_sigmas[:, :, k],
                                    header=master_header)
        hdu_sigma.writeto(sigma_name, overwrite=True)

    hdu = fits.PrimaryHDU(summary_stats[:, :, n_sum],
                          header=master_header)
    hdu.writeto(stats_filename[0].replace('stats', 'npts'),
                overwrite=True)

    # And store all the values in HDF5 format
    values_name = stats_filename[0].replace('stats.fits',
                                            'values_per_pixel.hd5')
    f = h5py.File(values_name, 'w')
    dt = h5py.special_dtype(vlen=np.dtype(np.float))
    for cur_stat in sum_stats:
        dset = f.create_dataset(cur_stat, (n_x, n_y), dtype=dt)
        for i, j in it.product(range(n_x), range(n_y)):
            dset[i, j] = values_foreach_pixel[cur_stat][i, j]
