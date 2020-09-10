from numba import njit, prange
import numpy as np
import dask.array as da
from daskms import xds_from_table
import pywt
from ducc0.wgridder import ms2dirty, dirty2ms
from ducc0.fft import r2c, c2r, c2c
from africanus.gps.kernels import exponential_squared as expsq
from africanus.linalg import kronecker_tools as kt
from pfb.wavelets.wavelets import wavedecn, waverecn, ravel_coeffs, unravel_coeffs

iFs = np.fft.ifftshift
Fs = np.fft.fftshift


@njit(parallel=True, nogil=True, fastmath=True, inline='always')
def freqmul(A, x):
    nchan, npix = x.shape
    out = np.zeros((nchan, npix), dtype=x.dtype)
    for i in prange(npix):
        for j in range(nchan):
            for k in range(nchan):
                out[j, i] += A[j, k] * x[k, i]
    return out

@njit(parallel=True, nogil=True, fastmath=True, inline='always')
def make_kernel(nx_psf, ny_psf, sigma0, length_scale):
    K = np.zeros((1, nx_psf, ny_psf), dtype=np.float64)
    for i in range(nx_psf):
        for j in range(ny_psf):
            l = float(i - (nx_psf//2))
            m = float(j - (ny_psf//2))
            K[0,i,j] = sigma0**2*np.exp(-(l**2+m**2)/(2*length_scale**2))
    return K

class Gridder(object):
    def __init__(self, uvw, freq, sqrtW, nx, ny, cell_size, nband=None, precision=1e-7, ncpu=8, do_wstacking=1):
        self.wgt = sqrtW
        self.uvw = uvw
        self.nrow = uvw.shape[0]
        self.nx = nx
        self.ny = ny
        self.cell = cell_size * np.pi/60/60/180
        self.freq = freq
        self.precision = precision
        self.nthreads = ncpu
        self.do_wstacking = do_wstacking
        self.flags = np.where(self.wgt==0, 1, 0)

        # freq mapping
        self.nchan = freq.size
        if nband is None or nband == 0:
            self.nband = self.nchan
        else:
            self.nband = nband
        step = self.nchan//self.nband
        freq_mapping = np.arange(0, self.nchan, step)
        self.freq_mapping = np.append(freq_mapping, self.nchan)
        self.freq_out = np.zeros(self.nband)
        for i in range(self.nband):
            Ilow = self.freq_mapping[i]
            Ihigh = self.freq_mapping[i+1]
            self.freq_out[i] = np.mean(self.freq[Ilow:Ihigh])  # weighted mean?

    def dot(self, x):
        model_data = np.zeros((self.nrow, self.nchan), dtype=np.complex128)
        for i in range(self.nband):
            Ilow = self.freq_mapping[i]
            Ihigh = self.freq_mapping[i+1]
            model_data[:, Ilow:Ihigh] = dirty2ms(uvw=self.uvw, freq=self.freq[Ilow:Ihigh], dirty=x[i], wgt=self.wgt[:, Ilow:Ihigh],
                                                 pixsize_x=self.cell, pixsize_y=self.cell, epsilon=self.precision,
                                                 nthreads=self.nthreads, do_wstacking=self.do_wstacking, verbosity=0)
        return model_data

    def hdot(self, x):
        image = np.zeros((self.nband, self.nx, self.ny), dtype=np.float64)
        for i in range(self.nband):
            Ilow = self.freq_mapping[i]
            Ihigh = self.freq_mapping[i+1]
            image[i] = ms2dirty(uvw=self.uvw, freq=self.freq[Ilow:Ihigh], ms=x[:, Ilow:Ihigh], wgt=self.wgt[:, Ilow:Ihigh],
                                npix_x=self.nx, npix_y=self.ny, pixsize_x=self.cell, pixsize_y=self.cell, epsilon=self.precision,
                                nthreads=self.nthreads, do_wstacking=self.do_wstacking, verbosity=0)
        return image

    def make_psf(self):
        psf_array = np.zeros((self.nband, 2*self.nx, 2*self.ny))
        for i in range(self.nband):
            Ilow = self.freq_mapping[i]
            Ihigh = self.freq_mapping[i+1]
            psf_array[i] = ms2dirty(uvw=self.uvw, freq=self.freq[Ilow:Ihigh],
                                    ms=self.wgt[:, Ilow:Ihigh].astype(np.complex128), wgt=self.wgt[:, Ilow:Ihigh],
                                    npix_x=2*self.nx, npix_y=2*self.ny, pixsize_x=self.cell, pixsize_y=self.cell,
                                    epsilon=self.precision, nthreads=self.nthreads, do_wstacking=self.do_wstacking)
        return psf_array

    def convolve(self, x):
        return self.hdot(self.dot(x))


class OutMemGridder(object):
    def __init__(self, table_name, nx, ny, cell_size, freq, nband=None, field=0, precision=1e-7, ncpu=8, do_wstacking=1,
                 data_column='DATA', weight_column='IMAGING_WEIGHT'):
        if precision > 1e-6:
            self.real_type = np.float32
            self.complex_type = np.complex64
        else:
            self.real_type = np.float64
            self.complex_type=np.complex128

        self.nx = nx
        self.ny = ny
        self.cell = cell_size * np.pi/60/60/180
        if isinstance(field, list):
            self.field = field
        else:
            self.field = [field]
        self.precision = precision
        self.nthreads = ncpu
        self.do_wstacking = do_wstacking

        # freq mapping
        self.freq = freq
        self.nchan = freq.size
        if nband is None:
            self.nband = self.nchan
        else:
            self.nband = nband
        step = self.nchan//self.nband
        freq_mapping = np.arange(0, self.nchan, step)
        self.freq_mapping = np.append(freq_mapping, self.nchan)
        self.freq_out = np.zeros(self.nband)
        for i in range(self.nband):
            Ilow = self.freq_mapping[i]
            Ihigh = self.freq_mapping[i+1]
            self.freq_out[i] = np.mean(self.freq[Ilow:Ihigh])

        self.chan_chunks = self.freq_mapping[1] - self.freq_mapping[0]

        # meta info for xds_from_table
        self.data_column = data_column
        self.weight_column = weight_column
        self.table_name = table_name
        self.schema = {
            data_column: {'dims': ('chan',)},
            weight_column: {'dims': ('chan', )},
            "UVW": {'dims': ('uvw',)},
        }

    def make_residual(self, x, v_dof=None):
        print("Making residual")
        residual = np.zeros(x.shape, dtype=x.dtype)
        xds = xds_from_table(self.table_name, group_cols=('FIELD_ID'), chunks={"row":-1, "chan": self.chan_chunks}, table_schema=self.schema)
        for ds in xds:
            if ds.FIELD_ID not in list(self.field):
                continue
            print("Processing field %i"%ds.FIELD_ID)
            data = getattr(ds, self.data_column).data
            weights = getattr(ds, self.weight_column).data
            uvw = ds.UVW.data.compute().astype(self.real_type)

            for i in range(self.nband):
                Ilow = self.freq_mapping[i]
                Ihigh = self.freq_mapping[i+1]
                weighti = weights.blocks[:, i].compute().astype(self.real_type)
                datai = data.blocks[:, i].compute().astype(self.complex_type)

                # TODO - load and apply interpolated fits beam patterns for field

                # get residual vis
                if weighti.any():
                    residual_vis = weighti * datai - ng.dirty2ms(uvw=uvw, freq=self.freq[Ilow:Ihigh], dirty=x[i], wgt=weighti,
                                                                pixsize_x=self.cell, pixsize_y=self.cell, epsilon=self.precision,
                                                                nthreads=self.nthreads, do_wstacking=self.do_wstacking, verbosity=0)

                    # make residual image
                    residual[i] += ng.ms2dirty(uvw=uvw, freq=self.freq[Ilow:Ihigh], ms=residual_vis, wgt=weighti,
                                            npix_x=self.nx, npix_y=self.ny, pixsize_x=self.cell, pixsize_y=self.cell,
                                            epsilon=self.precision, nthreads=self.nthreads, do_wstacking=self.do_wstacking, verbosity=0)
        return residual

    def make_dirty(self):
        print("Making dirty")
        dirty = np.zeros((self.nband, self.nx, self.ny), dtype=self.real_type)
        xds = xds_from_table(self.table_name, group_cols=('FIELD_ID'), chunks={"row":-1, "chan": self.chan_chunks}, table_schema=self.schema)
        for ds in xds:
            if ds.FIELD_ID not in list(self.field):
                continue
            print("Processing field %i"%ds.FIELD_ID)
            data = getattr(ds, self.data_column).data
            weights = getattr(ds, self.weight_column).data
            uvw = ds.UVW.data.compute().astype(self.real_type)

            for i in range(self.nband):
                Ilow = self.freq_mapping[i]
                Ihigh = self.freq_mapping[i+1]
                weighti = weights.blocks[:, i].compute().astype(self.real_type)
                datai = data.blocks[:, i].compute().astype(self.complex_type)

                # TODO - load and apply interpolated fits beam patterns for field
                if weighti.any():
                    dirty[i] += ng.ms2dirty(uvw=uvw, freq=self.freq[Ilow:Ihigh], ms=weighti*datai, wgt=weighti,
                                            npix_x=self.nx, npix_y=self.ny, pixsize_x=self.cell, pixsize_y=self.cell,
                                            epsilon=self.precision, nthreads=self.nthreads, do_wstacking=self.do_wstacking, verbosity=0)
        return dirty

    def make_psf(self):
        print("Making PSF")
        psf_array = np.zeros((self.nband, 2*self.nx, 2*self.ny))
        xds = xds_from_table(self.table_name, group_cols=('FIELD_ID'), chunks={"row":-1, "chan": self.chan_chunks}, table_schema=self.schema)
        for ds in xds:
            if ds.FIELD_ID not in list(self.field):
                continue
            print("Processing field %i"%ds.FIELD_ID)
            weights = getattr(ds, self.weight_column).data
            uvw = ds.UVW.data.compute().astype(self.real_type)

            for i in range(self.nband):
                Ilow = self.freq_mapping[i]
                Ihigh = self.freq_mapping[i+1]
                weighti = weights.blocks[:, i].compute().astype(self.real_type)

                if weighti.any():
                    psf_array[i] += ng.ms2dirty(uvw=uvw, freq=self.freq[Ilow:Ihigh], ms=weighti.astype(self.complex_type), wgt=weighti,
                                                npix_x=2*self.nx, npix_y=2*self.ny, pixsize_x=self.cell, pixsize_y=self.cell,
                                                epsilon=self.precision, nthreads=self.nthreads, do_wstacking=self.do_wstacking)
        return psf_array


class PSF(object):
    def __init__(self, psf, nthreads, sigma0=1.0):
        self.nthreads = nthreads
        self.nband, nx_psf, ny_psf = psf.shape
        nx = nx_psf//2
        ny = ny_psf//2
        npad_x = (nx_psf - nx)//2
        npad_y = (ny_psf - ny)//2
        self.padding = ((0,0), (npad_x, npad_x), (npad_y, npad_y))
        self.ax = (1,2)
        self.unpad_x = slice(npad_x, -npad_x)
        self.unpad_y = slice(npad_y, -npad_y)
        self.lastsize = ny + np.sum(self.padding[-1])
        self.psf = psf
        psf_pad = iFs(psf, axes=self.ax)
        self.psfhat = r2c(psf_pad, axes=self.ax, forward=True, nthreads=nthreads, inorm=0)

        # get pre-conditioner kernel
        length_scale = 2.5/(2*np.sqrt(2*np.log(2)))
        K = make_kernel(nx_psf, ny_psf, sigma0, length_scale)
        K_pad = iFs(K, axes=self.ax)
        Khat = r2c(K_pad, axes=self.ax, forward=True, nthreads=nthreads, inorm=0)
        self.Kinv = 1.0/Khat

        self.kernel = self.psfhat + self.Kinv

        # mask edges
        self.mask = np.zeros((self.nband, nx_psf, ny_psf), dtype=np.float64)
        indx = slice(nx//2 + int(0.05*nx), 3*nx//2 - int(0.05*nx))
        indy = slice(ny//2 + int(0.05*ny), 3*ny//2 - int(0.05*ny))

        self.mask[:, indx, indy] = 1.0

    def convolve(self, x):
        xhat = iFs(np.pad(x, self.padding, mode='constant'), axes=self.ax)
        xhat = r2c(xhat, axes=self.ax, nthreads=self.nthreads, forward=True, inorm=0)
        xhat = c2r(xhat * self.psfhat, axes=self.ax, forward=False, lastsize=self.lastsize, inorm=2, nthreads=self.nthreads)
        return Fs(xhat, axes=self.ax)[:, self.unpad_x, self.unpad_y]

    def hess(self, x):
        xhat = iFs(np.pad(x, self.padding, mode='constant'), axes=self.ax)
        xhat = r2c(xhat, axes=self.ax, nthreads=self.nthreads, forward=True, inorm=0)
        xhat = c2r(xhat * self.kernel, axes=self.ax, forward=False, lastsize=self.lastsize, inorm=2, nthreads=self.nthreads)
        return Fs(xhat, axes=self.ax)[:, self.unpad_x, self.unpad_y]

    # def ihess(self, x):
    #     xhat = iFs(np.pad(x, self.padding, mode='constant'), axes=self.ax)
    #     xhat = r2c(xhat, axes=self.ax, nthreads=self.nthreads, forward=True, inorm=0)
    #     xhat = c2r(xhat / self.kernel, axes=self.ax, forward=False, lastsize=self.lastsize, inorm=2, nthreads=self.nthreads)
    #     return Fs(xhat, axes=self.ax)[:, self.unpad_x, self.unpad_y]

class Prior(object):
    def __init__(self, sigma0, nband, nx, ny, nthreads=8):
        self.nthreads = nthreads
        self.nx = nx
        self.ny = ny
        self.nv = nband
        nx_psf = 2*self.nx
        npad_x = (nx_psf - nx)//2
        ny_psf = 2*self.ny
        npad_y = (ny_psf - ny)//2
        self.padding = ((0,0), (npad_x, npad_x), (npad_y, npad_y))
        self.ax = (1,2)
        self.unpad_x = slice(npad_x, -npad_x)
        self.unpad_y = slice(npad_y, -npad_y)
        self.lastsize = ny + np.sum(self.padding[-1])

        # get kernel (always pixel coordinates)
        length_scale = 2.5/(2*np.sqrt(2*np.log(2)))
        self.K = make_kernel(nx_psf, ny_psf, sigma0, length_scale)
        K_pad = iFs(self.K, axes=self.ax)
        self.Khat = r2c(K_pad, axes=self.ax, forward=True, nthreads=nthreads, inorm=0)

    def dot(self, x):
        xhat = iFs(np.pad(x, self.padding, mode='constant'), axes=self.ax)
        xhat = r2c(xhat, axes=self.ax, nthreads=self.nthreads, forward=True, inorm=0)
        xhat = c2r(xhat * self.Khat, axes=self.ax, forward=False, lastsize=self.lastsize, inorm=2, nthreads=self.nthreads)
        res = Fs(xhat, axes=self.ax)[:, self.unpad_x, self.unpad_y]
        return res

    def idot(self, x):
        xhat = iFs(np.pad(x, self.padding, mode='constant'), axes=self.ax)
        xhat = r2c(xhat, axes=self.ax, nthreads=self.nthreads, forward=True, inorm=0)
        xhat = c2r(xhat / self.Khat, axes=self.ax, forward=False, lastsize=self.lastsize, inorm=2, nthreads=self.nthreads)
        res = Fs(xhat, axes=self.ax)[:, self.unpad_x, self.unpad_y]
        return res

class PSI(object):
    def __init__(self, nband, nx, ny,
                 nlevels=2,
                 bases=['self', 'db1', 'db2', 'db3', 'db4', 'db5']):
        """
        Sets up operators to move between wavelet coefficients
        in each basis and the image x.

        Parameters
        ----------
        nband - number of bands
        nx - number of pixels in x-dimension
        ny - number of pixels in y-dimension
        nlevels - The level of the decomposition. Default=2
        basis - List holding basis names.
                Default is delta + first 8 DB wavelets

        Returns
        =======
        Psi - list of operators performing coeff to image where
            each entry corresponds to one of the basis elements.
        Psi_t - list of operators performing image to coeff where
                each entry corresponds to one of the basis elements.
        """
        self.real_type = np.float64
        self.nband = nband
        self.nx = nx
        self.ny = ny
        self.nlevels = nlevels
        self.P = len(bases)
        self.sqrtP = np.sqrt(self.P)
        self.bases = bases
        self.nbasis = len(bases)

        # do a mock decomposition to get max coeff size
        x = np.random.randn(nx, ny)
        self.ntot = []
        self.iy = []
        self.sy = []
        for i, b in enumerate(bases):
            if b=='self':
                alpha = x.flatten()
                y, iy, sy = x.flatten(), 0, 0
            else:
                alpha = pywt.wavedecn(x, b, mode='zero', level=self.nlevels)
                # a, coeffs = wavedecn(x, b, 'zero', level=self.nlevels)
                y, iy, sy = pywt.ravel_coeffs(alpha)
            self.iy.append(iy)
            self.sy.append(sy)
            self.ntot.append(y.size)
        
        # get padding info
        self.nmax = np.asarray(self.ntot).max()
        self.padding = []
        for i in range(self.nbasis):
            self.padding.append(slice(0, self.ntot[i]))


    def dot(self, alpha):
        """
        Takes array of coefficients to image.
        The input does not have the form expected by pywt
        so we have to reshape it. Comes in as a flat vector
        arranged as

        [cAn, cHn, cVn, cDn, ..., cH1, cV1, cD1]

        where each entry is a flattened array and n denotes the
        level of the decomposition. This has to be restructured as

        [cAn, (cHm, cVn, cDn), ..., (cH1, cV1, cD1)]

        where entries are arrays with size defined by set_index_scheme.
        """
        x = np.zeros((self.nband, self.nx, self.ny), dtype=self.real_type)
        for b in range(self.nbasis):
            base = self.bases[b]
            for l in range(self.nband):
                a = alpha[b, l, self.padding[b]]
                if base == 'self':
                    wave = a.reshape(self.nx, self.ny)
                else:
                    a, coeffs = unravel_coeffs(a, self.iy[b], self.sy[b])
                    wave = pywt.waverecn(alpha_rec, base, mode='zero')
                    wave = waverecn(a, coeffs, base, mode='zero')

                    # return reconstructed image from coeff
                x[l] += wave / self.sqrtP
        return x

    def hdot(self, x):
        """
        This implements the adjoint of Psi_func i.e. image to coeffs
        """
        alpha = np.zeros((self.nbasis, self.nband, self.nmax))
        for b in range(self.nbasis):
            base = self.bases[b]
            for l in range(self.nband):
                if base == 'self':
                    # just pad image to have same shape as flattened wavelet coefficients
                    alpha[b, l] = np.pad(x[l].reshape(self.nx*self.ny)/self.sqrtP, (0, self.nmax-self.ntot[b]), mode='constant')
                else:
                    # decompose
                    # alphal = pywt.wavedecn(x[l], base, mode='zero', level=self.nlevels)
                    a, coeffs = wavedecn(x[l], base, 'zero', level=self.nlevels)
                    # ravel and pad
                    tmp, _, _ = ravel_coeffs(a, coeffs)
                    alpha[b, l] = np.pad(tmp/self.sqrtP, (0, self.nmax-self.ntot[b]), mode='constant')
        return alpha


import dask.array as da

def _dot_internal(alpha, bases, padding, iy, sy, sqrtP, nx, ny, real_type):
    nbasis, nband, _ = alpha.shape
    # reduction over basis done externally since chunked
    x = np.zeros((nbasis, nband, nx, ny), dtype=real_type)
    for b in range(nbasis):
            base = bases[b]
            for l in range(nband):
                a = alpha[b, l, padding[b]]
                if base == 'self':
                    wave = a.reshape(nx, ny)
                else:
                    a, coeffs = unravel_coeffs(a, iy[b], sy[b], output_format='wavedecn')
                    # a, coeffs = alpha_rec[0], alpha_rec[1::]
                    wave = waverecn(a, coeffs, base, mode='zero')
                    # wave = pywt.waverecn(alpha_rec, base, mode='zero')

                x[l] += wave / sqrtP
    return x

def _dot_internal_wrapper(alpha, bases, padding, iy, sy, sqrtP, nx, ny, real_type):
    return _dot_internal(alpha[0], bases, padding, iy, sy, sqrtP, nx, ny, real_type)

def _hdot_internal(x, bases, ntot, nmax, nlevels, sqrtP, nx, ny, real_type):
    nband = x.shape[0]
    nbasis = len(bases)
    alpha = np.zeros((nbasis, nband, nmax), dtype=real_type)
    for b in range(nbasis):
            base = bases[b]
            for l in range(nband):
                if base == 'self':
                    # ravel and pad
                    alpha[b, l] = np.pad(x[l].reshape(nx*ny)/sqrtP, (0, nmax-ntot[b]), mode='constant')
                else:
                    # decompose
                    # alphal = pywt.wavedecn(x[l], base, mode='zero', level=nlevels)
                    a, coeffs = wavedecn(x[l], base, 'zero', level=nlevels)
                    # alphal = [a, coeffs]
                    # ravel and pad
                    tmp, _, _ = ravel_coeffs(a, coeffs)
                    alpha[b, l] = np.pad(tmp/sqrtP, (0, nmax-ntot[b]), mode='constant')

    return alpha

def _hdot_internal_wrapper(x, bases, ntot, nmax, nlevels, sqrtP, nx, ny, real_type):
    return _hdot_internal(x[0][0], bases, ntot, nmax, nlevels, sqrtP, nx, ny, real_type)

class DaskPSI(PSI):
    def __init__(self, nband, nx, ny,
                 nlevels=2,
                 bases=['self', 'db1', 'db2', 'db3', 'db4', 'db5'],
                 nthreads=8):
        PSI.__init__(self, nband, nx, ny, nlevels=nlevels,
                     bases=bases)
        # required to chunk over basis
        bases = np.array(self.bases, dtype=object)
        self.bases = da.from_array(bases, chunks=1)
        padding = np.array(self.padding, dtype=object)
        self.padding = da.from_array(padding, chunks=1)
        iy = np.array(self.iy, dtype=object)
        self.iy = da.from_array(iy, chunks=1)
        sy = np.array(self.sy, dtype=object)
        self.sy = da.from_array(sy, chunks=1)
        ntot = np.array(self.ntot, dtype=object)
        self.ntot = da.from_array(ntot, chunks=1)
        
    def dot(self, alpha):
        alpha_dask = da.from_array(alpha, chunks=(1, 1, self.nmax))
        x = da.blockwise(_dot_internal_wrapper, ("basis", "band", "nx", "ny"),
                         alpha_dask, ("basis", "band", "ntot"),
                         self.bases, ("basis",),
                         self.padding, ("basis",),
                         self.iy, ("basis",),
                         self.sy, ("basis",),
                         self.sqrtP, None,
                         self.nx, None,
                         self.ny, None,
                         self.real_type, None,
                         new_axes={"nx": self.nx, "ny": self.ny},
                         dtype=self.real_type,
                         align_arrays=False)

        return x.sum(axis=0).compute()

    def hdot(self, x):
        xdask = da.from_array(x, chunks=(1, self.nx, self.ny))
        alpha = da.blockwise(_hdot_internal_wrapper, ("basis", "band", "nmax"),
                             xdask, ("band", "nx", "ny"),
                             self.bases, ("basis", ),
                             self.ntot,("basis", ),
                             self.nmax, None,
                             self.nlevels, None,
                             self.sqrtP, None,
                             self.nx, None,
                             self.ny, None,
                             self.real_type, None,
                             new_axes={"nmax": self.nmax},
                             dtype=self.real_type,
                             align_arrays=False)

        return alpha.compute()