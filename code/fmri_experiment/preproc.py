# Author: Jean-Remi King
#
# Adapted from MNE-Python, Dipy and Pypreprocess (BSL Licence)

from copy import deepcopy

import nibabel as nib
import numpy as np
import pandas as pd
from dipy.align import imaffine, imwarp, metrics, transforms, vector_fields
from dipy.align.reslice import reslice
from h5io import read_hdf5, write_hdf5
from nilearn import image

from externals.pypreprocess.coreg import Coregister
from externals.pypreprocess.realign import MRIMotionCorrection
from externals.pypreprocess.slice_timing import fMRISTC


def do_preproc_fmri(anat, func, fsaverage):
    params = dict()
    stages = dict()

    # Slice-Timing Correction
    stc = fMRISTC(verbose=False)
    func = stc.fit(raw_data=func).transform(func)
    params['slice_timing'] = stc.kernel_

    stages['stc'] = deepcopy(func)

    # Motion correction
    mc = MRIMotionCorrection(n_sessions=1, verbose=False)
    mc_output = mc.fit([func, ]).transform(concat=True)
    func = mc_output['realigned_images'][0]
    params['motion'] = mc_output['realignment_parameters'][0]

    stages['mc'] = deepcopy(func)

    # Coregister to anatomy
    coreg = Coregister(verbose=False)
    coreg = coreg.fit(target=anat, source=func)
    func = coreg.transform(func)
    params['coreg'] = coreg.params_

    stages['coreg'] = deepcopy(func)

    # Concatenate
    func = image.concat_imgs(func)

    # Fit Morphing to FSaverage
    morpher = SDR(zooms=3.5,
                  niter_affine=(100, 50, 10),
                  niter_sdr=(100, 50, 10))
    morpher.fit(nib.load(anat), nib.load(fsaverage))
    attrs = ('affine_idx_in_', 'comp_', 'backward_',
             'mri_from_affine_', 'mri_to_affine_', 'shape_',
             'zooms', 'niter_affine', 'niter_sdr')
    params['morph'] = dict((k, getattr(morpher, k)) for k in attrs)

    return func, stages, params


class SDR():
    "Symmetric Diffeomorphic Registration"
    def __init__(self, zooms=2.,
                 niter_affine=(100, 100, 100),
                 niter_sdr=(20, 20, 20)):

        # use voxel size of mri_from
        zooms = np.atleast_1d(zooms).astype(float)
        if zooms.shape == (1,):
            zooms = np.repeat(zooms, 3)
        if zooms.shape != (3,):
            raise ValueError('zooms must be None, a singleton, or have shape '
                             '(3,), got shape %s' % (zooms.shape,))
        self.zooms = zooms
        self.niter_affine = niter_affine
        self.niter_sdr = niter_sdr

    def _reslice(self, mri):
        mri_res, mri_res_affine = reslice(
            mri.get_data(), mri.affine, mri.header.get_zooms()[:3], self.zooms)
        mri_res = nib.Nifti1Image(mri_res, mri_res_affine)
        return mri_res

    def _normalize(self, mri):
        mri = np.array(mri.dataobj, float)
        mri /= mri.max()
        return mri

    def fit(self, mri_from, mri_to):

        # Reslice mri_from
        mri_from = self._reslice(mri_from)
        mri_to = self._reslice(mri_to)

        # Normalize
        self.mri_from_affine_ = mri_from.affine
        self.mri_to_affine_ = mri_to.affine
        mri_from = self._normalize(mri_from)
        mri_to = self._normalize(mri_to)

        # Set up Affine Registration
        affreg = imaffine.AffineRegistration(
            metric=imaffine.MutualInformationMetric(nbins=32),
            level_iters=list(self.niter_affine),
            sigmas=[3.0, 1.0, 0.0],
            factors=[4, 2, 1])

        # Translation
        c_of_mass = imaffine.transform_centers_of_mass(
            mri_to, self.mri_to_affine_, mri_from, self.mri_from_affine_)
        translation = affreg.optimize(
            mri_to, mri_from, transforms.TranslationTransform3D(), None,
            self.mri_to_affine_, self.mri_from_affine_,
            starting_affine=c_of_mass.affine)

        # Rigid body transform (translation + rotation)
        rigid = affreg.optimize(
            mri_to, mri_from, transforms.RigidTransform3D(), None,
            self.mri_to_affine_, self.mri_from_affine_,
            starting_affine=translation.affine)

        # Affine transform (translation + rotation + scaling)
        pre_affine = affreg.optimize(
            mri_to, mri_from, transforms.AffineTransform3D(), None,
            self.mri_to_affine_, self.mri_from_affine_,
            starting_affine=rigid.affine)

        # Compute mapping
        sdr = imwarp.SymmetricDiffeomorphicRegistration(
            metrics.CCMetric(3), list(self.niter_sdr))
        sdr = sdr.optimize(mri_to, pre_affine.transform(mri_from))

        # Only store useful elements for easier storage
        self.comp_ = imwarp.npl.inv(pre_affine.codomain_grid2world) @ \
            pre_affine.affine @ pre_affine.domain_grid2world
        self.shape_ = np.array(sdr.codomain_shape, np.int32)
        self.affine_idx_in_ = np.array(sdr.disp_world2grid, np.float64)
        self.backward_ = sdr.backward.astype(float)

        self.tmp = [pre_affine, sdr]

        return self

    def transform(self, mri):
        # Make sure same coordinates
        if not np.array_equal(mri.affine, self.mri_from_affine_):
            mri = image.resample_img(mri, self.mri_from_affine_)

        # Reslice
        mri_res = self._reslice(mri)
        mri_res = np.array(mri_res.dataobj, float)

        # Affine transform
        # mri_affine = self.pre_affine_.transform(mri_res)
        mri_res = mri_res.astype(np.float64)
        mri_affine = vector_fields.transform_3d_affine(
            mri_res, self.shape_, self.comp_)

        # SDR transform
        # mri_sdr = self.sdr_morph_.transform(mri_affine)
        mri_sdr = imwarp.vfu.warp_3d(mri_affine.astype(
            float), self.backward_, self.affine_idx_in_)

        return nib.Nifti1Image(mri_sdr, self.mri_to_affine_)

    def save(self, fname):
        attrs = ('affine_idx_in_', 'comp_', 'backward_',
                 'mri_from_affine_', 'mri_to_affine_', 'shape_',
                 'zooms', 'niter_affine', 'niter_sdr')
        out = dict((attr, getattr(self, attr)) for attr in attrs)
        write_hdf5(fname, out, overwrite=True)
        return self

    def load(self, fname):
        data = read_hdf5(fname)
        for attr in ('affine_idx_in_', 'comp_', 'backward_',
                     'mri_from_affine_', 'mri_to_affine_', 'shape_',
                     'zooms', 'niter_affine', 'niter_sdr'):
            setattr(self, attr, data[attr])
        return self


def read_mri_events(event_fname):
    # Read MRI events
    events = pd.read_csv(event_fname, sep='\t')

    # Add context: sentence or word list?
    contexts = dict(WOORDEN='word_list', ZINNEN='sentence')
    for key, value in contexts.items():
        sel = events.value.str.contains(key)
        events.loc[sel, 'context'] = value
        events.loc[sel, 'condition'] = value

    # Clean up MRI event mess
    sel = ~events.context.isna()
    start = 0
    context = 'init'
    for idx, row in events.loc[sel].iterrows():
        events.loc[start:idx, 'context'] = context
        start = idx
        context = row.context
    events.loc[start:, 'context'] = context

    # Add event condition: word, blank, inter stimulus interval etc
    conditions = (('50', 'pulse'), ('blank', 'blank'), ('ISI', 'isi'))
    for key, value in conditions:
        sel = events.value == key
        events.loc[sel, 'condition'] = value

    events.loc[events.value.str.contains('FIX '), 'condition'] = 'fix'

    # Extract words from file
    sel = events.condition.isna()
    words = events.loc[sel, 'value'].apply(lambda s: s.strip('0123456789 '))
    events.loc[sel, 'word'] = words

    # Remove empty words
    sel = (events.word.astype(str).apply(len) == 0) & (events.condition.isna())
    events.loc[sel, 'word'] = pd.np.nan
    events.loc[sel, 'condition'] = 'blank'
    events.loc[~events.word.isna(), 'condition'] = 'word'

    return events
