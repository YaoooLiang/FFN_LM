import itertools
import sys
from scipy.special import expit
from scipy.special import logit
import torch
import six
import numpy as np
from scipy import ndimage
import random
import skimage.feature
import logging
import weakref
from collections import namedtuple
from collections import deque
import time
from torch.autograd import Variable
import cv2
import skimage
from scipy.stats import stats

MAX_SELF_CONSISTENT_ITERS = 32
HALT_SILENT = 0
PRINT_HALTS = 1
HALT_VERBOSE = 2

OriginInfo = namedtuple('OriginInfo', ['start_zyx', 'iters', 'walltime_sec'])
HaltInfo = namedtuple('HaltInfo', ['is_halt', 'extra_fetches'])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_seed(shape, pad=0.05, seed=0.95):
    """创建种子"""
    seed_array = np.full(list(shape), pad, dtype=np.float32)
    idx = tuple([slice(None)] + list(np.array(shape) // 2))
    seed_array[idx] = seed
    return seed_array


def fixed_offsets(seed, fov_moves, threshold=0.9):
    """offset偏移."""
    for off in itertools.chain([(0, 0, 0)], fov_moves):
        is_valid_move = seed[0,
                             seed.shape[1] // 2 + off[2],
                             seed.shape[2] // 2 + off[1],
                             seed.shape[3] // 2 + off[0]
                        ] >= logit(np.array(threshold))

        if not is_valid_move:
            continue

        yield off


def center_crop_and_pad(data, coor, target_shape):
    """根据中心坐标 crop patch"""
    target_shape = np.array(target_shape)

    start = coor - target_shape // 2
    end = start + target_shape

    #assert np.all(start >= 0)

    selector = [slice(s, e) for s, e in zip(start, end)]
    cropped = data[tuple(selector)]

    if target_shape is not None:

        if len(cropped.shape) > 3:
            target_shape = np.array(target_shape)
            delta = target_shape - cropped.shape[:-1]
            pre = delta // 2
            post = delta - delta // 2

            paddings = []  # no padding for batch
            paddings.extend(zip(pre, post))
            paddings.append((0, 0))
            cropped = np.pad(cropped, paddings, mode='constant')
        else:
            target_shape = np.array(target_shape)
            delta = target_shape - cropped.shape
            pre = delta // 2
            post = delta - delta // 2

            paddings = []  # no padding for batch
            paddings.extend(zip(pre, post))
            cropped = np.pad(cropped, paddings, mode='constant')

    return cropped


def crop_and_pad(data, offset, crop_shape, target_shape=None):
    """根据offset crop patch"""
    # Spatial dimensions only. All vars in zyx.
    shape = np.array(data.shape[1:])
    crop_shape = np.array(crop_shape)
    offset = np.array(offset[::-1])

    start = shape // 2 - crop_shape // 2 + offset
    end = start + crop_shape

    assert np.all(start >= 0)

    selector = [slice(s, e) for s, e in zip(start, end)]
    selector = tuple([slice(None)] + selector)
    cropped = data[selector]

    if target_shape is not None:
        target_shape = np.array(target_shape)
        delta = target_shape - crop_shape
        pre = delta // 2
        post = delta - delta // 2

        paddings = [(0, 0)]  # no padding for batch
        paddings.extend(zip(pre, post))
        paddings.append((0, 0))  # no padding for channels

        cropped = np.pad(cropped, paddings, mode='constant')

    return cropped


def get_example(loader, shape, get_offsets):
    while True:
        iteration, (image, targets, seed, coor) = next(enumerate(loader))
        seed = seed.numpy().copy()
        for off in get_offsets(seed):
            predicted = crop_and_pad(seed, off, shape)[np.newaxis, ...]
            patches = crop_and_pad(image.squeeze(), off, shape).unsqueeze(0)
            labels = crop_and_pad(targets, off, shape).unsqueeze(0)
            offset = off
            assert predicted.base is seed
            yield predicted, patches, labels, offset


def get_batch(loader, batch_size, shape, get_offsets):
    def _batch(iterable):
        for batch_vals in iterable:
            yield zip(*batch_vals)

    for seeds, patches, labels, offsets in _batch(six.moves.zip(
            *[get_example(loader, shape, get_offsets) for _
              in range(batch_size)])):

        batched_seeds = np.concatenate(seeds)

        yield (batched_seeds, torch.cat(patches, dim=0).float(), \
               torch.cat(labels, dim=0).float(), offsets)

        for i in range(batch_size):
            seeds[i][:] = batched_seeds[i, ...]


def update_seed(updated, seed, model, pos):
    start = pos - model.input_size // 2
    end = start + model.input_size
    assert np.all(start >= 0)

    selector = [slice(s, e) for s, e in zip(start, end)]
    seed[selector] = np.squeeze(updated)


def no_halt(verbosity=HALT_SILENT, log_function=logging.info):
    """Dummy HaltInfo."""

    def _halt_signaler(*unused_args, **unused_kwargs):
        return False

    def _halt_signaler_verbose(fetches, pos, **unused_kwargs):
        log_function('%s, %s' % (pos, fetches))
        return False

    if verbosity == HALT_VERBOSE:
        return HaltInfo(_halt_signaler_verbose, [])
    else:
        return HaltInfo(_halt_signaler, [])


def self_prediction_halt(
        threshold, orig_threshold=None, verbosity=HALT_SILENT,
        log_function=logging.info):
    """HaltInfo based on FFN self-predictions."""

    def _halt_signaler(fetches, pos, orig_pos, counters, **unused_kwargs):
        """Returns true if FFN prediction should be halted."""
        if pos == orig_pos and orig_threshold is not None:
            t = orig_threshold
        else:
            t = threshold

        # [0] is by convention the total incorrect proportion prediction.
        halt = fetches['self_prediction'][0] > t

        if halt:
            counters['halts'].Increment()

        if verbosity == HALT_VERBOSE or (
                halt and verbosity == PRINT_HALTS):
            log_function('%s, %s' % (pos, fetches))

        return halt

    # Add self_prediction to the extra_fetches.
    return HaltInfo(_halt_signaler, ['self_prediction'])


class BaseSeedPolicy(object):
    """Base class for seed policies."""

    def __init__(self, canvas, **kwargs):
        """Initializes the policy.

        Args:
          canvas: inference Canvas object; simple policies use this to access
              basic geometry information such as the shape of the subvolume;
              more complex policies can access the raw image data, etc.
          **kwargs: other keyword arguments
        """
        del kwargs
        # TODO(mjanusz): Remove circular reference between Canvas and seed policies.
        self.canvas = weakref.proxy(canvas)
        self.coords = None
        self.idx = 0

        self._init_coords()

    def _init_coords(self):
        raise NotImplementedError()

    def __iter__(self):
        return self

    def __next__(self):
        """Returns the next seed point as (z, y, x).

        Does initial filtering of seed points to exclude locations that are
        too close to the image border.

        Returns:
          (z, y, x) tuples.

        Raises:
          StopIteration when the seeds are exhausted.
        """
        if self.coords is None:
            self._init_coords()

        while self.idx < self.coords.shape[0]:
            curr = self.coords[self.idx, :]
            self.idx += 1

            # TODO(mjanusz): Get rid of this.
            # Do early filtering of clearly invalid locations (too close to image
            # borders) as late filtering might be expensive.
            if (np.all(curr - self.canvas.margin >= 0) and
                    np.all(curr + self.canvas.margin < self.canvas.shape)):
                yield tuple(curr)  # z, y, x

        raise StopIteration()

    def next(self):
        return self.__next__()

    def get_state(self):
        return self.coords, self.idx

    def set_state(self, state):
        self.coords, self.idx = state


def quantize_probability(prob):
    """Quantizes a probability map into a byte array."""
    ret = np.digitize(prob, np.linspace(0.0, 1.0, 255))

    # Digitize never uses the 0-th bucket.
    ret[np.isnan(prob)] = 0
    return ret.astype(np.uint8)


def get_scored_move_offsets(deltas, prob_map, threshold=0.9):
    """Looks for potential moves for a FFN.

    The possible moves are determined by extracting probability map values
    corresponding to cuboid faces at +/- deltas, and considering the highest
    probability value for every face.

    Args:
      deltas: (z,y,x) tuple of base move offsets for the 3 axes
      prob_map: current probability map as a (z,y,x) numpy array
      threshold: minimum score required at the new FoV center for a move to be
          considered valid

    Yields:
      tuples of:
        score (probability at the new FoV center),
        position offset tuple (z,y,x) relative to center of prob_map

      The order of the returned tuples is arbitrary and should not be depended
      upon. In particular, the tuples are not necessarily sorted by score.
    """
    center = np.array(prob_map.shape) // 2
    assert center.size == 3
    # Selects a working subvolume no more than +/- delta away from the current
    # center point.
    subvol_sel = [slice(c - dx, c + dx + 1) for c, dx
                  in zip(center, deltas)]

    done = set()
    for axis, axis_delta in enumerate(deltas):
        if axis_delta == 0:
            continue
        for axis_offset in (-axis_delta, axis_delta):
            # Move exactly by the delta along the current axis, and select the face
            # of the subvolume orthogonal to the current axis.
            face_sel = subvol_sel[:]
            face_sel[axis] = axis_offset + center[axis]
            face_prob = prob_map[tuple(face_sel)]
            shape = face_prob.shape

            # Find voxel with maximum activation.
            face_pos = np.unravel_index(face_prob.argmax(), shape)
            score = face_prob[face_pos]

            # Only move if activation crosses threshold.
            if score < threshold:
                continue

            # Convert within-face position to be relative vs the center of the face.
            relative_pos = [face_pos[0] - shape[0] // 2, face_pos[1] - shape[1] // 2]
            relative_pos.insert(axis, axis_offset)

            ret = (score, tuple(relative_pos))

            if ret not in done:
                done.add(ret)
                yield ret

    if deltas[0] % 2 == 1:
        deltas_even = deltas + 1
    else:
        deltas_even = deltas
    deltas_half = deltas_even / 2
    deltas_half = [int(deltas_half[0]), int(deltas_half[1]), int(deltas_half[2])]
    subvol_sel_half = [slice(c - dx, c + dx + 1) for c, dx
                       in zip(center, deltas_half)]

    for axis, axis_delta in enumerate(deltas_half):
        if axis_delta == 0:
            continue
        for axis_offset in (-axis_delta, axis_delta):
            # Move exactly by the delta along the current axis, and select the face
            # of the subvolume orthogonal to the current axis.
            face_sel = subvol_sel_half[:]
            face_sel[axis] = axis_offset + center[axis]
            face_prob = prob_map[tuple(face_sel)]
            shape = face_prob.shape

            # Find voxel with maximum activation.
            face_pos = np.unravel_index(face_prob.argmax(), shape)
            score = face_prob[face_pos]

            # Only move if activation crosses threshold.
            if score < threshold:
                continue

            # Convert within-face position to be relative vs the center of the face.
            relative_pos = [face_pos[0] - shape[0] // 2, face_pos[1] - shape[1] // 2]
            relative_pos.insert(axis, axis_offset)

            ret = (score, tuple(relative_pos))

            if ret not in done:
                done.add(ret)
                yield ret

    if deltas_half[0] % 2 == 1:
        deltas_half_even = np.array(deltas_half) + 1
    else:
        deltas_half_even = deltas_half
    deltas_half_even = np.array(deltas_half_even)
    deltas_q = deltas_half_even / 2
    deltas_q = [int(deltas_q[0]), int(deltas_q[1]), int(deltas_q[2])]
    subvol_sel_half = [slice(c - dx, c + dx + 1) for c, dx
                       in zip(center, deltas_q)]

    for axis, axis_delta in enumerate(deltas_q):
        if axis_delta == 0:
            continue
        for axis_offset in (-axis_delta, axis_delta):
            # Move exactly by the delta along the current axis, and select the face
            # of the subvolume orthogonal to the current axis.
            face_sel = subvol_sel_half[:]
            face_sel[axis] = axis_offset + center[axis]
            face_prob = prob_map[tuple(face_sel)]
            shape = face_prob.shape

            # Find voxel with maximum activation.
            face_pos = np.unravel_index(face_prob.argmax(), shape)
            score = face_prob[face_pos]

            # Only move if activation crosses threshold.
            if score < threshold:
                continue

            # Convert within-face position to be relative vs the center of the face.
            relative_pos = [face_pos[0] - shape[0] // 2, face_pos[1] - shape[1] // 2]
            relative_pos.insert(axis, axis_offset)

            ret = (score, tuple(relative_pos))

            if ret not in done:
                done.add(ret)
                yield ret


class PolicyPeaks(BaseSeedPolicy):
    """Attempts to find points away from edges in the image.

    Runs a 3d Sobel filter to detect edges in the raw data, followed
    by a distance transform and peak finding to identify seed points.
    """

    def _init_coords(self):
        logging.info('peaks: starting')

        # Edge detection.
        gray = np.array([cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in self.canvas.images])
        edges = ndimage.generic_gradient_magnitude(
            gray.astype(np.float32),
            ndimage.sobel)

        # Adaptive thresholding.
        sigma = 49.0 / 6.0
        thresh_image = np.zeros(edges.shape, dtype=np.float32)
        ndimage.gaussian_filter(edges, sigma, output=thresh_image, mode='reflect')
        filt_edges = edges > thresh_image

        del edges, thresh_image

        # # This prevents a border effect where the large amount of masked area
        # # screws up the distance transform below.
        # if (self.canvas.restrictor is not None and
        #         self.canvas.restrictor.mask is not None):
        #     filt_edges[self.canvas.restrictor.mask] = 1

        logging.info('peaks: filtering done')
        dt = ndimage.distance_transform_edt(1 - filt_edges).astype(np.float32)
        logging.info('peaks: edt done')

        # Use a specifc seed for the noise so that results are reproducible
        # regardless of what happens before the policy is called.
        state = np.random.get_state()
        np.random.seed(42)
        idxs = skimage.feature.peak_local_max(
            dt + np.random.random(dt.shape) * 1e-4,
            indices=True, min_distance=1, threshold_abs=0, threshold_rel=0)
        np.random.set_state(state)

        # After skimage upgrade to 0.13.0 peak_local_max returns peaks in
        # descending order, versus ascending order previously.  Sort ascending to
        # maintain historic behavior.
        idxs = np.array(sorted((z, y, x) for z, y, x in idxs))

        logging.info('peaks: found %d local maxima', idxs.shape[0])
        self.coords = idxs


class BaseMovementPolicy(object):
    """Base class for movement policy queues.

    The principal usage is to initialize once with the policy's parameters and
    set up a queue for candidate positions. From this queue candidates can be
    iteratively consumed and the scores should be updated in the FFN
    segmentation loop.
    """

    def __init__(self, canvas, scored_coords, deltas):
        """Initializes the policy.

        Args:
          canvas: Canvas object for FFN inference
          scored_coords: mutable container of tuples (score, zyx coord)
          deltas: step sizes as (z,y,x)
        """
        # TODO(mjanusz): Remove circular reference between Canvas and seed policies.
        self.canvas = weakref.proxy(canvas)
        self.scored_coords = scored_coords
        self.deltas = np.array(deltas)

    def __len__(self):
        return len(self.scored_coords)

    def __iter__(self):
        return self

    def next(self):
        raise StopIteration()

    def append(self, item):
        self.scored_coords.append(item)

    def update(self, prob_map, position):
        """Updates the state after an FFN inference call.

        Args:
          prob_map: object probability map returned by the FFN (in logit space)
          position: postiion of the center of the FoV where inference was performed
              (z, y, x)
        """
        raise NotImplementedError()

    def get_state(self):
        """Returns the state of this policy as a pickable Python object."""
        raise NotImplementedError()

    def restore_state(self, state):
        raise NotImplementedError()

    def reset_state(self, start_pos):
        """Resets the policy.

        Args:
          start_pos: starting position of the current object as z, y, x
        """
        raise NotImplementedError()


class FaceMaxMovementPolicy(BaseMovementPolicy):
    """Selects candidates from maxima on prediction cuboid faces."""

    def __init__(self, canvas, deltas=(4, 8, 8), score_threshold=0.9):
        self.done_rounded_coords = set()
        self.score_threshold = score_threshold
        self._start_pos = None
        super(FaceMaxMovementPolicy, self).__init__(canvas, deque([]), deltas)

    def reset_state(self, start_pos):
        self.scored_coords = deque([])
        self.done_rounded_coords = set()
        self._start_pos = start_pos

    def get_state(self):
        return [(self.scored_coords, self.done_rounded_coords)]

    def restore_state(self, state):
        self.scored_coords, self.done_rounded_coords = state[0]

    def __next__(self):
        """Pops positions from queue until a valid one is found and returns it."""
        while self.scored_coords:
            _, coord = self.scored_coords.popleft()
            coord = tuple(coord)
            if self.quantize_pos(coord) in self.done_rounded_coords:
                continue
            if self.canvas.is_valid_pos(coord):
                break
        else:  # Else goes with while, not with if!
            raise StopIteration()

        return tuple(coord)

    def next(self):
        return self.__next__()

    def quantize_pos(self, pos):
        """Quantizes the positions symmetrically to a grid downsampled by deltas."""
        # Compute offset relative to the origin of the current segment and
        # shift by half delta size. This ensures that all directions are treated
        # approximately symmetrically -- i.e. the origin point lies in the middle of
        # a cell of the quantized lattice, as opposed to a corner of that cell.
        rel_pos = (np.array(pos) - self._start_pos)
        coord = (rel_pos + self.deltas // 2) // np.maximum(self.deltas, 1)
        return tuple(coord)

    def update(self, prob_map, position):
        """Adds movements to queue for the cuboid face maxima of ``prob_map``."""
        qpos = self.quantize_pos(position)
        self.done_rounded_coords.add(qpos)

        scored_coords = get_scored_move_offsets(self.deltas, prob_map,
                                                threshold=self.score_threshold)
        scored_coords = sorted(scored_coords, reverse=True)
        for score, rel_coord in scored_coords:
            # convert to whole cube coordinates
            coord = [rel_coord[i] + position[i] for i in range(3)]
            self.scored_coords.append((score, coord))


class Canvas(object):

    def __init__(self, model, images, seed_list, size, delta, seg_thr, mov_thr, act_thr):
        self.model = model
        self.images = images
        self.shape = images.shape[:-1]
        self.input_size = np.array(size)
        self.margin = np.array(size) // 2
        self.seg_thr = logit(seg_thr)
        self.mov_thr = logit(mov_thr)
        self.act_thr = logit(act_thr)

        self.segmented_mask = np.zeros(self.shape, dtype=np.int32)
        self.done_mask = np.zeros(self.shape, dtype=bool)

        self.segmentation = np.zeros(self.shape, dtype=np.int32)
        self.seed = np.zeros(self.shape, dtype=np.float32)
        self.seg_prob = np.zeros(self.shape, dtype=np.uint8)
        self.seg_prob_i = np.zeros(self.shape, dtype=np.uint8)
        self.target_dic = {}

        self.seed_policy = None
        self.seed_list = seed_list
        self.max_id = 0
        # Maps of segment id -> ..
        self.origins = {}  # seed location
        self.overlaps = {}  # (ids, number overlapping voxels)

        self.movement_policy = FaceMaxMovementPolicy(self, deltas=delta, score_threshold=self.mov_thr)

        self.reset_state((0, 0, 0))

    def init_seed(self, pos):
        """Reinitiailizes the object mask with a seed.

        Args:
          pos: position at which to place the seed (z, y, x)
        """
        self.seed[...] = np.nan
        self.seed[pos] = self.act_thr

    def reset_state(self, start_pos):
        # Resetting the movement_policy is currently necessary to update the
        # policy's bitmask for whether a position is already segmented (the
        # canvas updates the segmented mask only between calls to segment_at
        # and therefore the policy does not update this mask for every call.).
        self.movement_policy.reset_state(start_pos)
        self.history = []
        self.history_deleted = []

        self._min_pos = np.array(start_pos)
        self._max_pos = np.array(start_pos)

    def is_valid_pos(self, pos, ignore_move_threshold=False):
        """Returns True if segmentation should be attempted at the given position.

        Args:
          pos: position to check as (z, y, x)
          ignore_move_threshold: (boolean) when starting a new segment at pos the
              move threshold can and must be ignored.

        Returns:
          Boolean indicating whether to run FFN inference at the given position.
        """

        if not ignore_move_threshold:
            if self.seed[pos] < self.mov_thr:
                return False

        # Not enough image context?
        np_pos = np.array(pos)
        low = np_pos - self.margin
        high = np_pos + self.margin

        if np.any(low < 0) or np.any(high >= self.shape):
            return False

        # Location already segmented?
        if self.segmentation[pos] > 0:
            return False

        return True

    def predict(self, pos):
        """Runs a single step of FFN prediction.
        """
        # Top-left corner of the FoV.
        start = np.array(pos) - self.margin
        end = start + self.input_size

        assert np.all(start >= 0)

        # selector = [slice(s, e) for s, e in zip(start, end)]
        images = self.images[start[0]:end[0], start[1]:end[1], start[2]:end[2], :].transpose(3, 0, 1, 2)
        seeds = self.seed[start[0]:end[0], start[1]:end[1], start[2]:end[2]].copy()
        init_prediction = np.isnan(seeds)
        seeds[init_prediction] = np.float32(logit(0.05))
        images = torch.from_numpy(images).float().unsqueeze(0)
        seeds = torch.from_numpy(seeds).float().unsqueeze(0).unsqueeze(0)

        # slice = seeds[:, :, seeds.shape[2] // 2, :, :].sigmoid()
        # seeds[:, :, seeds.shape[2] // 2, :, :] = slice

        input_data = torch.cat([images, seeds], dim=1)
        input_data = Variable(input_data.cuda())

        logits = self.model(input_data)
        updated = (seeds.cuda() + logits).detach().cpu().numpy()
        # update_seed(updated, self.seed, self.model, pos)

        prob = expit(updated)
        return np.squeeze(prob), np.squeeze(updated)

    def update_at(self, pos):
        """Updates object mask prediction at a specific position.
        """
        global old_err
        off = self.input_size // 2  # zyx

        start = np.array(pos) - off
        start_cent = np.array(pos) - 1
        end = start + self.input_size
        end_cent = start_cent + 3
        # print(start_cent)
        # print(end_cent)
        start_cent[0] = 0
        end_cent[0] = 590
        sel = [slice(s, e) for s, e in zip(start, end)]

        logit_seed = np.array(self.seed[tuple(sel)])
        init_prediction = np.isnan(logit_seed)
        logit_seed[init_prediction] = np.float32(logit(0.05))

        prob_seed = expit(logit_seed)
        for _ in range(MAX_SELF_CONSISTENT_ITERS):
            """网络inference"""
            prob, logits = self.predict(pos)
            break

        """更新seed"""
        sel = [slice(s, e) for s, e in zip(start, end)]
        sel_cent = [slice(s, e) for s, e in zip(start_cent, end_cent)]
        # Bias towards oversegmentation by making it impossible to reverse
        # disconnectedness predictions in the course of inference.
        th_max = logit(0.5)
        old_seed = self.seed[tuple(sel)]

        if np.mean(logits >= self.mov_thr) > 0:
            # Because (x > NaN) is always False, this mask excludes positions that
            # were previously uninitialized (i.e. set to NaN in old_seed).
            try:
                old_err = np.seterr(invalid='ignore')
                mask = ((old_seed < th_max) & (logits > old_seed))
            finally:
                np.seterr(**old_err)
            logits[mask] = old_seed[mask]

        # Update working space.
        self.seed[tuple(sel)] = logits

        return logits, sel, sel_cent

    def segment_at(self, start_pos, id):

        try:
            if not self.is_valid_pos(start_pos, ignore_move_threshold=True):
                return
            if self.segmented_mask[start_pos] != 0:
                return
            if self.done_mask[start_pos] != 0:
                return

            self.segmentation = np.zeros(self.shape, dtype=np.int32)
            self.seed = np.zeros(self.shape, dtype=np.float32)
            self.seg_prob = np.zeros(self.shape, dtype=np.uint8)
            self.seg_prob_i = np.zeros(self.shape, dtype=np.uint8)

            self.init_seed(start_pos)
            num_iters = 0
            self.reset_state(start_pos)

            if not self.movement_policy:
                # Add first element with arbitrary priority 1 (it will be consumed
                # right away anyway).
                item = (self.movement_policy.score_threshold * 2, start_pos)
                self.movement_policy.append(item)

            sel_i_s = [()]
            sel_cent_s = [()]
            for pos in self.movement_policy:
                # Terminate early if the seed got too weak.
                # print(len(self.movement_policy.scored_coords))
                if self.seed[start_pos] < self.mov_thr:
                    break

                """根据移动后的坐标分割"""
                pred, sel_i, sel_cent = self.update_at(pos)
                # print(sel_cent)

                sel_i_s = sel_i
                sel_cent_s = sel_cent

                self._min_pos = np.minimum(self._min_pos, pos)
                self._max_pos = np.maximum(self._max_pos, pos)
                num_iters += 1
                try:

                    mask = self.seed[tuple(sel_i_s)] >= self.seg_thr
                    self.seg_prob_i[tuple(sel_i_s)][mask] = quantize_probability(expit(self.seed[tuple(sel_i_s)][mask]))

                except RuntimeError:
                    return False
                skimage.io.imsave('./data/FFN_object1_inf_{}_step{}.tif'.format(id, num_iters), self.seg_prob_i)

                """更新移动策略"""
                self.movement_policy.update(pred, pos)

                assert np.all(pred.shape == self.input_size)

            try:

                mask = self.seed[tuple(sel_i_s)] >= self.seg_thr
                self.seg_prob_i[tuple(sel_i_s)][mask] = quantize_probability(expit(self.seed[tuple(sel_i_s)][mask]))
            except RuntimeError:
                return False

            mask_seg_prob_i = (self.seg_prob_i >= 100)
            self.done_mask[mask_seg_prob_i] = True
            if np.sum(mask_seg_prob_i) >= 500:

                seg_prob_mask = (self.seg_prob_i > 100)
                num_segmented_voxels = np.sum(seg_prob_mask)

                segmented_mask = (self.segmented_mask != 0)
                overlap_mask = (seg_prob_mask * segmented_mask)
                # num_overlap_mask_voxels = np.sum(overlap_mask)
                id_of_overlap_list = stats.mode(self.segmented_mask[overlap_mask])
                try:
                    id_of_overlap = id_of_overlap_list[0][0]
                    print("id_of_overlap", id_of_overlap)
                except IndexError:
                    id_of_overlap =1


                id_of_done_mask = (self.segmented_mask == id_of_overlap)
                num_segmented_done_voxels = np.sum(id_of_done_mask)

                id_of_overlap_done_mask = (id_of_done_mask * overlap_mask)
                over_lap_voxels = np.sum(id_of_overlap_done_mask)

                ratio1 = over_lap_voxels / num_segmented_voxels
                ratio2 = over_lap_voxels / num_segmented_done_voxels
                print("ratio", ratio1)
                print(ratio2)
                if (ratio1 <= 0.2) | (ratio2 <= 0.2):
                    print("id",id)
                    self.segmented_mask[seg_prob_mask] = id
                else:
                    self.segmented_mask[seg_prob_mask] = id_of_overlap
                    print("merged")

                # stacked_img = np.stack((self.segmented_mask,) * 3, axis=-1)
                # stacked_img[tuple(sel_cent_s)] = (250, 0, 0)
                # sel_i = None

                # if num_iters % 30 ==29 :

                ids = np.unique(self.segmented_mask)
                # print(ids)

                stacked_img = np.stack((self.segmented_mask,) * 3, axis=-1)

                for id_i in ids:
                    # print(id_i)
                    id_mask = (self.segmented_mask == id_i)

                    rad2 = random.randrange(10, 254, 1)
                    rad3 = random.randrange(10, 254, 1)
                    rad1 = random.randrange(10, 254, 1)
                    if id_i == 0:
                        rad1 = rad2 = rad3 = 0
                    stacked_img[id_mask] = (rad1, rad2, rad3)

                skimage.io.imsave('./data/id_{}_{}.tif'.format(id, start_pos), stacked_img.astype('uint8'))
                print("id", id)
                return True
            else:
                print("too small", id)

        except RuntimeError:
            return False

    def segment_all(self):

        mbd = np.array([1, 1, 1])
        iter = 0
        # print()
        try:
            for pos in self.seed_list:

                count = round(1.0 * iter / len(self.seed_list) * 50)

                sys.stdout.write('[ {}/{}: [{}{}]\r'.format(iter + 1, len(self.seed_list),
                                                            '#' * count, ' ' * (50 - count)))
                iter += 1

                """根据有效坐标计算slice"""
                if not self.is_valid_pos(pos, ignore_move_threshold=True):
                    continue

                low = np.array(pos) - mbd
                high = np.array(pos) + mbd + 1
                sel = [slice(s, e) for s, e in zip(low, high)]
                if np.any(self.segmentation[tuple(sel)] > 0):
                    self.segmentation[pos] = -1
                    continue

                seg_start = time.time()
                """分割当前坐标cube"""
                num_iters = self.segment_at(pos)
                t_seg = time.time() - seg_start

                if num_iters <= 0:
                    continue

                if self.seed[pos] < self.mov_thr:
                    # Mark this location as excluded.
                    if self.segmentation[pos] == 0:
                        self.segmentation[pos] = -1
                    continue

                """根据seed内容计算最后的分割图"""
                sel = [slice(max(s, 0), e + 1) for s, e in
                       zip(self._min_pos - self.input_size // 2, self._max_pos + self.input_size // 2)]
                mask = self.seed[tuple(sel)] >= self.seg_thr
                raw_segmented_voxels = np.sum(mask)
                overlapped_ids, counts = np.unique(self.segmentation[tuple(sel)][mask], return_counts=True)
                valid = overlapped_ids > 0
                overlapped_ids = overlapped_ids[valid]
                counts = counts[valid]
                mask &= self.segmentation[tuple(sel)] <= 0
                actual_segmented_voxels = np.sum(mask)
                if actual_segmented_voxels < 1000:
                    if self.segmentation[pos] == 0:
                        self.segmentation[pos] = -1
                    continue

                """每次不同目标通过id+1实现区分"""
                self.max_id += 1
                while self.max_id in self.origins:
                    self.max_id += 1

                self.segmentation[tuple(sel)][mask] = self.max_id
                self.seg_prob[tuple(sel)][mask] = quantize_probability(expit(self.seed[tuple(sel)][mask]))
                self.overlaps[self.max_id] = np.array([overlapped_ids, counts])
                self.origins[self.max_id] = OriginInfo(pos, num_iters, t_seg)
                max_value = self.segmentation.max()
                self.segmentation[self.segmentation == -1] = 0
                self.segmentation = self.segmentation * (1.0 * 255 / max_value)
                self.target_dic[self.max_id] = self.segmentation.astype(np.uint8)
                self.segmentation = np.zeros(self.shape, dtype=np.int32)

        except RuntimeError:
            return True
