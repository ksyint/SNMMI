from functools import lru_cache
import datetime
import time
import os
from utils.logconf import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

CLASSES = {1: 'Brain', 2: 'Left-Cerebral-White-Matter', 3: 'Left-Cerebral-Cortex', 4: 'Left-Lateral-Ventricle', 5: 'Left-Inf-Lat-Vent', 7: 'Left-Cerebellum-White-Matter', 8: 'Left-Cerebellum-Cortex', 10: 'Left-Thalamus-Proper*', 11: 'Left-Caudate', 12: 'Left-Putamen', 13: 'Left-Pallidum', 14: '3rd-Ventricle', 15: '4th-Ventricle', 16: 'Brain-Stem', 17: 'Left-Hippocampus', 18: 'Left-Amygdala', 24: 'CSF', 26: 'Left-Accumbens-area', 28: 'Left-VentralDC', 30: 'Left-vessel', 31: 'Left-choroid-plexus', 41: 'Right-Cerebral-White-Matter', 42: 'Right-Cerebral-Cortex', 43: 'Right-Lateral-Ventricle', 44: 'Right-Inf-Lat-Vent', 46: 'Right-Cerebellum-White-Matter', 47: 'Right-Cerebellum-Cortex', 49: 'Right-Thalamus-Proper*', 50: 'Right-Caudate', 51: 'Right-Putamen', 52: 'Right-Pallidum', 53: 'Right-Hippocampus', 54: 'Right-Amygdala', 58: 'Right-Accumbens-area', 60: 'Right-VentralDC', 63: 'Right-choroid-plexus', 77: 'WM-hypointensities', 85: 'Optic-Chiasm', 251: 'CC_Posterior', 252: 'CC_Mid_Posterior', 253: 'CC_Central', 254: 'CC_Mid_Anterior', 255: 'CC_Anterior', 1001: 'ctx-lh-bankssts', 1002: 'ctx-lh-caudalanteriorcingulate', 1003: 'ctx-lh-caudalmiddlefrontal', 1005: 'ctx-lh-cuneus', 1006: 'ctx-lh-entorhinal', 1007: 'ctx-lh-fusiform', 1008: 'ctx-lh-inferiorparietal', 1009: 'ctx-lh-inferiortemporal', 1010: 'ctx-lh-isthmuscingulate', 1011: 'ctx-lh-lateraloccipital', 1012: 'ctx-lh-lateralorbitofrontal', 1013: 'ctx-lh-lingual', 1014: 'ctx-lh-medialorbitofrontal', 1015: 'ctx-lh-middletemporal', 1016: 'ctx-lh-parahippocampal', 1017: 'ctx-lh-paracentral', 1018: 'ctx-lh-parsopercularis', 1019: 'ctx-lh-parsorbitalis', 1020: 'ctx-lh-parstriangularis', 1021: 'ctx-lh-pericalcarine', 1022: 'ctx-lh-postcentral', 1023: 'ctx-lh-posteriorcingulate', 1024: 'ctx-lh-precentral', 1025: 'ctx-lh-precuneus', 1026: 'ctx-lh-rostralanteriorcingulate', 1027: 'ctx-lh-rostralmiddlefrontal', 1028: 'ctx-lh-superiorfrontal', 1029: 'ctx-lh-superiorparietal', 1030: 'ctx-lh-superiortemporal', 1031: 'ctx-lh-supramarginal', 1032: 'ctx-lh-frontalpole', 1033: 'ctx-lh-temporalpole', 1034: 'ctx-lh-transversetemporal', 1035: 'ctx-lh-insula', 2001: 'ctx-rh-bankssts', 2002: 'ctx-rh-caudalanteriorcingulate', 2003: 'ctx-rh-caudalmiddlefrontal', 2005: 'ctx-rh-cuneus', 2006: 'ctx-rh-entorhinal', 2007: 'ctx-rh-fusiform', 2008: 'ctx-rh-inferiorparietal', 2009: 'ctx-rh-inferiortemporal', 2010: 'ctx-rh-isthmuscingulate', 2011: 'ctx-rh-lateraloccipital', 2012: 'ctx-rh-lateralorbitofrontal', 2013: 'ctx-rh-lingual', 2014: 'ctx-rh-medialorbitofrontal', 2015: 'ctx-rh-middletemporal', 2016: 'ctx-rh-parahippocampal', 2017: 'ctx-rh-paracentral', 2018: 'ctx-rh-parsopercularis', 2019: 'ctx-rh-parsorbitalis', 2020: 'ctx-rh-parstriangularis', 2021: 'ctx-rh-pericalcarine', 2022: 'ctx-rh-postcentral', 2023: 'ctx-rh-posteriorcingulate', 2024: 'ctx-rh-precentral', 2025: 'ctx-rh-precuneus', 2026: 'ctx-rh-rostralanteriorcingulate', 2027: 'ctx-rh-rostralmiddlefrontal', 2028: 'ctx-rh-superiorfrontal', 2029: 'ctx-rh-superiorparietal', 2030: 'ctx-rh-superiortemporal', 2031: 'ctx-rh-supramarginal', 2032: 'ctx-rh-frontalpole', 2033: 'ctx-rh-temporalpole', 2034: 'ctx-rh-transversetemporal', 2035: 'ctx-rh-insula'}


COLORDICT = {0: (0, 0, 0), 1: (255,255,255), 2: (245, 245, 245), 3: (205, 62, 78), 4: (120, 18, 134), 5: (196, 58, 250), 7: (220, 248, 164), 8: (230, 148, 34), 10: (0, 118, 14), 11: (122, 186, 220), 12: (236, 13, 176), 13: (12, 48, 255), 14: (204, 182, 142), 15: (42, 204, 164), 16: (119, 159, 176), 17: (220, 216, 20), 18: (103, 255, 255), 24: (60, 60, 60), 26: (255, 165, 0), 28: (165, 42, 42), 30: (160, 32, 240), 31: (0, 200, 200), 41: (245, 245, 245), 42: (205, 62, 78), 43: (120, 18, 134), 44: (196, 58, 250), 46: (220, 248, 164), 47: (230, 148, 34), 49: (0, 118, 14), 50: (122, 186, 220), 51: (236, 13, 176), 52: (13, 48, 255), 53: (220, 216, 20), 54: (103, 255, 255), 58: (255, 165, 0), 60: (165, 42, 42), 63: (0, 200, 221), 77: (200, 70, 255), 85: (234, 169, 30), 251: (0, 0, 64), 252: (0, 0, 112), 253: (0, 0, 160), 254: (0, 0, 208), 255: (0, 0, 255), 1001: (25, 100, 40), 1002: (125, 100, 160), 1003: (100, 25, 0), 1005: (220, 20, 100), 1006: (220, 20, 10), 1007: (180, 220, 140), 1008: (220, 60, 220), 1009: (180, 40, 120), 1010: (140, 20, 140), 1011: (20, 30, 140), 1012: (35, 75, 50), 1013: (225, 140, 140), 1014: (200, 35, 75), 1015: (160, 100, 50), 1016: (20, 220, 60), 1017: (60, 220, 60), 1018: (220, 180, 140), 1019: (20, 100, 50), 1020: (220, 60, 20), 1021: (120, 100, 60), 1022: (220, 20, 20), 1023: (220, 180, 220), 1024: (60, 20, 220), 1025: (160, 140, 180), 1026: (80, 20, 140), 1027: (75, 50, 125), 1028: (20, 220, 160), 1029: (20, 180, 140), 1030: (140, 220, 220), 1031: (80, 160, 20), 1032: (100, 0, 100), 1033: (70, 70, 70), 1034: (150, 150, 200), 1035: (255, 192, 32), 2001: (25, 100, 40), 2002: (125, 100, 160), 2003: (100, 25, 0), 2005: (220, 20, 100), 2006: (220, 20, 10), 2007: (180, 220, 140), 2008: (220, 60, 220), 2009: (180, 40, 120), 2010: (140, 20, 140), 2011: (20, 30, 140), 2012: (35, 75, 50), 2013: (225, 140, 140), 2014: (200, 35, 75), 2015: (160, 100, 50), 2016: (20, 220, 60), 2017: (60, 220, 60), 2018: (220, 180, 140), 2019: (20, 100, 50), 2020: (220, 60, 20), 2021: (120, 100, 60), 2022: (220, 20, 20), 2023: (220, 180, 220), 2024: (60, 20, 220), 2025: (160, 140, 180), 2026: (80, 20, 140), 2027: (75, 50, 125), 2028: (20, 220, 160), 2029: (20, 180, 140), 2030: (140, 220, 220), 2031: (80, 160, 20), 2032: (100, 0, 100), 2033: (70, 70, 70), 2034: (150, 150, 200), 2035: (255, 192, 32)}


def enumerateWithEstimate(
        iter,
        desc_str,
        start_ndx=0,
        print_ndx=4,
        backoff=None,
        iter_len=None,
):
    """
    In terms of behavior, `enumerateWithEstimate` is almost identical
    to the standard `enumerate` (the differences are things like how
    our function returns a generator, while `enumerate` returns a
    specialized `<enumerate object at 0x...>`).

    However, the side effects (logging, specifically) are what make the
    function interesting.

    :param iter: `iter` is the iterable that will be passed into
        `enumerate`. Required.

    :param desc_str: This is a human-readable string that describes
        what the loop is doing. The value is arbitrary, but should be
        kept reasonably short. Things like `"epoch 4 training"` or
        `"deleting temp files"` or similar would all make sense.

    :param start_ndx: This parameter defines how many iterations of the
        loop should be skipped before timing actually starts. Skipping
        a few iterations can be useful if there are startup costs like
        caching that are only paid early on, resulting in a skewed
        average when those early iterations dominate the average time
        per iteration.

        NOTE: Using `start_ndx` to skip some iterations makes the time
        spent performing those iterations not be included in the
        displayed duration. Please account for this if you use the
        displayed duration for anything formal.

        This parameter defaults to `0`.

    :param print_ndx: determines which loop interation that the timing
        logging will start on. The intent is that we don't start
        logging until we've given the loop a few iterations to let the
        average time-per-iteration a chance to stablize a bit. We
        require that `print_ndx` not be less than `start_ndx` times
        `backoff`, since `start_ndx` greater than `0` implies that the
        early N iterations are unstable from a timing perspective.

        `print_ndx` defaults to `4`.

    :param backoff: This is used to how many iterations to skip before
        logging again. Frequent logging is less interesting later on,
        so by default we double the gap between logging messages each
        time after the first.

        `backoff` defaults to `2` unless iter_len is > 1000, in which
        case it defaults to `4`.

    :param iter_len: Since we need to know the number of items to
        estimate when the loop will finish, that can be provided by
        passing in a value for `iter_len`. If a value isn't provided,
        then it will be set by using the value of `len(iter)`.

    :return:
    """
    if iter_len is None:
        iter_len = len(iter)

    if backoff is None:
        backoff = 2
        while backoff ** 7 < iter_len:
            backoff *= 2

    assert backoff >= 2
    while print_ndx < start_ndx * backoff:
        print_ndx *= backoff

    log.warning("{} ----/{}, starting".format(
        desc_str,
        iter_len,
    ))
    start_ts = time.time()
    for (current_ndx, item) in enumerate(iter):
        yield (current_ndx, item)
        if current_ndx == print_ndx:
            duration_sec = (
                    (time.time() - start_ts) / (current_ndx - start_ndx + 1) * (iter_len-start_ndx)
            )

            done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
            done_td = datetime.timedelta(seconds=duration_sec)

            log.info("{} {:-4}/{}, done at {}, {}".format(
                desc_str,
                current_ndx,
                iter_len,
                str(done_dt).rsplit('.', 1)[0],
                str(done_td).rsplit('.', 1)[0],
            ))

            print_ndx *= backoff

        if current_ndx + 1 == start_ndx:
            start_ts = time.time()

    log.warning("{} ----/{}, done at {}".format(
        desc_str,
        iter_len,
        str(datetime.datetime.now()).rsplit('.', 1)[0],
    ))