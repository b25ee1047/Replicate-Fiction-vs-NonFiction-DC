"""Lazy exports for corpus loaders."""


def preprocess_brown_corpus(*args, **kwargs):
    from .load_brown import preprocess_brown_corpus as _impl
    return _impl(*args, **kwargs)


def extract_brown_features(*args, **kwargs):
    from .load_brown import extract_document_features as _impl
    return _impl(*args, **kwargs)


def preprocess_bnc_corpus(*args, **kwargs):
    from .load_bnc import preprocess_bnc_corpus as _impl
    return _impl(*args, **kwargs)


def load_bnc_news(*args, **kwargs):
    from .load_bnc import load_bnc_news as _impl
    return _impl(*args, **kwargs)


def preprocess_jefp_corpus(*args, **kwargs):
    from .load_jefp import preprocess_jefp_corpus as _impl
    return _impl(*args, **kwargs)


def inspect_jefp_metadata(*args, **kwargs):
    from .load_jefp import inspect_jefp_metadata as _impl
    return _impl(*args, **kwargs)
