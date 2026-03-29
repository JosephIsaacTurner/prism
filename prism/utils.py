from tqdm import tqdm as _tqdm

# Global factory for progress bars. Defaults to tqdm.
# This can be overridden by a calling application (like a PyQt app)
# by calling set_progress_handler with a custom factory.
_progress_factory = _tqdm

def set_progress_handler(factory):
    """
    Sets the global progress bar handler.
    
    Args:
        factory: A callable that accepts the same arguments as tqdm.tqdm.
                 It should return an object that behaves like a tqdm progress bar
                 (e.g., has .update(), .close(), .set_description(), and is an iterable).
    """
    global _progress_factory
    _progress_factory = factory

def progress_bar(iterable=None, **kwargs):
    """
    Creates a progress bar using the current global handler.
    
    Args:
        iterable: The iterable to wrap.
        **kwargs: Additional arguments to pass to the handler (e.g., desc, total, disable).
    """
    return _progress_factory(iterable, **kwargs)
