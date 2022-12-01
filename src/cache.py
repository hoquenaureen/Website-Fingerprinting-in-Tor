from pathlib import Path
import pickle
import sys
from typing import Any, Callable


class Cache:
    """A helper class for caching function call results to disk."""

    def __init__(self, cache_path: str, quiet: bool = False):
        """Initialize a new cache pointing to a new or existing path on disk.

        :param cache_path: The path to store and load cached results to and from. If the
            path does not already exist, it will be created.
        :type cache_path: str
            :param quiet: Whether to silence progress messages on caching and loading
            results, defaults to False
        :type quiet: bool, optional
        """
        self.cache_path = cache_path
        self.quiet = quiet

        cache_dir = Path(cache_path)
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)
        elif not cache_dir.is_dir():
            print(
                f"ERROR: path {cache_dir} for the cache directory already exists and "
                "is a file. Please rename the file or choose a new path for the cache "
                "directory."
            )
            sys.exit(1)

    def is_cached(self, function: Callable[..., Any]) -> bool:
        """Check if a function has cached results saved.

        This method uses the name of a function to determine if the results have been
        saved, so different functions with the same name will conflict with each other.

        :param function: A function that may have previously been called with a cacher.
        :type function: Callable[..., Any]
        :return: True if the function has cached results available, False otherwise.
        :rtype: bool
        """
        cache_dir = Path(self.cache_path)

        return (cache_dir / f"{function.__name__}.pkl").is_file()

    def run(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run a function with the specified arguments with caching enabled.

        This function will first check if the function has been called using the
        is_cached method (note this method does not check if the arguments to the
        function are the same). If the function has cached results, then the cached data
        will be loaded and returned.

        If the function does not have cached results, then the function will be executed
        with the given arguments. When the function is finished executing, then the
        results will be saved to disk and returned.

        :param function: A function whose results should be cached.
        :type function: Callable[..., Any]
        :return: The return value of the function.
        :rtype: Any
        """
        cache_dir = Path(self.cache_path)

        key = function.__name__

        # Check if the results have already been cached
        cache_file = cache_dir / f"{key}.pkl"
        if self.is_cached(function):
            self._log(f"Loading cache from {self.cache_path}...")
            with cache_file.open("rb") as cf:
                results = pickle.load(cf)

            self._log("Finished loading cached data.")

        else:
            self._log(f"No cache found for {key}. Running function...")
            results = function(*args, **kwargs)
            self._log(f"Function finished executing. Saving cache to {cache_file}...")
            with cache_file.open("wb") as cf:
                pickle.dump(results, cf)

            self._log("Finished saving cached data.")

        return results

    def _log(self, message: str):
        if not self.quiet:
            print(message)
