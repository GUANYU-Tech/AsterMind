"""Make src packages available to the repository-local simulator CLI."""

from . import SOURCE as SOURCE, WORKSPACE as WORKSPACE

from astermind.bootstrap import DEFAULT_ML, activate
