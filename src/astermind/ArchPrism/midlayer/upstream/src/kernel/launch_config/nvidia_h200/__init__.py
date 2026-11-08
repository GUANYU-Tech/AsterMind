"""NVIDIA H200 launch selection for kernels the A800 package does not select.

The dispatcher (select_launch_config._OWN_PACKAGE_SELECTORS) routes only the
(module, function) selectors listed there to this package; every other H200
launch uses the projected A800 package.  Modules here must not import another
hardware package.
"""
