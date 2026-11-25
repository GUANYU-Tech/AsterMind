# Prefer Conda's C++ runtime before Python imports any native extensions.
astermind_use_conda_cxx_runtime() {
    local prefix=$1
    local library="$prefix/lib/libstdc++.so.6"
    if [[ ! -d "$prefix/conda-meta" || ! -r "$library" ]]; then
        return 0
    fi

    # The loader splits LD_PRELOAD on whitespace. Its library search path
    # can still represent a Conda prefix that contains spaces.
    if [[ "$library" == *[[:space:]]* ]]; then
        export LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        return 0
    fi

    case "${LD_PRELOAD:-}" in
        "$library"|"$library":*|"$library "*) ;;
        *) export LD_PRELOAD="$library${LD_PRELOAD:+:$LD_PRELOAD}" ;;
    esac
}
