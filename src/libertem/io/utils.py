try:
    import pwd

    def get_owner_name(full_path, stat):
        return pwd.getpwuid(stat.st_uid).pw_name
# Assume that we are on Windows
# TODO do a proper abstraction layer if this simple solution doesn't work anymore
except ModuleNotFoundError:
    from libertem.win_tweaks import get_owner_name  # noqa: F401
