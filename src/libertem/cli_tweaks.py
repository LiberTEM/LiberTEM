import platform


def console_tweaks():
    if platform.system() == "Windows":
        from libertem.win_tweaks import disable_quickedit
        disable_quickedit()
