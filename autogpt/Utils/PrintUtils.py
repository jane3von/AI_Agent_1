from colorama import init, Fore, Back, Style

THOUGHT_COLOR = Fore.GREEN
OBSERVATION_COLOR = Fore.YELLOW
HIGH_NOTICE_COLOR = Fore.RED
BLUE_COLOR = Fore.CYAN
def color_print(text, color=None):
    if color is not None:
        print(color + text + Style.RESET_ALL)
    else:
        print(text)