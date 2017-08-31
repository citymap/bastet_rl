# This file uses modified code of pwntools[https://github.com/Gallopsled/pwntools]
# which is licensed as following:
#
# Copyright (c) 2015 Gallopsled et al.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import shutil
import platform


def run_in_new_terminal(command, terminal=None, args=None):
    """run_in_new_terminal(command, terminal = None) -> None

    Run a command in a new terminal.

    When ``terminal`` is not set:
        - If ``context.terminal`` is set it will be used.
          If it is an iterable then ``context.terminal[1:]`` are default arguments.
        - If a ``pwntools-terminal`` command exists in ``$PATH``, it is used
        - If ``$TERM_PROGRAM`` is set, that is used.
        - If X11 is detected (by the presence of the ``$DISPLAY`` environment
          variable), ``x-terminal-emulator`` is used.
        - If tmux is detected (by the presence of the ``$TMUX`` environment
          variable), a new pane will be opened.

    Arguments:
        command (str): The command to run.
        terminal (str): Which terminal to use.
        args (list): Arguments to pass to the terminal

    Note:
        The command is opened with ``/dev/null`` for stdin, stdout, stderr.

    Returns:
      PID of the new terminal process
    """

    if not terminal:
        if 'DISPLAY' in os.environ and shutil.which('x-terminal-emulator'):
            terminal = 'x-terminal-emulator'
            args = ['-e']
        elif 'TMUX' in os.environ and shutil.which('tmux'):
            terminal = 'tmux'
            args = ['splitw', '-h']

    if isinstance(args, tuple):
        args = list(args)

    argv = [shutil.which(terminal)] + args

    if isinstance(command, (list, tuple)):
        argv += list(command)
    elif isinstance(command, str):
        argv += [command]

    print("Launching a new terminal: %r" % argv)

    pid = os.fork()

    if pid == 0:
        # Closing the file descriptors makes everything fail under tmux on OSX.
        if platform.system() != 'Darwin':
            devnull = open(os.devnull, 'r+b')
            os.dup2(devnull.fileno(), 0)
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
        os.execv(argv[0], argv)
        os._exit(1)

    return pid
