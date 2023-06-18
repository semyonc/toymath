# -*- coding: utf-8 -*-
import sys
import logging
import base64

from ipykernel.comm import CommManager
from ipywidgets import register_comm_target
from ipykernel.kernelapp import IPKernelApp
from ipykernel.kernelbase import Kernel

import engine.mathShell as mathShell
from engine import setShell, setHandler

try:
    from ipywidgets.widgets.widget import Widget
except ImportError:
    Widget = None

try:
    from IPython.utils.PyColorize import NeutralColors

    RED = NeutralColors.colors["header"]
    NORMAL = NeutralColors.colors["normal"]
except:
    from IPython.core.excolors import TermColors

    RED = TermColors.Red
    NORMAL = TermColors.Normal


class MathKernel(Kernel):
    implementation = 'toymath'
    implementation_version = '1.0'
    language = 'tex'
    language_version = '0.1'
    language_info = {
        'name': 'LaTex',
        'mimetype': 'text/plain',
        'file_extension': '.tex',
        'pygments_lexer': 'pygments.lexers.markup.TexLexer'
    }
    banner = "Toy math kernel"
    svgimage = '''<svg viewBox="-40 0 512 512" width="12px" xmlns="http://www.w3.org/2000/svg"><path d="m271 512h-191c-44.113281 0-80-35.886719-80-80v-271c0-44.113281 35.886719-80 80-80h191c44.113281 0 80 35.886719 80 80v271c0 44.113281-35.886719 80-80 80zm-191-391c-22.054688 0-40 17.945312-40 40v271c0 22.054688 17.945312 40 40 40h191c22.054688 0 40-17.945312 40-40v-271c0-22.054688-17.945312-40-40-40zm351 261v-302c0-44.113281-35.886719-80-80-80h-222c-11.046875 0-20 8.953125-20 20s8.953125 20 20 20h222c22.054688 0 40 17.945312 40 40v302c0 11.046875 8.953125 20 20 20s20-8.953125 20-20zm0 0"/></svg>'''
    copyButton = '''<button onclick=javascript:navigator.clipboard.writeText(atob("{}")) style="color: #777; background-color: rgba(0,0,0,0); border-color: rgba(0,0,0,0); cursor:copy;">''' + svgimage + '</button>'
    copySpan = '<span style="padding-left: 5x;">' + copyButton + '</span>'

    def __init__(self, **kwargs):
        super(MathKernel, self).__init__(**kwargs)

        if self.log is None:
            self.log = logging.Logger(".toymath")
        else:
            try:
                sys.stdout.write = self.Write
            except:
                pass

        setHandler(self.Display)
        self.redirect_to_log = False

        self.comm_manager = CommManager(parent=self, kernel=self)
        register_comm_target(self)
        comm_msg_types = ['comm_open', 'comm_msg', 'comm_close']
        for msg_type in comm_msg_types:
            self.shell_handlers[msg_type] = getattr(self.comm_manager, msg_type)

        self.mathShell = mathShell.MathShell()
        setShell(self.mathShell)


    def makeCopySpan(self, latex):
        latex_bytes = latex.encode('utf-8')
        return MathKernel.copySpan.format(str(base64.b64encode(latex_bytes), 'ascii'))

    def do_execute(self, code, silent, store_history=True,
                   user_expressions=None, allow_stdin=False, *, cell_id=None):
        # Set the ability for the kernel to get standard-in:
        self._allow_stdin = allow_stdin
        # Create a default response:
        kernel_resp = {
            'status': 'ok',
            # The base class increments the execution count
            'execution_count': self.execution_count,
            'payload': [],
            'user_expressions': {},
        }
        try:
            self.mathShell.exec(code, self.execution_count, store_history, cell_id=cell_id)
            return kernel_resp
        except Exception as e:
            self.Error(e)
            return {'status': 'error',
                    # The base class increments the execution count
                    'ename': type(e).__name__,  # Exception name, as a string
                    'evalue': e.args[0]  # Exception value, as a string
                    }

    def repr(self, item):
        """The repr of the kernel."""
        return repr(item)

    def clear_output(self, wait=False):
        """Clear the output of the kernel."""
        self.send_response(self.iopub_socket, 'clear_output',
                           {'wait': wait})

    def Write(self, message):
        """Write message directly to the iopub stdout with no added end character."""
        stream_content = {
            'name': 'stdout', 'text': message}
        self.log.debug('Write: %s' % message)
        if self.redirect_to_log:
            self.log.info(message)
        else:
            self.send_response(self.iopub_socket, 'stream', stream_content)

    def Error(self, *objects, **kwargs):
        """Print `objects` to stdout, separated by `sep` and followed by `end`.
        Objects are cast to strings.
        """
        message = format_message(*objects, **kwargs)
        self.log.debug('Error: %s' % message.rstrip())
        stream_content = {
            'name': 'stderr',
            'text': RED + message + NORMAL
        }
        if self.redirect_to_log:
            self.log.info(message.rstrip())
        else:
            self.send_response(self.iopub_socket, 'stream', stream_content)

    def Display(self, *objects, **kwargs):
        """Display one or more objects using rich display.
        Supports a `clear_output` keyword argument that clears the output before displaying.
        See https://ipython.readthedocs.io/en/stable/config/integrating.html?highlight=display#rich-display
        """
        if kwargs.get('clear_output'):
            self.clear_output(wait=True)

        for item in objects:
            if Widget and isinstance(item, Widget):
                self.log.debug('Display Widget')
                data = {
                    'text/plain': repr(item),
                    'application/vnd.jupyter.widget-view+json': {
                        'version_major': 2,
                        'version_minor': 0,
                        'model_id': item._model_id
                    }
                }
                content = {
                    'data': data,
                    'metadata': {}
                }
                self.send_response(
                    self.iopub_socket,
                    'display_data',
                    content
                )
            else:
                self.log.debug('Display Data')
                try:
                    data = _formatter(item, self.repr)
                except Exception as e:
                    self.Error(e)
                    return
                content = {
                    'data': data[0],
                    'metadata': data[1]
                }
                self.send_response(
                    self.iopub_socket,
                    'display_data',
                    content
                )


def format_message(*objects, **kwargs):
    """
    Format a message like print() does.
    """
    objects = [str(i) for i in objects]
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    return sep.join(objects) + end


def _formatter(data, repr_func):
    reprs = {'text/plain': repr_func(data)}

    lut = [("_repr_png_", "image/png"),
           ("_repr_jpeg_", "image/jpeg"),
           ("_repr_html_", "text/html"),
           ("_repr_markdown_", "text/markdown"),
           ("_repr_svg_", "image/svg+xml"),
           ("_repr_latex_", "text/latex"),
           ("_repr_json_", "application/json"),
           ("_repr_javascript_", "application/javascript"),
           ("_repr_pdf_", "application/pdf")]

    for (attr, mimetype) in lut:
        obj = getattr(data, attr, None)
        if obj:
            reprs[mimetype] = obj

    format_dict = {}
    metadata_dict = {}
    for (mimetype, value) in reprs.items():
        metadata = None
        try:
            value = value()
        except Exception:
            pass
        if not value:
            continue
        if isinstance(value, tuple):
            metadata = value[1]
            value = value[0]
        if isinstance(value, bytes):
            try:
                value = value.decode('utf-8')
            except Exception:
                value = base64.encodestring(value)
                value = value.decode('utf-8')
        try:
            format_dict[mimetype] = str(value)
        except:
            format_dict[mimetype] = value
        if metadata is not None:
            metadata_dict[mimetype] = metadata
    return format_dict, metadata_dict


class MathKernelApp(IPKernelApp):
    kernel_class = MathKernel


if __name__ == '__main__':
    logging.disable(logging.ERROR)
    MathKernelApp.launch_instance()
