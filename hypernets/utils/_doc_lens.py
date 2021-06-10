import inspect
import re
from collections import OrderedDict

_KEY_PARAMETERS = 'Parameters'


class DocLens(object):
    def __init__(self, doc):
        self.doc_ = doc  # original doc

        self.synopsis, self.sections = self._parse_doc(doc)

    @property
    def parameters(self):
        if _KEY_PARAMETERS in self.sections.keys():
            return self._parse_parameters(self.sections[_KEY_PARAMETERS])
        else:
            return OrderedDict()

    @parameters.setter
    def parameters(self, params):
        assert isinstance(params, OrderedDict)

        content = '\n'.join([f'{k}{v}' for k, v in params.items()])
        self.sections[_KEY_PARAMETERS] = content

    def render(self):
        result = []
        if self.synopsis is not None and len(self.synopsis) > 0:
            result.append(self.synopsis)

        for name, doc in self.sections.items():
            result.append(f'{name}\n------\n{doc}')

        return '\n'.join(result)

    def merge_parameters(self, other, exclude=None):
        if exclude is None:
            exclude = []

        params = self.parameters
        for k, v in other.parameters.items():
            if k not in exclude and k not in params.keys():
                params[k] = v

        return params

    @staticmethod
    def _parse_doc(doc):
        """
        split pydoc into synopsis and sections dict
        """
        doc = inspect.cleandoc(doc)

        sa = re.split(r'\n[-=]+\n', doc)
        if len(sa) <= 1:
            return doc, {}

        synopsis = ''
        sections = OrderedDict()
        section_name = None

        for s in sa[:-1]:
            i = s.rfind('\n')
            if i >= 0:
                content = s[:i]
                next_section_name = s[i + 1:]
            else:
                content = ''
                next_section_name = s

            if section_name is None:
                synopsis = content
            else:
                sections[section_name] = s[:i]
            section_name = next_section_name
        sections[section_name] = sa[-1]

        return synopsis, sections

    @staticmethod
    def _parse_parameters(doc):
        """
        split pydoc into parameter dict
        """
        params = OrderedDict()
        detail = None
        for line in doc.split('\n'):
            m = re.search(r'^(\w+)(.*)', line)
            if m:
                name, synopsis = m.groups()
                detail = [synopsis]
                params[name] = detail
            elif detail is not None:
                detail.append(line)

        result = OrderedDict()
        for name, detail in params.items():
            result[name] = '\n'.join(detail)

        return result
