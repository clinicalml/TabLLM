from collections import defaultdict
from string import Template
import pandas as pd


class NoteTemplate(Template):
    def __init__(self, template, prefixes=None, suffixes=None, defaults=None, pre=None, post=None, fns=None):
        super().__init__(template)

        if pre is None:
            pre = {}
        if post is None:
            post = {}
        self.pre = pre
        self.post = post

        def initialize_default_dict(d, default):
            if d is None:
                d = {}
            return defaultdict(lambda: default, d)

        self.prefixes = initialize_default_dict(prefixes, '')
        self.suffixes = initialize_default_dict(suffixes, '')
        self.defaults = initialize_default_dict(defaults, '')

        # Custom functions
        self.fns = [] if fns is None else fns

        # Formatters
        self.format_timestamp = lambda dt: (dt.to_pydatetime()).strftime('%B %-d, %Y')
        self.format_timedelta = lambda td: str((td.to_pytimedelta()).days)

    def substitute(self, mapping, **kwargs):
        if isinstance(mapping, pd.Series):
            mapping = mapping.to_dict()
        assert type(mapping) is dict

        # Pre-formatting
        mapping = {k: self.pre[k](mapping[k]) if k in self.pre.keys() else mapping[k] for k in mapping.keys()}

        # Remove empty string or None
        mapping = {k: mapping[k] for k in mapping.keys() if (mapping[k] != "" and not pd.isna(mapping[k]))}

        # Format special datatypes
        for k in mapping.keys():
            if isinstance(mapping[k], pd.Timestamp):
                mapping[k] = self.format_timestamp(mapping[k])
            if isinstance(mapping[k], pd.Timedelta):
                mapping[k] = self.format_timedelta(mapping[k])

        # For existing groups add prefixes and suffixes
        for k in mapping.keys():
            mapping[k] = self.prefixes[k] + str(mapping[k]) + self.suffixes[k]

        # Post-formatting
        mapping = {k: self.post[k](mapping[k]) if k in self.post.keys() else mapping[k] for k in mapping.keys()}

        # For non existing keys add defaults
        for _, _, k, _ in self.pattern.findall(self.template):
            if k not in mapping.keys():
                mapping[k] = self.defaults[k]

        text = super().substitute(mapping)

        for fn in self.fns:
            text = fn(text)

        return text

