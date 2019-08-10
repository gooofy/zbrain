#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Copyright 2019 Guenter Bartsch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#

import os
import sys
import codecs
import time
import logging
import datetime
import imp

from optparse import OptionParser

from nltools  import misc

skill_name = 'personal'

sys.path.insert(0, 'zamiaai/skills')

fp, pathname, description = imp.find_module(skill_name)

m = imp.load_module(skill_name, fp, pathname, description)

class fake_dte:
    def __init__(self):
        pass
    def set_prefixes(self, prefixes):
        self.prefixes = prefixes

    def dt(self, lang, q, a):

        outf.write('<dlg lang=\''+lang+'\'>\n')
        outf.write('  <utt>\n')
        if isinstance(q, str):
            outf.write('    <s> %s </s>\n' % q.replace('(','%(%').replace('|','%|%').replace(')','%)%'))
        else:
            for s in q:
                outf.write('    <s> %s </s>\n' % s.replace('(','%(%').replace('|','%|%').replace(')','%)%'))
        outf.write('  </utt>\n')
        outf.write('  <utt>\n')
        outf.write('    <s> %s </s>\n' % a)
        outf.write('  </utt>\n')
        outf.write('</dlg>\n')
        pass

    def ts(self, lang, tn, l):
        pass

class fake_kernel:
    def __init__(self):
        self.dte = fake_dte()

fk = fake_kernel()

outfn = 'foo.xml'
outf = codecs.open(outfn, 'w', 'utf8')

m.get_data(fk)

outf.close()

