# Code data source samples (10 per source)

Random seed 123. Each sample is the first 1500 characters of one document.

These are the actual raw training documents from each data source.

---

## Overview: what each source is, why we used it, and how big it is

Token counts use the **Llama-3.1 tokenizer** (`meta-llama/Meta-Llama-3.1-8B`) — the same one used in C5/B4/A5 training. `avg_tok/doc` is measured by sampling 500 docs per source (750 for StarCoderData = 50 per language × 15 languages). `published_rows` is from the HF datasets-server REST API or dataset card. `est_total_tokens` = `avg_tok × published_rows`.

| # | Source | Used in | Why chosen | avg_tok/doc | local_rows | local_total_tok | published_rows | published_total_tok |
|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | `bigcode/starcoderdata` (Stack v1 + StarCoder filters) | **C5-stage1** (100% × 15.4 B trained); **C5-final stage 2** (~8% of stage 2 = 1.2 B) | Aryabumi et al's *exact* data source at their *exact* Tables 3+4 mixed-language ratios. C5 is a faithful replication of "To Code, or Not To Code?" — picking any other source would have made the failure-to-replicate result un-attributable. Locally we have 15 of ~86 languages (the Aryabumi-selected subset). | 1,893 | 19,920,523 | **~37.7 B** | 206,642,239 (all langs) | **~391 B** (all langs) |
| 2 | `codeparrot/github-code` (Python only) | **B4 as "aryabumi_web" slice** (1.35 B = 4.4% of B4) | B4 mirrored Aryabumi's "synth-code vs web-code" split. github-code (unfiltered Python crawl) was the closest source we had to "raw web code". | 1,465 | 634,376 | **~0.93 B** | 7,226,626 (Python only) | **~10.6 B** |
| 3 | `nvidia/OpenCodeReasoning` (FULL: problem + `<think>` + solution) | **B4 as "aryabumi_synth" slice** (5.4 B = 17.5% of B4) | Hypothesis: reasoning-heavy synthetic code would boost both code AND math/reasoning evals (kill two birds). OpenCodeReasoning was the only public dataset with explicit `<think>` traces at scale. B4 used 5.4 B from our 4.88 B local snapshot → ~**1.1 epoch** through it. | 8,594 | 567,850 | **~4.88 B** | 337,766 ‡ | **~2.9 B** ‡ |
| 4 | `nvidia/OpenCodeReasoning` solution-only | **not used** | Natural ablation pair for #3 ("does the `<think>` trace help?"), but we never ran the controlled comparison. Open ablation. | 289 | 567,806 | **~0.16 B** | 337,766 ‡ | **~0.10 B** ‡ |
| 5 | `OpenCoder-LLM/opc-annealing-corpus` / `algorithmic_corpus` | **B4 as "opc" slice** (0.94 B = 3.1% of B4); **code25 v2 baseline** (50 M unique × 16 epochs ≈ 800 M trained) | Format-aligned with MBPP eval ("write a python function that does X" → clean solution). For B4 the smallest "format-target" slice; for code25 v2 this was the *only* code source — a single high-quality dataset to fit a 50 M-unique slot. | 184 | 5,322,920 | **~0.98 B** | 5,322,920 | **~0.98 B** |
| 6 | `OpenCoder-LLM/opc-annealing-corpus` / `synthetic_code_snippet` (phi-style textbook code) | **not used** | Sampled when studying *why* phi-1.5 wins HumanEval — we hadn't yet pivoted to testing phi-style data ourselves. This is the obvious next experiment given C5's failure to replicate Aryabumi. | 379 | (HF stream) | — | 3,081,235 | **~1.17 B** |
| 7 | `OpenCoder-LLM/opc-annealing-corpus` / `synthetic_qa` | **not used** | Sampled as the cleaner LeetCode-style alternative to #3 (no `<think>`). Not used — B4 was already locked in to #3 + #2 by then. | 434 | (HF stream) | — | 3,238,929 | **~1.41 B** |

**‡** For nvidia/OpenCodeReasoning the HF datasets-server reports 337,766 rows (sum of `split_0` + `split_1` parquet configs) but our local snapshot has 567,850 rows — likely an earlier dataset revision or additional non-parquet splits. The "local_total_tok" column reflects our actual snapshot and is the right number for "what B4 trained on."

**Note on tokenization**: published_rows × avg_tok/doc gives Llama-3.1-tokenizer total — different from token counts published in source-paper Tables (which often use the source's native tokenizer). The relevant number for "what we actually trained on" is the trained-tokens figure in §2 of `experiments/data_efficiency/EVALUATION.md`.

**Note on local snapshot**: For sources 1–5 we have local jsonl.gz files; for 6–7 we read directly from HF streaming for the samples below.

---

## StarCoderData / python (raw Stack v1 w/ Starcoder filters — what C5-stage1 trained on)

Source: `/fsx/users/dongweij/marin/outputs/raw/starcoderdata/python.jsonl.gz`

### Sample 1

```
import os.path
import time
import logging
import yaml
from piecrust.processing.base import Processor


logger = logging.getLogger(__name__)


class _ConcatInfo(object):
    timestamp = 0
    files = None
    delim = "\n"


class ConcatProcessor(Processor):
    PROCESSOR_NAME = 'concat'

    def __init__(self):
        super(ConcatProcessor, self).__init__()
        self._cache = {}

    def matches(self, path):
        return path.endswith('.concat')

    def getDependencies(self, path):
        info = self._load(path)
        return info.files

    def getOutputFilenames(self, filename):
        return [filename[:-7]]

    def process(self, path, out_dir):
        dirname, filename = os.path.split(path)
        out_path = os.path.join(out_dir, filename[:-7])
        info = self._load(path)
        if not info.files:
            raise Exception("No files specified in: %s" %
                            os.path.relpath(path, self.app.root_dir))

        logger.debug("Concatenating %d files to: %s" %
                     (len(info.files), out_path))
        encoded_delim = info.delim.encode('utf8')
        with open(out_path, 'wb') as ofp:
            for p in info.files:
                with open(p, 'rb') as ifp:
                    ofp.write(ifp.read())
                if info.delim:
                    ofp.write(encoded_delim)
        return True

    def _load(self, path):
        cur_time = time.time()
        info = self._cache.get(path)
        if (info is not None and
  
... [truncated; full doc has 2,385 chars]
```

### Sample 2

```
<reponame>sebtelko/pulumi-azure-native
# coding=utf-8
# *** WARNING: this file was generated by the Pulumi SDK Generator. ***
# *** Do not edit by hand unless you're certain you know what you are doing! ***

import warnings
import pulumi
import pulumi.runtime
from typing import Any, Mapping, Optional, Sequence, Union, overload
from .. import _utilities

__all__ = ['StorageAccountStaticWebsiteArgs', 'StorageAccountStaticWebsite']

@pulumi.input_type
class StorageAccountStaticWebsiteArgs:
    def __init__(__self__, *,
                 account_name: pulumi.Input[str],
                 resource_group_name: pulumi.Input[str],
                 error404_document: Optional[pulumi.Input[str]] = None,
                 index_document: Optional[pulumi.Input[str]] = None):
        """
        The set of arguments for constructing a StorageAccountStaticWebsite resource.
        :param pulumi.Input[str] account_name: The name of the storage account within the specified resource group.
        :param pulumi.Input[str] resource_group_name: The name of the resource group within the user's subscription. The name is case insensitive.
        :param pulumi.Input[str] error404_document: The absolute path to a custom webpage that should be used when a request is made which does not correspond to an existing file.
        :param pulumi.Input[str] index_document: The webpage that Azure Storage serves for requests to the root of a website or any sub-folder. For example, 'index.html'. The value is case
... [truncated; full doc has 9,829 chars]
```

### Sample 3

```
"""Support for Epson Workforce Printer."""
from datetime import timedelta
import logging

import voluptuous as vol

from homeassistant.components.sensor import PLATFORM_SCHEMA
from homeassistant.const import CONF_HOST, CONF_MONITORED_CONDITIONS
from homeassistant.exceptions import PlatformNotReady
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers.entity import Entity

REQUIREMENTS = ['epsonprinter==0.0.8']

_LOGGER = logging.getLogger(__name__)
MONITORED_CONDITIONS = {
    'black': ['Inklevel Black', '%', 'mdi:water'],
    'magenta': ['Inklevel Magenta', '%', 'mdi:water'],
    'cyan': ['Inklevel Cyan', '%', 'mdi:water'],
    'yellow': ['Inklevel Yellow', '%', 'mdi:water'],
    'clean': ['Inklevel Cleaning', '%', 'mdi:water'],
}
PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend({
    vol.Required(CONF_HOST): cv.string,
    vol.Required(CONF_MONITORED_CONDITIONS):
        vol.All(cv.ensure_list, [vol.In(MONITORED_CONDITIONS)]),
})
SCAN_INTERVAL = timedelta(minutes=60)


def setup_platform(hass, config, add_devices, discovery_info=None):
    """Set up the cartridge sensor."""
    host = config.get(CONF_HOST)

    from epsonprinter_pkg.epsonprinterapi import EpsonPrinterAPI
    api = EpsonPrinterAPI(host)
    if not api.available:
        raise PlatformNotReady()

    sensors = [EpsonPrinterCartridge(api, condition)
               for condition in config[CONF_MONITORED_CONDITIONS]]

    add_devices(sensors, True)


class EpsonPrinterCartridge(Entity):
    """
... [truncated; full doc has 2,590 chars]
```

### Sample 4

```
import hashlib
import mimetypes
from urllib.parse import unquote

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models
from django.http import HttpResponseRedirect
from django.template.loader import render_to_string
from django.urls import reverse
from django.utils.functional import cached_property
from django.utils.safestring import mark_safe
from django.utils.text import slugify
from django.utils.translation import ugettext_lazy as _
from django_extensions.db.fields import CreationDateTimeField, ModificationDateTimeField
from great_components.mixins import GA360Mixin
from modelcluster.contrib.taggit import ClusterTaggableManager
from modelcluster.models import ClusterableModel, ParentalKey
from taggit.managers import TaggableManager
from taggit.models import ItemBase, TagBase, TaggedItemBase
from wagtail.admin.edit_handlers import (
    FieldPanel,
    InlinePanel,
    MultiFieldPanel,
    ObjectList,
    PageChooserPanel,
    StreamFieldPanel,
    TabbedInterface,
)
from wagtail.contrib.redirects.models import Redirect
from wagtail.contrib.settings.models import BaseSetting, register_setting
from wagtail.core import blocks
from wagtail.core.blocks.stream_block import StreamBlockValidationError
from wagtail.core.fields import RichTextField, StreamField
from wagtail.core.models import Orderable, Page
from wagtail.images import get_image_model_string
from wagtail.images.edit_handlers import ImageChooserPanel
from wag
... [truncated; full doc has 41,687 chars]
```

### Sample 5

```
# model settings
model = dict(
    type='Semi_AppSup_TempSup_SimCLR_Crossclip_PTV_Recognizer3D',
    backbone=dict(
        type='ResNet3d',
        depth=18,
        pretrained=None,
        pretrained2d=False,
        norm_eval=False,
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=dict(type='SyncBN', requires_grad=True, eps=1e-3),
        act_cfg=dict(type='ReLU'),
        conv1_kernel=(3, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(1, 1, 1, 1),
        spatial_strides=(1, 2, 2, 2),
        temporal_strides=(1, 2, 2, 2),
        zero_init_residual=False),
    cls_head=dict(
        type='I3DHead',
        num_classes=400,
        in_channels=512,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01),
    cls_head_temp=None,
    temp_backbone='same',
    temp_sup_head='same',
    train_cfg=dict(
        warmup_epoch=10,
        fixmatch_threshold=0.3,
        temp_align_indices=(0, 1, 2, 3),
        align_loss_func='Cosine',
        pseudo_label_metric='avg',
        crossclip_contrast_loss=[],
        crossclip_contrast_range=[],
    ),
    test_cfg=dict(average_clips='score'))

# dataset settings
dataset_type = 'VideoDataset'
dataset_type_labeled = 'VideoDataset_Contrastive'
dataset_type_unlabeled = 'UnlabeledVideoDataset_MultiView_Contrastive'
# dataset_type_appearance = 'RawframeDataset_withAPP'

data_root = 'data/kinetics400/videos_train'
data_root_val = 'data/kinetics400/videos_val'

labeled_percentage = 1
... [truncated; full doc has 7,469 chars]
```

### Sample 6

```
#   copyright (c) 2018 paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

from __future__ import print_function

import os
import numpy as np
import random
import unittest
import logging
import warnings

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid import core
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.framework import IrGraph
from paddle.fluid.contrib.slim.quantization import ImperativeQuantAware
from paddle.fluid.contrib.slim.quantization import OutScaleForTrainingPass, OutScaleForInferencePass, QuantizationTransformPass
from paddle.fluid.dygraph.container import Sequential
from paddle.fluid.dygraph.io import INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX
from paddle.nn.layer import ReLU, LeakyReLU, Sigmoid, Softmax, PReLU
from paddle.nn import Linear, Conv2D, Softmax, BatchNorm2D, MaxPool2D
from paddle.fluid.dygraph.nn import Pool2D
from paddle.fluid.log_helper import get_log
... [truncated; full doc has 19,452 chars]
```

### Sample 7

```
<filename>expyfun/_utils.py
"""Some utility functions"""

# Authors: <NAME> <<EMAIL>>
#
# License: BSD (3-clause)

import warnings
import operator
from copy import deepcopy
import subprocess
import importlib
import os
import os.path as op
import inspect
import sys
import tempfile
import ssl
from shutil import rmtree
import atexit
import json
from functools import partial
from distutils.version import LooseVersion
from numpy import sqrt, convolve, ones
import logging
import datetime
from timeit import default_timer as clock
from threading import Timer

import numpy as np
import scipy as sp

from ._externals import decorator

# set this first thing to make sure it "takes"
try:
    import pyglet
    pyglet.options['debug_gl'] = False
    del pyglet
except Exception:
    pass


# for py3k (eventually)
if sys.version.startswith('2'):
    string_types = basestring  # noqa
    input = raw_input  # noqa, input is raw_input in py3k
    text_type = unicode  # noqa
    from __builtin__ import reload
    from urllib2 import urlopen  # noqa
    from cStringIO import StringIO  # noqa
else:
    string_types = str
    text_type = str
    from urllib.request import urlopen
    input = input
    from io import StringIO  # noqa, analysis:ignore
    from importlib import reload  # noqa, analysis:ignore

###############################################################################
# LOGGING

EXP = 25
logging.addLevelName(EXP, 'EXP')


def exp(self, message, *args, **kwargs):
    """Experiment-l
... [truncated; full doc has 28,546 chars]
```

### Sample 8

```
import unittest

from recipe import utils


class UtilTestCase(unittest.TestCase):
    def test_valid_project_slug(self):
        project_slug = "Recipe0123456789_mock"
        self.assertTrue(utils.valid_project_slug(project_slug))

        project_slug = 'Recipe00000000000000000000000000000000000000000000'
        self.assertTrue(utils.valid_project_slug(project_slug))

        project_slug = ""
        self.assertFalse(utils.valid_project_slug(project_slug))

        project_slug = "Recipe000000000000000000000000000000000000000000001"
        self.assertFalse(utils.valid_project_slug(project_slug))

        project_slug = "-!@#$%^&*()_+"
        self.assertFalse(utils.valid_project_slug(project_slug))

```

### Sample 9

```
<reponame>CrazyIvanPro/Optimal_Transport<filename>ADMM_primal.py<gh_stars>1-10
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =======================================
# File Name: ADMM_primal.py
# Purpose  : implementation for ADMM method
#            for solving primal problem
# =======================================

from utils import get_params
import numpy as np
import sys


def ADMM_primal(mu, nu, c, iters=10000, rho=1024, alpha=1.618):
    """ADMM_primal
    """
    # initialize
    m, n = c.shape
    pi = np.zeros((m, n))
    pi_dag = np.zeros((m, n))
    w = np.zeros((m, n))
    u = np.zeros(m)
    v = np.zeros(n)

    rho_tilde = rho * 32
    while rho_tilde >= rho:
        for _ in range(iters):
            r = ((-w + u.reshape((m, 1)) + v.reshape((1, n)) - c) / rho + 
                mu.reshape((m, 1)) + nu.reshape((1, n)) + pi_dag)
        
            pi = (r - ((r.sum(axis=1) - r.sum() / (m + n + 1)) / (n + 1)).reshape((m, 1))
                - ((r.sum(axis=0) - r.sum() / (m + n + 1)) / (m + 1)).reshape((1, n)))

            pi_dag = np.maximum(pi + w / rho, 0.0)

            u = u + alpha * rho * (mu - pi.sum(axis=1))
            v = v + alpha * rho * (nu - pi.sum(axis=0))
            w = w + alpha * rho * (pi - pi_dag)

            rho_tilde = rho_tilde / 2

        print('error_mu = %.5e' % np.linalg.norm(pi_dag.sum(axis = 1) - mu, 1))
        print('error_nu = %.5e' % np.linalg.norm(pi_dag.sum(axis = 0) - nu, 1))
        print('fvall    = %.5e' % (c * pi_da
... [truncated; full doc has 1,744 chars]
```

### Sample 10

```
<filename>application/mod_user/forms.py
from wtforms import Form, TextField, PasswordField, SelectField, TextAreaField, BooleanField, validators, ValidationError, RadioField
import re


phone_regex = "(\+\d+-?)?((\(?\d{3}\)?)|(\d{3}))-?\d{3}-?\d{4}$"

gender_choices = [
    ("", "Gender"),
    ("male", "Male"),
    ("female", "Female"),
    ("other", "Other"),
    ("rns", "Rather Not Say")
]

beginner_choices = [
    ("", "Are you a beginner?"),
    ("yes", "Yes"),
    ("no", "No")
]


ethnicity_choices = [
    ("", "Ethnicity"),
    ("white", "White"),
    ("african_american", "African American"),
    ("asian_pacific", "Asian or Pacific Islander"),
    ("american_indian_alaskan_native", "American Indian or Alaskan Native"),
    ("multiracial", "Multiracial"),
    ("hispanic", "Hispanic origin"),
    ("other", "Other"),
    ("rns", "Rather Not Say")
]

num_hackathons_choices = [
    ("", "How many hackathons have you been to?"),
    ("0", "0"),
    ("1", "1"),
    ("2", "2"),
    ("3", "3"),
    ("4", "4"),
    ("5", "5+")
]

num_hackathons_choices_mentor = [
    ("", "How many hackathons have you mentored at?"),
    ("0", "0"),
    ("1", "1"),
    ("2", "2"),
    ("3", "3"),
    ("4", "4"),
    ("5", "5+")
]

grade_choices = [
    ("", "What grade are you in?"),
    ("9", "9th"),
    ("10", "10th"),
    ("11", "11th"),
    ("12", "12th")
]

shirt_sizes = [
    ("", "What is your shirt size?"),
    ("XS", "Extra Small"),
    ("S", "Small"),
    ("M", "Medium"),
    ("L", "Lar
... [truncated; full doc has 19,558 chars]
```

---

## codeparrot/github-code Python (unfiltered GitHub crawl — closest to 'raw web code')

Source: `/fsx/users/dongweij/marin/outputs/raw/code_web.jsonl.gz`

### Sample 1

```
package org.liquidizer.snippet

import scala.xml._
import scala.xml.parsing._

import net.liftweb.util._
import net.liftweb.http._
import net.liftweb.http.js._
import net.liftweb.http.js.JsCmds._
import net.liftweb.common._

import org.liquidizer.model._

object Markup {

  val URL1= "(https?:/[^\\s\"]*[^\\s!?&.<>])"
  val URL2= "\\["+URL1+" ([^]]*)\\]"
  val URL_R= (URL1+"|"+URL2).r
 
  def renderComment(in : String) : NodeSeq = {
    if (in==null || in.length==0)
      NodeSeq.Empty
    else
      toXHTML(in)
  }

  def toXHTML(input : String) : NodeSeq = {
    try {
      val src= scala.io.Source.fromString("<span>"+input+"</span>")
      tidy(XhtmlParser(src).first.child, false)
    } catch {
      case e:FatalError => <p>{input}</p>
    }
  }
  
  def tidy(seq : NodeSeq, isLink : Boolean) : NodeSeq = 
    seq.flatMap { tidy(_, isLink) }

  def tidy(node : Node, isLink : Boolean) : NodeSeq = node match {
    case Elem(ns, tag, attr, scope, ch @ _*) =>
      tag match {
	case "img" | "em" | "i"  =>
	  val allowed= Set("src", "width", "height")
	  val fAttr= attr.filter { n=> allowed.contains(n.key) }
	  Elem(ns, tag, fAttr, scope, tidy(ch, true) :_*)
	case _ => Text(node.toString)
      }
    case Text(text) if !isLink => renderBlock(text.split("\n").toList)
    case _ => Text(node.toString)
  }

  def renderBlock(in : List[String]) : List[Node] = {
    def tail= renderBlock(in.tail)
    in match {
      case Nil => Nil
      case List(line, _*) if line.matches(" *[*-] .*"
... [truncated; full doc has 7,634 chars]
```

### Sample 2

```
import numpy as np
from numpy import cumsum, sum, searchsorted
from numpy.random import rand
import math
import utils
import core.sentence as sentence
import core.markovchain as mc
import logging

logger = logging.getLogger(__name__)

# Dialogue making class. Need to review where to return a string, where to return a list of tokens, etc.
# setters: list of speakers, pronouns, priors etc.
# random transitions
# Internal: build list of structures:
#     e.g.{:speaker_name "Alice", :speaker_pronoun "she", :speaker_str "she", :speech_verb "said", :position "end"}
# Then end with fn that maps that out to a suitable string
#     e.g. "<SPEECH>, she said."
# External bit then replaces <SPEECH> with a markov-chain-generated sentence (or several).


class dialogue_maker(object):
    """Class to handle creating dialogue based on a list of speakers and a sentence generator."""
    def __init__(self, names, pronouns, mc):
        self.speakers = [{"name": n, "pronoun": p} for n, p in list(zip(names, pronouns))]
        self._transitions = self.make_transition_probs()
        self._speech_acts = ["said", "whispered", "shouted", "cried"]
        self._acts_transitions = [25, 2, 2, 2]
        self.mc = mc
        # self.seeds = seeds
        self.target_len = np.random.randint(5, 50, size=len(names))  # rough words per sentence

    def make_transition_probs(self):
        """Make transition matrix between speakers, with random symmetric biases added in"""
        n = len(self.speakers)  # 
... [truncated; full doc has 5,911 chars]
```

### Sample 3

```
require 'spec_helper'

describe Fastball do
  it 'has a version number' do
    expect(Fastball::VERSION).not_to be nil
  end
end

```

### Sample 4

```
load File.expand_path("../target.rb", __FILE__)
module ActiveRecord::Magic
  class Param::Server < Param::Target
    
    def default_options
      { online:nil,
        wildcard:false, autocomplete:true,
        current_server:false, current_channel:false,
        users:false, channels:false, servers:true,
        ambigious: false }
    end
    
  end
end
```

### Sample 5

```
'use strict';
module.exports = function(sequelize, DataTypes) {
  var Student = sequelize.define('Student', {
    name: DataTypes.STRING,
    timeReq: DataTypes.INTEGER,
  }, {
    classMethods: {
      associate: function() {
      }
    }
  });
  return Student;
};

```

### Sample 6

```
/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package de.comci.bigbib;

import com.mongodb.BasicDBObject;
import com.mongodb.DBObject;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;
import org.bson.types.ObjectId;
import org.codehaus.jackson.annotate.JsonProperty;
import org.jbibtex.BibTeXEntry;
import org.jbibtex.Key;
import org.jbibtex.StringValue;
import org.jbibtex.Value;

/**
 *
 * @author Sebastian
 */
@XmlRootElement()
@XmlAccessorType(XmlAccessType.NONE)
public class PeristentBibTexEntry extends BibTeXEntry {

    private ObjectId id;
    
    public PeristentBibTexEntry(Key type, Key key) {
        super(type, key);
    }
    
    static Map<String, Key> keyMapping = new HashMap<String, Key>();
    static {
        keyMapping.put("address", BibTeXEntry.KEY_ADDRESS);
        keyMapping.put("annote", BibTeXEntry.KEY_ANNOTE);
        keyMapping.put("author", BibTeXEntry.KEY_AUTHOR);
        keyMapping.put("booktitle", BibTeXEntry.KEY_BOOKTITLE);
        keyMapping.put("chapter", BibTeXEntry.KEY_CHAPTER);
        keyMapping.put("crossref", BibTeXEntry.KEY_CROSSREF);
        keyMapping.put("doi", BibTeXEntry.KEY_DOI);
        keyMapping.put("edition", BibTeXEntry.KEY_EDITION);
        keyMapping.put("e
... [truncated; full doc has 4,524 chars]
```

### Sample 7

```
<?php

namespace ErenMustafaOzdal\LaravelModulesBase;

use Illuminate\Database\Eloquent\Model;

class Neighborhood extends Model
{
    /**
     * The database table used by the model.
     *
     * @var string
     */
    protected $table = 'neighborhoods';

    /**
     * The attributes that are mass assignable.
     *
     * @var array
     */
    protected $fillable = ['neighborhood'];
    public $timestamps = false;





    /*
    |--------------------------------------------------------------------------
    | Model Relations
    |--------------------------------------------------------------------------
    */

    /**
     * Get the postal code of the district.
     *
     * @return \Illuminate\Database\Eloquent\Relations\HasMany
     */
    public function postalCode()
    {
        return $this->hasOne('App\PostalCode');
    }





    /*
    |--------------------------------------------------------------------------
    | Model Set and Get Attributes
    |--------------------------------------------------------------------------
    */

    /**
     * get the neighborhood uc first
     *
     * @return string
     */
    public function getNeighborhoodUcFirstAttribute()
    {
        return ucfirst_tr($this->neighborhood);
    }
}

```

### Sample 8

```
package ch.bisi.koukan.job;

import ch.bisi.koukan.provider.XMLExchangeRatesProvider;
import ch.bisi.koukan.repository.DataAccessException;
import ch.bisi.koukan.repository.ExchangeRatesRepository;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamReader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

/**
 * Executes scheduled tasks for updating the in memory exchange rates
 * by querying the European Central Bank endpoints.
 */
@Component
public class ECBDataLoaderScheduler {

  private static final Logger logger = LoggerFactory.getLogger(ECBDataLoaderScheduler.class);

  private final XMLExchangeRatesProvider xmlExchangeRatesProvider;
  private final ExchangeRatesRepository exchangeRatesRepository;
  private final URL dailyEndpoint;
  private final URL pastDaysEndpoint;

  /**
   * Instantiates a new {@link ECBDataLoaderScheduler}.
   *
   * @param xmlExchangeRatesProvider the provider of exchange rates
   * @param exchangeRatesRepository the repository
   * @param dailyEndpoint the ECB daily endpoint {@link URL}
   * @param pastDaysEndpoint the ECB endpoint {@link URL} for retrieving past 
... [truncated; full doc has 3,982 chars]
```

### Sample 9

```
// Karma configuration
// http://karma-runner.github.io/0.10/config/configuration-file.html

module.exports = function(config) {
  config.set({
    // base path, that will be used to resolve files and exclude
    basePath: '',

    // testing framework to use (jasmine/mocha/qunit/...)
    frameworks: ['mocha', 'chai', 'sinon'],

    // list of files / patterns to load in the browser
    files: [
      'app/bower_components/angular/angular.js',
      'app/bower_components/angular-mocks/angular-mocks.js',
      'app/scripts/*.js',
      'app/scripts/**/*.js',
      'test/mock/**/*.js',
      'test/spec/**/*.js'
    ],

    // list of files / patterns to exclude
    exclude: [],

    // web server port
    port: 8080,

    // level of logging
    // possible values: LOG_DISABLE || LOG_ERROR || LOG_WARN || LOG_INFO || LOG_DEBUG
    logLevel: config.LOG_INFO,


    // enable / disable watching file and executing tests whenever any file changes
    autoWatch: false,


    // Start these browsers, currently available:
    // - Chrome
    // - ChromeCanary
    // - Firefox
    // - Opera
    // - Safari (only Mac)
    // - PhantomJS
    // - IE (only Windows)
    browsers: ['Chrome'],


    // Continuous Integration mode
    // if true, it capture browsers, run tests and exit
    singleRun: false
  });
};

```

### Sample 10

```
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>multiplier: Not compatible 👼</title>
    <link rel="shortcut icon" type="image/png" href="../../../../../favicon.png" />
    <link href="../../../../../bootstrap.min.css" rel="stylesheet">
    <link href="../../../../../bootstrap-custom.css" rel="stylesheet">
    <link href="//maxcdn.bootstrapcdn.com/font-awesome/4.2.0/css/font-awesome.min.css" rel="stylesheet">
    <script src="../../../../../moment.min.js"></script>
    <!-- HTML5 Shim and Respond.js IE8 support of HTML5 elements and media queries -->
    <!-- WARNING: Respond.js doesn't work if you view the page via file:// -->
    <!--[if lt IE 9]>
      <script src="https://oss.maxcdn.com/html5shiv/3.7.2/html5shiv.min.js"></script>
      <script src="https://oss.maxcdn.com/respond/1.4.2/respond.min.js"></script>
    <![endif]-->
  </head>
  <body>
    <div class="container">
      <div class="navbar navbar-default" role="navigation">
        <div class="container-fluid">
          <div class="navbar-header">
            <a class="navbar-brand" href="../../../../.."><i class="fa fa-lg fa-flag-checkered"></i> Coq bench</a>
          </div>
          <div id="navbar" class="collapse navbar-collapse">
            <ul class="nav navbar-nav">
              <li><a href="../..">clean / released</a></li>
              <li class="active"><a href="">8.9.1 / multiplier - 8.5.0</a></
... [truncated; full doc has 6,993 chars]
```

---

## OpenCodeReasoning full (problem + <think> trace + verified Python solution)

Source: `/fsx/users/dongweij/marin/outputs/raw/code_synth_full.jsonl.gz`

### Sample 1

```
Problem:
A string is called a k-string if it can be represented as k concatenated copies of some string. For example, the string "aabaabaabaab" is at the same time a 1-string, a 2-string and a 4-string, but it is not a 3-string, a 5-string, or a 6-string and so on. Obviously any string is a 1-string.

You are given a string s, consisting of lowercase English letters and a positive integer k. Your task is to reorder the letters in the string s in such a way that the resulting string is a k-string.
Input

The first input line contains integer k (1 ≤ k ≤ 1000). The second line contains s, all characters in s are lowercase English letters. The string length s satisfies the inequality 1 ≤ |s| ≤ 1000, where |s| is the length of string s.

Output

Rearrange the letters in string s in such a way that the result is a k-string. Print the result on a single output line. If there are multiple solutions, print any of them.

If the solution doesn't exist, print "-1" (without quotes).

Examples

Input

2
aazz


Output

azaz


Input

3
abcabcabz


Output

-1

Reasoning:
<think>
Okay, let's see. I need to solve this problem where I have to rearrange the characters of a string so that the resulting string is a k-string. Hmm. So a k-string is one that can be divided into k equal parts, each part being the same. Like, if k is 2, the string should be two copies of some substring.

First, I need to check if it's possible to form such a string. If not, output -1. Otherwise, rearrange the characters
... [truncated; full doc has 12,942 chars]
```

### Sample 2

```
Problem:
Artem is building a new robot. He has a matrix a consisting of n rows and m columns. The cell located on the i-th row from the top and the j-th column from the left has a value a_{i,j} written in it. 

If two adjacent cells contain the same value, the robot will break. A matrix is called good if no two adjacent cells contain the same value, where two cells are called adjacent if they share a side. 

Artem wants to increment the values in some cells by one to make a good.

More formally, find a good matrix b that satisfies the following condition — 

  * For all valid (i,j), either b_{i,j} = a_{i,j} or b_{i,j} = a_{i,j}+1. 



For the constraints of this problem, it can be shown that such a matrix b always exists. If there are several such tables, you can output any of them. Please note that you do not have to minimize the number of increments.
Input

Each test contains multiple test cases. The first line contains the number of test cases t (1 ≤ t ≤ 10). Description of the test cases follows.

The first line of each test case contains two integers n, m (1 ≤ n ≤ 100, 1 ≤ m ≤ 100) — the number of rows and columns, respectively.

The following n lines each contain m integers. The j-th integer in the i-th line is a_{i,j} (1 ≤ a_{i,j} ≤ 10^9).

Output

For each case, output n lines each containing m integers. The j-th integer in the i-th line is b_{i,j}.

Example

Input


3
3 2
1 2
4 5
7 8
2 2
1 1
3 3
2 2
1 3
2 2


Output


1 2
5 6
7 8
2 1
4 3
2 4
3 2

Note

In all the cas
... [truncated; full doc has 30,582 chars]
```

### Sample 3

```
Problem:
problem

There are $ V $ islands, numbered $ 0, 1, ..., V-1 $, respectively. There are $ E $ bridges, numbered $ 0, 1, ..., E-1 $, respectively. The $ i $ th bridge spans island $ s_i $ and island $ t_i $ and is $ c_i $ wide.

The AOR Ika-chan Corps (commonly known as the Squid Corps), which has a base on the island $ 0 $, is small in scale, so it is always being killed by the Sosu-Usa Corps (commonly known as the Sosuusa Corps) on the island $ V-1 $. .. One day, the squid group got the information that "the Sosusa group will attack tomorrow." A large number of Sosusa's subordinates move across the island from the Sosusa's base, and if even one of their subordinates reaches the Squid's base, the Squid will perish ...

Therefore, the squid group tried to avoid the crisis by putting a road closure tape on the bridge to prevent it from passing through. The length of the tape used is the sum of the widths of the closed bridges. Also, if all the subordinates of the Sosusa group always pass through a bridge with a width of $ 1 $, the squid group can ambush there to prevent them from passing through the Sosusa group.

The length of the tape it holds is $ 10 ^ 4 $. Considering the future, I would like to consume as little tape as possible. Find the minimum length of tape used to keep Sosusa's subordinates from reaching the squid's base. However, if the tape is not long enough and you are attacked by all means, output $ -1 $.



output

Output the minimum length of tape used 
... [truncated; full doc has 47,213 chars]
```

### Sample 4

```
Problem:
Masha really loves algebra. On the last lesson, her strict teacher Dvastan gave she new exercise.

You are given geometric progression b defined by two integers b1 and q. Remind that a geometric progression is a sequence of integers b1, b2, b3, ..., where for each i > 1 the respective term satisfies the condition bi = bi - 1·q, where q is called the common ratio of the progression. Progressions in Uzhlyandia are unusual: both b1 and q can equal 0. Also, Dvastan gave Masha m "bad" integers a1, a2, ..., am, and an integer l.

Masha writes all progression terms one by one onto the board (including repetitive) while condition |bi| ≤ l is satisfied (|x| means absolute value of x). There is an exception: if a term equals one of the "bad" integers, Masha skips it (doesn't write onto the board) and moves forward to the next term.

But the lesson is going to end soon, so Masha has to calculate how many integers will be written on the board. In order not to get into depression, Masha asked you for help: help her calculate how many numbers she will write, or print "inf" in case she needs to write infinitely many integers.
Input

The first line of input contains four integers b1, q, l, m (-109 ≤ b1, q ≤ 109, 1 ≤ l ≤ 109, 1 ≤ m ≤ 105) — the initial term and the common ratio of progression, absolute value of maximal number that can be written on the board and the number of "bad" integers, respectively.

The second line contains m distinct integers a1, a2, ..., am (-109 ≤ ai ≤ 109)
... [truncated; full doc has 53,190 chars]
```

### Sample 5

```
Problem:
Mike has a frog and a flower. His frog is named Xaniar and his flower is named Abol. Initially(at time 0), height of Xaniar is h1 and height of Abol is h2. Each second, Mike waters Abol and Xaniar.

<image>

So, if height of Xaniar is h1 and height of Abol is h2, after one second height of Xaniar will become <image> and height of Abol will become <image> where x1, y1, x2 and y2 are some integer numbers and <image> denotes the remainder of a modulo b.

Mike is a competitive programmer fan. He wants to know the minimum time it takes until height of Xania is a1 and height of Abol is a2.

Mike has asked you for your help. Calculate the minimum time or say it will never happen.
Input

The first line of input contains integer m (2 ≤ m ≤ 106).

The second line of input contains integers h1 and a1 (0 ≤ h1, a1 < m).

The third line of input contains integers x1 and y1 (0 ≤ x1, y1 < m).

The fourth line of input contains integers h2 and a2 (0 ≤ h2, a2 < m).

The fifth line of input contains integers x2 and y2 (0 ≤ x2, y2 < m).

It is guaranteed that h1 ≠ a1 and h2 ≠ a2.

Output

Print the minimum number of seconds until Xaniar reaches height a1 and Abol reaches height a2 or print -1 otherwise.

Examples

Input

5
4 2
1 1
0 1
2 3


Output

3


Input

1023
1 2
1 0
1 2
1 1


Output

-1

Note

In the first sample, heights sequences are following:

Xaniar: <image>

Abol: <image>

Reasoning:
<think>
Okay, let's try to figure out this problem. So, the problem is about Mike's frog and
... [truncated; full doc has 60,108 chars]
```

### Sample 6

```
Problem:
The number obtained by multiplying 1 by 2, 3, 5 several times (0 or more times) is called the Hamming numbers. For example

* 1
* 1 x 2 x 2 = 4
* 1 x 2 x 2 x 3 x 5 x 5 = 300



Etc. are humming numbers, but 11, 13, 14 etc. are not humming numbers.

All humming numbers are divisible by a power of 60 (for example, 54 is divisible by 603 = 21600), so they have long been known as convenient numbers for sexagesimal calculations such as time. In just intonation, which is one of the scales used for tuning musical instruments, the ratio of the frequencies of the sounds is a sequence of humming numbers of 24, 27, 30, 32, 36, 40, 45, 48.

Create a program that takes integers m and n as
inputs and outputs the number of humming numbers that are m or more and n or less.



Input

A sequence of multiple datasets is given as input. The end of the input is indicated by a single line of zeros.

For each dataset, two integers m and n (1 ≤ m, n ≤ 1000000, m ≤ n) are given on one line, separated by blanks.

The number of datasets does not exceed 20.

Output

Outputs the number of humming numbers from m to n for each data set on one line.

Example

Input

3 8
1 27
1 86
0


Output

5
17
31

Reasoning:
<think>
Okay, I need to solve this problem where I have to count the number of Hamming numbers between m and n inclusive. Hmm, Hamming numbers are numbers that can be formed by multiplying 2, 3, and 5 any number of times, including zero. So like 1 is 2^0 *3^0 *5^0. Then numbers like 2,3,5, 2
... [truncated; full doc has 22,498 chars]
```

### Sample 7

```
Problem:
Example
Input

3
y 7
y 6
n 5


Output

1

Reasoning:
<think>
Okay, let's see. The problem says that we need to process some input and compute an output based on certain conditions. Let me read the example given to understand better.

The input example is:
3
y 7
y 6
n 5

And the output is 1. Hmm. So, three lines of input, each with a character (y or n) and a number. The output is 1 here. Let's figure out what the rule is here.

Wait, maybe the problem is about counting the number of times the 'y' entries have a value that's higher than all the 'n' entries? Let's see. In the example, the 'y' entries are 7 and 6, and the 'n' is 5. The highest 'y' is 7, which is higher than 5, so maybe each 'y' that's higher than all 'n's contributes to the count. Wait, but how does that sum up to 1? Because in the example, there are two 'y's. Oh, perhaps it's the number of 'y's that are strictly greater than all 'n's. Let's check.

In the example, the 'n' is 5. The 'y's are 7 and 6. Both are greater than 5. So if the answer is 2, but the output is 1. So that can't be right.

Hmm, maybe there's a different rule. Let's think again. Maybe it's the number of 'y's that are greater than all the 'n's, but each 'y' is considered, and if there are multiple 'y's, perhaps only the maximum 'y' is considered. Wait, that doesn't fit either.

Wait, the output in the example is 1. Let's look for another approach.

Alternatively, perhaps the problem is to find the maximum 'y' value and subtract the maxi
... [truncated; full doc has 5,989 chars]
```

### Sample 8

```
Problem:
It is lunch time for Mole. His friend, Marmot, prepared him a nice game for lunch.

Marmot brought Mole n ordered piles of worms such that i-th pile contains ai worms. He labeled all these worms with consecutive integers: worms in first pile are labeled with numbers 1 to a1, worms in second pile are labeled with numbers a1 + 1 to a1 + a2 and so on. See the example for a better understanding.

Mole can't eat all the worms (Marmot brought a lot) and, as we all know, Mole is blind, so Marmot tells him the labels of the best juicy worms. Marmot will only give Mole a worm if Mole says correctly in which pile this worm is contained.

Poor Mole asks for your help. For all juicy worms said by Marmot, tell Mole the correct answers.
Input

The first line contains a single integer n (1 ≤ n ≤ 105), the number of piles.

The second line contains n integers a1, a2, ..., an (1 ≤ ai ≤ 103, a1 + a2 + ... + an ≤ 106), where ai is the number of worms in the i-th pile.

The third line contains single integer m (1 ≤ m ≤ 105), the number of juicy worms said by Marmot.

The fourth line contains m integers q1, q2, ..., qm (1 ≤ qi ≤ a1 + a2 + ... + an), the labels of the juicy worms.

Output

Print m lines to the standard output. The i-th line should contain an integer, representing the number of the pile where the worm labeled with the number qi is.

Examples

Input

5
2 7 3 4 9
3
1 25 11


Output

1
5
3

Note

For the sample input:

  * The worms with labels from [1, 2] are in the first pi
... [truncated; full doc has 8,305 chars]
```

### Sample 9

```
Problem:
The only difference between easy and hard versions is constraints.

A session has begun at Beland State University. Many students are taking exams.

Polygraph Poligrafovich is going to examine a group of n students. Students will take the exam one-by-one in order from 1-th to n-th. Rules of the exam are following:

  * The i-th student randomly chooses a ticket. 
  * if this ticket is too hard to the student, he doesn't answer and goes home immediately (this process is so fast that it's considered no time elapses). This student fails the exam. 
  * if the student finds the ticket easy, he spends exactly t_i minutes to pass the exam. After it, he immediately gets a mark and goes home. 



Students take the exam in the fixed order, one-by-one, without any interruption. At any moment of time, Polygraph Poligrafovich takes the answer from one student.

The duration of the whole exam for all students is M minutes (max t_i ≤ M), so students at the end of the list have a greater possibility to run out of time to pass the exam.

For each student i, you should count the minimum possible number of students who need to fail the exam so the i-th student has enough time to pass the exam.

For each student i, find the answer independently. That is, if when finding the answer for the student i_1 some student j should leave, then while finding the answer for i_2 (i_2>i_1) the student j student does not have to go home.
Input

The first line of the input contains two integers n and M
... [truncated; full doc has 20,031 chars]
```

### Sample 10

```
Problem:
Little X has met the following problem recently. 

Let's define f(x) as the sum of digits in decimal representation of number x (for example, f(1234) = 1 + 2 + 3 + 4). You are to calculate <image>

Of course Little X has solved this problem quickly, has locked it, and then has tried to hack others. He has seen the following C++ code: 
    
    
      
        ans = solve(l, r) % a;  
        if (ans <= 0)  
          ans += a;  
      
    

This code will fail only on the test with <image>. You are given number a, help Little X to find a proper test for hack.
Input

The first line contains a single integer a (1 ≤ a ≤ 1018).

Output

Print two integers: l, r (1 ≤ l ≤ r < 10200) — the required test data. Leading zeros aren't allowed. It's guaranteed that the solution exists.

Examples

Input

46


Output

1 10


Input

126444381000032


Output

2333333 2333333333333

Reasoning:
<think>
Okay, I need to find two numbers l and r such that when the function solve(l, r) is called, the result modulo a is a certain value, and when adjusted, it becomes negative, causing the code to fail. But wait, the problem says that the code fails only on a test where the sum of f(x) for x from l to r modulo a is such that when they compute ans = solve(l, r) % a, if ans <=0, they add a. So the code's ans would be incorrect in a case where the actual modulo is zero. Because when the sum is a multiple of a, then ans would be 0, but the code checks if ans <=0 and then adds a. So in that case,
... [truncated; full doc has 43,787 chars]
```

---

## OpenCodeReasoning solution-only (just the verified Python)

Source: `/fsx/users/dongweij/marin/outputs/raw/code_synth_solution.jsonl.gz`

### Sample 1

```
import sys

def main():
    t = int(sys.stdin.readline())
    for _ in range(t):
        n = int(sys.stdin.readline())
        for _ in range(n-1):
            sys.stdin.readline()  # skip edges
        k1 = int(sys.stdin.readline())
        x_list = list(map(int, sys.stdin.readline().split()))
        k2 = int(sys.stdin.readline())
        y_list = list(map(int, sys.stdin.readline().split()))
        y_set = set(y_list)
        found = False
        
        # Check up to 3 nodes from my subtree
        for x in x_list[:3]:
            print(f"A {x}")
            sys.stdout.flush()
            y = int(sys.stdin.readline())
            if y == -1:
                exit(0)  # invalid query or too many
            if y in y_set:
                print(f"C {x}")
                sys.stdout.flush()
                found = True
                break
        if found:
            continue
        
        x_set = set(x_list)
        # Check up to 2 nodes from Li Chen's subtree
        for y in y_list[:2]:
            print(f"B {y}")
            sys.stdout.flush()
            x = int(sys.stdin.readline())
            if x == -1:
                exit(0)
            if x in x_set:
                print(f"C {x}")
                sys.stdout.flush()
                found = True
                break
        if not found:
            print("C -1")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
```

### Sample 2

```
n = int(input())
total = 0
for a in range(1, n):
    total += (n - 1) // a
print(total)
```

### Sample 3

```
import sys

def main():
    t = int(sys.stdin.readline())
    for _ in range(t):
        n = int(sys.stdin.readline())
        count = bin(n).count('1')
        print("Sherlock" if count % 2 == 1 else "Watson")

if __name__ == "__main__":
    main()
```

### Sample 4

```
import heapq

n = int(input())
primes = list(map(int, input().split()))
k = int(input())

heap = []
heapq.heappush(heap, (1, 0))

count = 0
result = 0

while heap:
    current_val, last_idx = heapq.heappop(heap)
    count += 1
    if count == k:
        result = current_val
        break
    for i in range(last_idx, n):
        next_val = current_val * primes[i]
        heapq.heappush(heap, (next_val, i))

print(result)
```

### Sample 5

```
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        n = int(input[ptr])
        ptr += 1
        D = list(map(int, input[ptr:ptr+n]))
        ptr += n
        sources = []
        dests = []
        for i in range(n):
            d = D[i]
            if d > 0:
                sources.append([i, d])
            elif d < 0:
                dests.append([i, -d])
        sources.sort()
        dests.sort()
        total = 0
        i = j = 0
        while i < len(sources) and j < len(dests):
            s_pos, s_cnt = sources[i]
            d_pos, d_cnt = dests[j]
            min_cnt = min(s_cnt, d_cnt)
            total += min_cnt * abs(s_pos - d_pos)
            sources[i][1] -= min_cnt
            dests[j][1] -= min_cnt
            if sources[i][1] == 0:
                i += 1
            if dests[j][1] == 0:
                j += 1
        print(total)

if __name__ == "__main__":
    main()
```

### Sample 6

```
import bisect

n, m, k = map(int, input().split())

subjects = []
for idx in range(m):
    a, b, c = map(int, input().split())
    subjects.append((c, a, b, idx + 1))

subjects.sort()
sorted_c = [s[0] for s in subjects]

dp = [{} for _ in range(n + 1)]

# Initialize day 1
for i in range(len(subjects)):
    c_i, a_i, b_i, idx_i = subjects[i]
    for x in range(a_i, b_i + 1):
        key = (i, x)
        dp[1][key] = (x, None)

# Fill dp for days 2 to n
for d in range(2, n + 1):
    for (i_prev, x_prev), (sum_prev, _) in dp[d-1].items():
        c_prev = subjects[i_prev][0]
        j_start = bisect.bisect_right(sorted_c, c_prev)
        for j in range(j_start, len(subjects)):
            c_j, a_j, b_j, idx_j = subjects[j]
            # Check addition
            x_plus = x_prev + k
            if a_j <= x_plus <= b_j:
                key = (j, x_plus)
                new_sum = sum_prev + x_plus
                if key not in dp[d] or new_sum > dp[d].get(key, (-float('inf'),))[0]:
                    dp[d][key] = (new_sum, (i_prev, x_prev))
            # Check multiplication
            x_mult = x_prev * k
            if a_j <= x_mult <= b_j:
                key = (j, x_mult)
                new_sum = sum_prev + x_mult
                if key not in dp[d] or new_sum > dp[d].get(key, (-float('inf'),))[0]:
                    dp[d][key] = (new_sum, (i_prev, x_prev))

if not dp[n]:
    print("NO")
else:
    max_sum = -float('inf')
    best_state = None
    for key, (s, prev) in dp[n]
... [truncated; full doc has 2,014 chars]
```

### Sample 7

```
import math

m = int(input())
s = list(map(int, input().split()))
g = s[0]

# Check if all elements are multiples of g
for num in s:
    if num % g != 0:
        print(-1)
        exit()

# Transform to T by dividing each element by g
t = [num // g for num in s]
t_max = t[-1]

# Check if all elements in T divide t_max
for num in t:
    if t_max % num != 0:
        print(-1)
        exit())

# Check if the first element of T is 1
if t[0] != 1:
    print(-1)
    exit()

# Prepare candidates (T excluding 1)
candidates = [x for x in t if x != 1]
found = False

# Check if any pair in candidates has GCD 1
for i in range(len(candidates)):
    for j in range(i + 1, len(candidates)):
        if math.gcd(candidates[i], candidates[j]) == 1:
            found = True
            break
    if found:
        break

# Determine transformed sequence
if found:
    transformed = candidates
else:
    transformed = t

# Generate original sequence by multiplying back with g
original = [x * g for x in transformed]

# Output the result
print(len(original))
print(' '.join(map(str, original)))
```

### Sample 8

```
n = int(input())
divisors = 0
for i in range(1, int(n**0.5) + 1):
    if n % i == 0:
        if i * i == n:
            divisors += 1
        else:
            divisors += 2
print(divisors - 1)
```

### Sample 9

```
T, S, q = map(int, input().split())
count = 0
while S < T:
    S *= q
    count += 1
print(count)
```

### Sample 10

```
n, k = map(int, input().split())
a = list(map(int, input().split()))

if k == 1:
    print(sum(a))
else:
    suffix_sums = [0] * n
    suffix_sums[-1] = a[-1]
    for i in range(n-2, -1, -1):
        suffix_sums[i] = a[i] + suffix_sums[i+1]
    possible = suffix_sums[1:]
    possible.sort(reverse=True)
    sum_split = sum(possible[:k-1])
    print(sum(a) + sum_split)
```

---

## OpenCoder algorithmic_corpus (== opc_algorithmic; MBPP-style 'write a python function')

Source: `/fsx/users/dongweij/marin/outputs/raw/opc_algorithmic.jsonl.gz`

### Sample 1

```

Write a function to find the maximum sum of a subarray with the length of exactly k.

```python
def find_max_sum(arr, k):
    if not arr or k <= 0 or k > len(arr):
        return 0
    max_sum = current_sum = sum(arr[:k])
    for i in range(k, len(arr)):
        current_sum += arr[i] - arr[i-k]
        max_sum = max(max_sum, current_sum)
    return max_sum
```

assert find_max_sum([2,3,4,1,5],2) == 7
assert find_max_sum([1,4,2,10,23,3,1,0,20],4) == 39

```

### Sample 2

```

Write a python function to find the maximum number of consecutive 1's in the binary representation of a given number.

assert max_consecutive_ones(139) == 2
assert max_consecutive_ones(15) == 4

```python
def max_consecutive_ones(n):
    binary = bin(n)[2:]
    max_count = 0
    current_count = 0
    for digit in binary:
        if digit == '1':
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0
    return max_count
```

```

### Sample 3

```

Write a Python function to find the longest common prefix string amongst an array of strings. If there is no common prefix, return an empty string "".

assert longestCommonPrefix(["flower","flow","flight"]) == "fl"
assert longestCommonPrefix(["dog","racecar","car"]) == ""
assert longestCommonPrefix(["interspecies","interstellar","interstate"]) == "inters"

```python
def longestCommonPrefix(strs):
    if not strs:
        return ""
    shortest_str = min(strs, key=len)
    for i, char in enumerate(shortest_str):
        for other in strs:
            if other[i] != char:
                return shortest_str[:i]
    return shortest_str
```

```

### Sample 4

```

Write a function to determine if a given string is a valid bracket sequence. A valid bracket sequence consists of parentheses, square brackets, and curly braces that are correctly matched and nested. For example, "([]{})" is a valid sequence while "([)]" is not.

```python
def is_valid_bracket_sequence(s):
    stack = []
    bracket_map = {')': '(', ']': '[', '}': '{'}

    for char in s:
        if char in bracket_map.values():
            stack.append(char)
        elif char in bracket_map.keys():
            if stack == [] or bracket_map[char] != stack.pop():
                return False
        else:
            return False

    return stack == []
```

assert is_valid_bracket_sequence("([]{})") == True
assert is_valid_bracket_sequence("({[)]") == False
assert is_valid_bracket_sequence("{{[[(())]]}}") == True
assert is_valid_bracket_sequence("((()))") == True
assert is_valid_bracket_sequence("([)]") == False

```

### Sample 5

```

Write a python function to check if a given string is an anagram of another string.

```python
def is_anagram(str1, str2):
    return sorted(str1) == sorted(str2)
```

assert is_anagram('rat', 'car') == False
assert is_anagram('listen', 'silent') == True
assert is_anagram('triangle', 'integral') == True
assert is_anagram('apple', 'papel') == True

```

### Sample 6

```

Write a function to find the kth smallest element in an unsorted array.

```python
def kth_smallest(arr, k):
    """
    Find the kth smallest element in an unsorted array.
    
    :param arr: List[int] -- unsorted array
    :param k: int -- order of the smallest element to find
    :return: int -- kth smallest element
    """
    # Sort the array
    arr.sort()
    
    # Return the kth element in the sorted array
    return arr[k-1]
```

assert kth_smallest([3, 2, 1, 5, 6, 4], 2) == 2

```

### Sample 7

```

Write a python function to find the maximum sum of a subarray with the length of exactly k.

assert max_sum_subarray([2,1,5,1,3,2],3) == 9
assert max_sum_subarray([1,1,1,1,1],3) == 3
assert max_sum_subarray([2,3,4,1,5],2) == 7

```python
def max_sum_subarray(arr,k):
    if len(arr) < k:
        return "Invalid operation"
    maxSum = sum(arr[:k])
    windowSum = maxSum
    for i in range(len(arr) - k):
        windowSum = windowSum - arr[i] + arr[i+k]
        maxSum = max(maxSum, windowSum)
    return maxSum
```

```

### Sample 8

```

Write a function that takes a list of integers as input and returns the sum of all unique pairs in the list. A unique pair is defined as two different elements from the list, and each pair should be counted only once, regardless of the order of elements.
For example, if the input list is [1, 2, 3], the function should return the sum of (1, 2), (1, 3), and (2, 3), which is 1 + 2 + 1 + 3 + 2 + 3 = 12.

```python
def unique_pair_sum(nums):
    total_sum = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            total_sum += nums[i] + nums[j]
    return total_sum
```

assert unique_pair_sum([1, 2, 3]) == 12
assert unique_pair_sum([4, 4, 4]) == 24
assert unique_pair_sum([1, 2, 3, 4]) == 30

```

### Sample 9

```

Write a function to find the maximum subarray sum in a given list of integers using Kadane's algorithm.

assert max_subarray_sum([4,-1,-4,5]) == 5
assert max_subarray_sum([-2,1,-3,4,-1,2,1,-5,4]) == 6
assert max_subarray_sum([-1,-2,-3,-4]) == -1

```python
def max_subarray_sum(nums):
    if len(nums) == 0:
        return 0
    res = nums[0]
    currMax = 0
    for n in nums:
        if currMax + n < 0:
            currMax = 0
            res = max(n, res)
        else:
            currMax += n
            res = max(currMax, res)
    return res
```

```

### Sample 10

```

Write a function to calculate the total cost of painting houses under certain rules. The rule is that each house must be painted in a different color than its neighbors. The cost of painting each house in each color is given. The function should return the minimum cost to paint all the houses.

```python
def min_cost_paint(costs):
    for i in range(1, len(costs)):
        costs[i][0] += min(costs[i-1][1], costs[i-1][2])
        costs[i][1] += min(costs[i-1][0], costs[i-1][2])
        costs[i][2] += min(costs[i-1][0], costs[i-1][1])
    return min(costs[-1])
```

assert min_cost_paint([[17,2,17],[16,16,5]]) == 7
assert min_cost_paint([[17]]) == 17
# Test case 1: Only one house, which can be painted with any color


# Test case 3: Multiple houses, the minimum cost to paint them with different colors from their neighbors
# Test case 2: Two houses, the minimum cost to paint them with different colors

```

---

## OpenCoder-LLM / opc-annealing-corpus / synthetic_code_snippet (phi-style textbook code)

Source: `OpenCoder-LLM/opc-annealing-corpus` config `synthetic_code_snippet` (HF streaming)

### Sample 1

```
/**
 * Given an array of daily temperatures T, returns an array such that, for each day in the input,
 * it tells you how many days you would have to wait until a warmer temperature. If there is no
 * future day for which this is possible, it puts 0 instead.
 *
 * For example, given the array of temperatures T = [73, 74, 75, 71, 69, 72, 76, 73],
 * the output should be [1, 1, 4, 2, 1, 1, 0, 0].
 *
 * Note:
 * The length of temperatures will be in the range [1, 30000].
 * Each temperature will be an integer in the range [30, 100].
 */
function dailyTemperatures(temperatures) {
    // Initialize the result array with zeros, same length as the input array
    let result = new Array(temperatures.length).fill(0);
    
    // Initialize a stack to keep track of the indices of temperatures
    let stack = [];
    
    // Iterate through the array of temperatures
    for (let i = 0; i < temperatures.length; i++) {
        let temp = temperatures[i];
        // While the stack is not empty and the current temperature is greater than the temperature at the index on the top of the stack
        while (stack.length > 0 && temperatures[stack[stack.length - 1]] < temp) {
            // Pop the index from the stack
            let prevIndex = stack.pop();
            // Update the result for the previous index with the difference between the current index and the previous index
            result[prevIndex] = i - prevIndex;
        }
        
        // Push the current index onto the stack
... [truncated; full doc has 1,587 chars]
```

### Sample 2

```
[BEGIN OF PHP CODE]
<?php

/**
 * Given an array of daily temperatures T, return an array such that, for each day in the input,
 * tells you how many days you would have to wait until a warmer temperature. If there is no
 * future day for which this is possible, put 0 instead.
 *
 * For example, given the array of temperatures T = [73, 74, 75, 71, 69, 72, 76, 73],
 * your output should be [1, 1, 4, 2, 1, 1, 0, 0].
 *
 * Note:
 * The length of temperatures will be in the range [1, 30000].
 * Each temperature will be an integer in the range [30, 100].
 */
function daily_temperatures($temperatures) {
    // Initialize the result array with zeros, same length as the input array
    $result = array_fill(0, count($temperatures), 0);
    
    // Initialize a stack to keep track of the indices of temperatures
    $stack = [];
    
    // Iterate through the array of temperatures
    for ($i = 0; $i < count($temperatures); $i++) {
        // While the stack is not empty and the current temperature is greater than the temperature at the index on the top of the stack
        while (!empty($stack) && $temperatures[end($stack)] < $temperatures[$i]) {
            // Pop the index from the stack
            $prev_index = array_pop($stack);
            // Update the result for the previous index with the difference between the current index and the previous index
            $result[$prev_index] = $i - $prev_index;
        }
        
        // Push the current index onto the stack
        a
... [truncated; full doc has 1,610 chars]
```

### Sample 3

```
#!/bin/bash

# Given a list of daily temperatures, return a list such that, for each day in the input,
# tells you how many days you would have to wait until a warmer temperature. If there is no
# future day for which this is possible, put 0 instead.
# Examples:
# >>> $(daily_temperatures "73 74 75 71 69 72 76 73")
# "1 1 4 2 1 1 0 0"

daily_temperatures() {
    local temperatures=($1)
    local -a result
    local -a stack=()
    local i temp

    # Initialize the result array with zeros, same length as the input list
    for (( i=0; i<${#temperatures[@]}; i++ )); do
        result[i]=0
    done

    # Iterate through the list of temperatures
    for (( i=0; i<${#temperatures[@]}; i++ )); do
        temp=${temperatures[i]}

        # While the stack is not empty and the current temperature is greater than the temperature at the index on the top of the stack
        while [ ${#stack[@]} -gt 0 ] && [ ${temperatures[${stack[-1]}]} -lt $temp ]; do
            # Pop the index from the stack
            prev_index=${stack[-1]}
            unset stack[-1]
            stack=("${stack[@]}")

            # Update the result for the previous index with the difference between the current index and the previous index
            result[$prev_index]=$((i - prev_index))
        done

        # Push the current index onto the stack
        stack+=($i)
    done

    # Return the result list
    echo "${result[@]}"
}

```

### Sample 4

```
import java.util.*;

class Problem {
    /**
     * This function aims to find the majority element in a given array.
     * The majority element is defined as the element that appears more than n/2 times in the array.
     * The function assumes that the array is non-empty and the majority element always exists in the array.
     * The function iterates through the array and uses a voting algorithm to find the majority element.
     *
     * Note:
     * * The array can contain both positive and negative integers.
     *
     * Examples:
     * * findMajorityElement(new int[]{3, 3, 4, 2, 4, 4, 2, 4, 4}) => 4
     * * findMajorityElement(new int[]{2, 2, 1, 1, 1, 2, 2}) => 2
     *
     * @param arr the input array of integers
     * @return the majority element
     */
    public static int findMajorityElement(int[] arr) {
        int count = 0;
        int result = 0;
        for (int num : arr) {
            if (count == 0) {
                result = num;
                count += 1;
            } else if (num == result) {
                count += 1;
            } else {
                count -= 1;
            }
        }
        return result;
    }
}

```

### Sample 5

```
#include <iostream>
#include <vector>
#include <cassert>

// This function aims to find the majority element in a given array.
// The majority element is defined as the element that appears more than n/2 times in the array.
// The function assumes that the array is non-empty and the majority element always exists in the array.
// The function iterates through the array and uses a voting algorithm to find the majority element.

int find_majority_element(const std::vector<int>& arr) {
    int count = 0;
    int result = 0;

    for (int num : arr) {
        if (count == 0) {
            result = num;
            count += 1;
        } else if (num == result) {
            count += 1;
        } else {
            count -= 1;
        }
    }

    return result;
}

// Test cases to verify the correctness of the function.
void test_find_majority_element() {
    assert(find_majority_element({3, 3, 4, 2, 4, 4, 2, 4, 4}) == 4);
    assert(find_majority_element({2, 2, 1, 1, 1, 2, 2}) == 2);
}

int main() {
    test_find_majority_element();
    std::cout << "All tests passed successfully." << std::endl;
    return 0;
}

```

### Sample 6

```
using System.Security.Cryptography;
using System.Text;
using System.Numerics;
using System.Diagnostics;
using System.Linq;
using System.Collections.Generic;
using System;

class Problem {
    // This function aims to find the majority element in a given array.
    // The majority element is defined as the element that appears more than n/2 times in the array.
    // The function assumes that the array is non-empty and the majority element always exists in the array.
    // The function iterates through the array and uses a voting algorithm to find the majority element.
    public static int FindMajorityElement(int[] arr) {
        // Initialize a count variable to keep track of the current candidate for majority element
        // and a result variable to store the majority element found.
        int count = 0;
        int result = 0;

        // Iterate through each number in the array.
        foreach (int num in arr) {
            // If the count is 0, it means we haven't found a candidate yet or the current candidate is not num.
            // So, we set the result to num and increment the count.
            if (count == 0) {
                result = num;
                count += 1;
            }
            // If the current number is the same as the result (i.e., the current candidate), increment the count.
            else if (num == result) {
                count += 1;
            }
            // If the current number is different from the result, decrement the coun
... [truncated; full doc has 1,706 chars]
```

### Sample 7

```
// This function aims to find the majority element in a given array.
// The majority element is defined as the element that appears more than n/2 times in the array.
// The function assumes that the array is non-empty and the majority element always exists in the array.
// The function iterates through the array and uses a voting algorithm to find the majority element.

function findMajorityElement(arr: number[]): number {
    /**
     * Given a non-empty array of integers, where the majority element is the element that appears more than n/2 times,
     * and you may assume the array is non-empty and the majority element always exist in the array.
     * This function finds the majority element in the array.

     * The array can contain both positive and negative integers.

     * Examples:
     * findMajorityElement([3, 3, 4, 2, 4, 4, 2, 4, 4]) => 4
     * findMajorityElement([2, 2, 1, 1, 1, 2, 2]) => 2

     */

    // Initialize a count variable to keep track of the current candidate for majority element
    // and a result variable to store the majority element found.
    let count: number = 0;
    let result: number = 0;

    // Iterate through each number in the array.
    for (let num of arr) {
        // If the count is 0, it means we haven't found a candidate yet or the current candidate is not num.
        // So, we set the result to num and increment the count.
        if (count === 0) {
            result = num;
            count += 1;
        }
        // If the
... [truncated; full doc has 1,920 chars]
```

### Sample 8

```
/**
 * This function aims to find the majority element in a given array.
 * The majority element is defined as the element that appears more than n/2 times in the array.
 * The function assumes that the array is non-empty and the majority element always exists in the array.
 * The function iterates through the array and uses a voting algorithm to find the majority element.
 *
 * Note:
 * * The array can contain both positive and negative integers.
 *
 * Examples:
 * * findMajorityElement([3, 3, 4, 2, 4, 4, 2, 4, 4]) => 4
 * * findMajorityElement([2, 2, 1, 1, 1, 2, 2]) => 2
 */
function findMajorityElement(arr) {
    let count = 0;
    let result = 0;

    for (let num of arr) {
        if (count === 0) {
            result = num;
            count += 1;
        } else if (num === result) {
            count += 1;
        } else {
            count -= 1;
        }
    }

    return result;
}

```

### Sample 9

```
<?php
/**
 * This function aims to find the majority element in a given array.
 * The majority element is defined as the element that appears more than n/2 times in the array.
 * The function assumes that the array is non-empty and the majority element always exists in the array.
 * The function iterates through the array and uses a voting algorithm to find the majority element.
 *
 * Note:
 * * The array can contain both positive and negative integers.
 *
 * Examples:
 * * findMajorityElement(array(3, 3, 4, 2, 4, 4, 2, 4, 4)) => 4
 * * findMajorityElement(array(2, 2, 1, 1, 1, 2, 2)) => 2
 *
 * @param array $arr The input array of integers.
 * @return int The majority element in the array.
 */
function findMajorityElement($arr) {
    $count = 0;
    $result = 0;

    foreach ($arr as $num) {
        if ($count == 0) {
            $result = $num;
            $count += 1;
        } elseif ($num == $result) {
            $count += 1;
        } else {
            $count -= 1;
        }
    }

    return $result;
}
?>

```

### Sample 10

```
#!/bin/bash

# This function aims to find the majority element in a given space-separated list of numbers.
# The majority element is defined as the element that appears more than n/2 times in the list.
# The function assumes that the list is non-empty and the majority element always exists in the list.
# The function iterates through the list and uses a voting algorithm to find the majority element.

find_majority_element() {
    # The function expects a space-separated list of numbers as an argument
    local arr=($1)
    local count=0
    local result=0

    # Iterate through each number in the list
    for num in "${arr[@]}"; do
        # If the count is 0, set the result to num and increment the count
        if [[ $count -eq 0 ]]; then
            result=$num
            ((count++))
        # If the current number is the same as the result, increment the count
        elif [[ $num -eq $result ]]; then
            ((count++))
        # If the current number is different from the result, decrement the count
        else
            ((count--))
        fi
    done

    # After iterating through the list, the result variable should hold the majority element
    echo $result
}

# Example usage:
# echo $(find_majority_element "3 3 4 2 4 4 2 4 4")
# echo $(find_majority_element "2 2 1 1 1 2 2")

```

---

## OpenCoder-LLM / opc-annealing-corpus / synthetic_qa (curated LeetCode-style solutions)

Source: `OpenCoder-LLM/opc-annealing-corpus` config `synthetic_qa` (HF streaming)

### Sample 1

```
package main

type TreeNode struct {
	Val         int
	Left, Right *TreeNode
}

func levelOrder(root *TreeNode) [][]int {
	type Elem struct {
		Node    *TreeNode
		Visited bool
		Level   int
	}

	resWithLevel := make([]Elem, 0)
	stack := make([]Elem, 0)
	stack = append(stack, Elem{root, false, 0})

	for len(stack) > 0 {
		cur_node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		if cur_node.Node != nil {
			if !cur_node.Visited {
				resWithLevel = append(resWithLevel, Elem{cur_node.Node, true, cur_node.Level})
				stack = append(stack, Elem{cur_node.Node, true, cur_node.Level})
				stack = append(stack, Elem{cur_node.Node.Right, false, cur_node.Level + 1})
				stack = append(stack, Elem{cur_node.Node.Left, false, cur_node.Level + 1})
			}
		}
	}

	ms := make(map[int][]int, 0)
	retSlice := make([][]int, 0)

	for _, v := range resWithLevel {
		l := v.Level
		ms[l] = append(ms[l], v.Node.Val)
	}

	// 因为 map 会无序输出，并且其键值最大值即为其长度减一（level）
	for i := 0; i < len(ms); i++ {
		if value, ok := ms[i]; ok {
			retSlice = append(retSlice, value)
		}
	}

	return retSlice
}

// levelOrderWithQueue 使用队列完成层序遍历
func levelOrderWithQueue(root *TreeNode) [][]int {
	if root == nil {
		return nil
	}
	res := make([][]int, 0)
	// nextLevelNodes 其实就代表着下一次待处理的队列
	nextLevelNodes := make([]*TreeNode, 0)
	nextLevelNodes = append(nextLevelNodes, root)

	for len(nextLevelNodes) > 0 {
		tmpLevelNodes := make([]*TreeNode, 0)
		aLevelRes := make([]int, 0)

		for _, v := range nextLevelNodes {
			aLevelR
... [truncated; full doc has 2,568 chars]
```

### Sample 2

```
package leetcode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class leetcodepra {
    public static void main(String[] args) {
        int[] a= {1,3};
        int[] b= {2};
        List list = new ArrayList<>(Arrays.asList(a));
        list.addAll(Arrays.asList(b));
        Object[] c = list.toArray();
        System.out.println(Arrays.toString(c));
    }
}

```

### Sample 3

```
#
# @lc app=leetcode id=914 lang=python3
#
# [914] X of a Kind in a Deck of Cards
#
# https://leetcode.com/problems/x-of-a-kind-in-a-deck-of-cards/description/
#
# algorithms
# Easy (31.96%)
# Likes:    1632
# Dislikes: 422
# Total Accepted:    105.4K
# Total Submissions: 339.7K
# Testcase Example:  '[1,2,3,4,4,3,2,1]'
#
# You are given an integer array deck where deck[i] represents the number
# written on the i^th card.
#
# Partition the cards into one or more groups such that:
#
#
# Each group has exactly x cards where x > 1, and
# All the cards in one group have the same integer written on them.
#
#
# Return true if such partition is possible, or false otherwise.
#
#
# Example 1:
#
#
# Input: deck = [1,2,3,4,4,3,2,1]
# Output: true
# Explanation: Possible partition [1,1],[2,2],[3,3],[4,4].
#
#
# Example 2:
#
#
# Input: deck = [1,1,1,2,2,2,3,3]
# Output: false
# Explanation: No possible partition.
#
#
#
# Constraints:
#
#
# 1 <= deck.length <= 10^4
# 0 <= deck[i] < 10^4
#
#
#


# @lc code=start
from collections import Counter


class Solution:
    def hasGroupsSizeX(self, deck: List[int]) -> bool:
        """
        Solution 1: Hashmap + GCA

        Count all the numbers in the deck and store them in a hashmap.
        Find the greatest common divisor of all the counts.

        Time:
        """

        def gca(a, b):
            """calcualte the greatest common divisor of a and b

            Time: O(log(min(a, b)))
            Space: O(1)
            """
            i
... [truncated; full doc has 1,887 chars]
```

### Sample 4

```
def swap(A, i, j):
    A[i], A[j] = A[j], A[i]

def partition(A):
    '''
    Find the position of x in A
    '''
    x = A[-1] # change to random
    j = -1
    for i in range(len(A) - 1):
        if A[i] <= x:
            j += 1
            swap(A, i, j)
    j += 1
    swap(A, -1, j)
    return j

def select(A, i):
    '''
    Find the element that should be placed in A[i]
    '''
    if len(A) == 1:
        return A[0]
    k = partition(A)
    if i == k: # found it!
        return A[i]
    if k > i:
        return select(A[:k], i)
    return select(A[k:], i - k)

def findKthLargest(nums, k):
    i = len(nums) - k
    return select(nums, i)

print(findKthLargest([4,5,6,7], 4))
print('-----')
print(findKthLargest([7,6,5,4], 4))
print('-----')
print(findKthLargest([9,3,2,4,8], 3))
```

### Sample 5

```
/*
 * @Date: 2023-05-23 20:11:59
 * @LastEditTime: 2023-05-23 20:53:35
 * @题目描述: 
 * @思路解法: 参考18中的思路可以得到适用范围更广泛的算法, 也可以参考2sum的实现
 * @优化思路: 
 * @关键算法: 
 * @复杂度: 
 * @边界条件: 
 * @静态Debug易错点: 
 * @相关题目: 1, 18
 */
/*
 * @lc app=leetcode.cn id=15 lang=cpp
 *
 * [15] 三数之和
 */

// @lc code=start
class Solution {
    // 设置Nsum函数, n代表有相加sum的数目
    vector<vector<int>> Nsum(vector<int>& nums, long target, int n, int start){
        int length = nums.size();
        vector<vector<int>> res;
        // 递归基
        if(n == 2){
            int left = start, right = length - 1;
            long sum;
            int leftnum, rightnum;
            while(left < right){
                leftnum = nums[left];
                rightnum = nums[right];
                sum = leftnum + rightnum;
                if(sum < target){
                    while(left < right && nums[left] == leftnum) left++;
                }else if(sum > target){
                    while(left < right && nums[right] == rightnum) right--;
                }else{
                    res.push_back({leftnum, rightnum});
                    while(left < right && nums[left] == leftnum) left++;
                    while(left < right && nums[right] == rightnum) right--;
                }
            }
            return res;
        }
        for(int i = start; i < length; i++){
            vector<vector<int>> prevres = Nsum(nums, target - nums[i], n - 1, i + 1);
            if(!prevres.empty()){
                for(vector<int>& r: prev
... [truncated; full doc has 1,883 chars]
```

### Sample 6

```
#Beginning of Program 5 

#Create Variables 
count = 0 #This will be used in our function 

#Create File to write to 
file = open("Prime.dat", "w") #This is the name of the file written to


#Create Function to test to see if Number entered is Prime

def getfactor(num): #constructor for function with num as argument. This will be what the user enters
    count = 0 #Create a local variable count inside the function 
    for i in range(1, num + 1): #Now create a for loop which will start at 1 and end at the number entered plus 1
        if (num % i) == 0: #This checks if the remainder of the number entered divided by the current i (iteration) is equal to 0 
            count = count + 1 #This displays the number of factors within the number entered
    if (count > 2): #There there are more than two factors for num entered then it is not prime
         print("This is not a prime Number")
    elif (count == 2): #If Prime numbers are only 1 and itself, so if count = 2 it is prime
         print(num) #Display the number entered


#The function below will retrieve the first 200 Prime Numbers

def solution(): #Function name
    iterations = 0  #Iterations shows how many prime numbers are displayed
    for n in range(2,1225): #This iterates the numbers from 2 - 1225 to check each for prime numbers
        count = 0   #This count is going to be used to show the number of factors each number has
        for i in range(1, n+1): #Prepare second for loop to check for factor
             if
... [truncated; full doc has 2,258 chars]
```

### Sample 7

```

# @Title: 快速公交 (快速公交)
# @Author: KivenC
# @Date: 2020-09-13 13:17:50
# @Runtime: 180 ms
# @Memory: 15.3 MB

class Solution:
    def busRapidTransit(self, target: int, inc: int, dec: int, jump: List[int], cost: List[int]) -> int:

        @functools.lru_cache(None)
        def helper(target):
            if target == 0:
                return 0
            if target == 1:
                return inc
            res = target * inc
            for j, c in zip(jump, cost):
                res = min(res, helper(target // j) + c + (target % j) * inc)
                if target % j > 0:
                    res = min(res, helper(target // j + 1) + c + (j - target % j) * dec)
            return res

        return helper(target) % (10 ** 9 + 7)


```

### Sample 8

```
package com.javarush.task.task31.task3101;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.List;

/*
Проход по дереву файлов
*/

/*
Проходим рекурсивно по всем папкам, и собираем в ArrayList<File> файлы согласно условию,
затем сортируем через собственный компаратор, затем переименовываем файл и проходим по листу, записывая в него содержимое.
*/


public class Solution {
    public static void main(String[] args) throws Exception {
        File path = new File(args[0]);
        File currentFile = new File(args[1]);

        List<File> files = new ArrayList<>();

        recursiveDirectoryWalk(files, path);

        File renamedFile = new File(currentFile.getParent() + File.separator + "allFilesContent.txt");
        FileUtils.renameFile(currentFile, renamedFile);

        try (FileOutputStream out = new FileOutputStream(renamedFile)) {

            for (File f : files) {

                try (FileInputStream in = new FileInputStream(f.getAbsoluteFile())) {

                    byte[] buffer = new byte[1024];
                    int length;
                    while ((length = in.read(buffer)) > 0) {
                        out.write(buffer, 0, length);
                        out.flush();
                    }
                    out.write(10);
                }
            }
        }
    }

    public static void recursiveDirectoryWalk(List<File> files, File directory) {

        if (director
... [truncated; full doc has 1,808 chars]
```

### Sample 9

```
'''
给定一个 n × n 的二维矩阵表示一个图像。

将图像顺时针旋转 90 度。

说明：

你必须在原地旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要使用另一个矩阵来旋转图像。

示例 1:

给定 matrix = 
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
],

原地旋转输入矩阵，使其变为:
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]
示例 2:

给定 matrix =
[
  [ 5, 1, 9,11],
  [ 2, 4, 8,10],
  [13, 3, 6, 7],
  [15,14,12,16]
], 

原地旋转输入矩阵，使其变为:
[
  [15,13, 2, 5],
  [14, 3, 4, 1],
  [12, 6, 8, 9],
  [16, 7,10,11]
]
'''

class Solution:
    def rotate(self, matrix):
        """
        Do not return anything, modify matrix in-place instead.
        """
        if len(matrix) == 0:
            return None
        w = len(matrix)
        h = len(matrix[0])
        for i in range(h):
            for j in range(w//2):
                matrix[i][j], matrix[i][w-j-1] = matrix[i][w-j-1], matrix[i][j]
        for i in range(h):
            for j in range(w-i-1):
                matrix[i][j], matrix[w-j-1][h-i-1] = matrix[w-j-1][h-i-1], matrix[i][j]
matrix = [
  [1,2,3],
  [4,5,6],
  [7,8,9]
]
Solution().rotate(matrix)
print(matrix)
```

### Sample 10

```
#include <stdio.h>
#include <map>
#include <string>

struct ListNode {
	std::string key;
	int val;
	ListNode *next;
	ListNode(int x) : val(x), next(NULL) {}
};

int main(){
	std::map<std::string, int> hash_map;	
	std::string str1 = "abc";
	std::string str2 = "aaa";
	std::string str3 = "xxxxx";	
	hash_map[str1] = 1;
	hash_map[str2] = 2;
	hash_map[str3] = 100;	
	if (hash_map.find(str1) != hash_map.end()){
		printf("%s is in hash_map, value is %d\n",
			str1.c_str(), hash_map[str1]);
	}	
	std::map<std::string, int> ::iterator it;
	for (it = hash_map.begin(); it != hash_map.end(); it++){
		printf("hash_map[%s] = %d\n", it->first.c_str(), it->second);
	}	
	return 0;
}

```

---


---

# Additional sources (added 2026-06-07 PST — C5-v2 candidates)


## Stack-Edu / Python (SmolLM2's code data — classifier-filtered Stack v2; content fetched on-demand from Software Heritage S3)

Source: ``/fsx/users/dongweij/marin/outputs/raw/stack-edu/Python/*.parquet` (metadata) → softwareheritage S3 `content/{blob_id}` (gzipped)`

### Sample 1

_(deyvisonguilherme/TCGuaruja / /api/bll/categoria.py (score 2.52))_

```
import falcon
from dal.categoria import categoriaDAL

class categoriaBLL(object):
    def on_get(self, req, resp):
        try:
            resp.status = falcon.HTTP_200
            resp.body = ('\nO homem não é nada além daquilo que a educação faz dele.\n\n'
                         '    ~ Immanuel Kant\n\n')
        except(IOError):
            raise falcon.HTTPError(falcon.HTTP_725,'ERROR')

    def on_post(self, req, resp):
        try:
            negocio = categoriaDAL()
            negocio.add(req['categoria'])
            resp.status = falcon.HTTP_200
            resp.body = ('cadastro efetuado com sucesso')
        except(IOError):
            raise falcon.HTTPError


    def on_put(self, req, resp):
        try:
            pass
        except(IOError):
            raise falcon.HTTPError


    def on_delete(self, req, resp):
        try:
            pass
        except(IOError):
            raise falcon.HTTPError
```

### Sample 2

_(maximilianh/pubMunch / /lib/maxXml.py (score 3.45))_

```
#!/usr/bin/env python
import logging, urllib
import xml.etree.cElementTree as etree

class XmlParser(object):
    """ class to represent an xml tree (using ElementTree)
        Functions Accept PATH which is a /a/b/c style xpath-like expression to refer to elements
        PATH is not a complete XPATH implementation

        getText... functions return just a string
        getXml... functions return an XmlParser-object
        ...First  functions get only the first instance
        ...All    functions return an iterator

    >>> xp = XmlParser(string="<fruit><apple size='big'>boskoop</apple><apple size='small'>granny smith</apple><pear>mypear</pear></fruit>")
    >>> xp.getTextFirst("pineapple", default="NothingAtAll")
    'NothingAtAll'
    >>> xp.getTextFirst("apple")
    'boskoop'
    >>> list(xp.getTextAll("apple"))
    ['boskoop', 'granny smith']
    >>> list(xp.getTextAll("apple", reqAttrDict={'size':'big'}))
    ['boskoop']

    """
    def __init__(self, string=None, url=None, root=None, removeNamespaces=False):
        self.root=None
        if string!=None:
            self.fromString(string, removeNamespaces)
        elif url!=None:
            self.fromUrl(url, removeNamespaces)
        elif root!=None:
            self.root=root

    def getAttr(self, name):
        return self.root.attrib.get(name, None)

    def getText(self):
        if self.root.text==None:
            return ""
        else:
            return self.root.text

    def getTextTail(self):
    
... [truncated; full doc has 6,421 chars]
```

### Sample 3

_(kumaraguru-pk/cloud-sheet / /server.py (score 2.64))_

```
import flask
from werkzeug.utils import ArgumentValidationError
app = flask.Flask(__name__)
from sheets import *
from flask import jsonify
sheet_manager = SheetManager()


@app.route("/sheets/<sheet_id>/worksheets/<wsheet_id>",methods=['GET'])
def index(sheet_id, wsheet_id):
    sheet_manager.create(sheet_id, wsheet_id)
    rows = [str(i+1) for i in range(10)]
    columns = [chr(i+ord('A')) for i in range(10)]
    worksheet_local = {}
    for row in rows:
        for col in columns:
            if col+row in sheet_manager.sheets[sheet_id].worksheets[wsheet_id].cells:
                info = {
                    'value':sheet_manager.sheets[sheet_id].worksheets[wsheet_id].cells[col+row].value,
                    'formula':sheet_manager.sheets[sheet_id].worksheets[wsheet_id].cells[col+row].formula
                    }
                worksheet_local[col+row] = info
            else:
                worksheet_local[col+row] = {}
    return flask.render_template('index.html', rows=rows, columns=columns, worksheet_local=worksheet_local, sheet_id=sheet_id, wsheet_id=wsheet_id)

@app.route('/sheets/<sheet_id>/worksheets/<wsheet_id>/cell/<cell_id>/<value>', methods=['PUT'])
def update_cell(sheet_id, wsheet_id, cell_id,value):
    # show the user profile for that user
    try:
        return jsonify(sheet_manager.sheets[sheet_id].worksheets[wsheet_id].update_cell(cell_id,value))
    except CircularReferenceException as err:
        return "Circular reference", 422
    except Argumen
... [truncated; full doc has 1,857 chars]
```

### Sample 4

_(karinafl510/postman-practice / /api/count.py (score 3.00))_

```
# define functions for HTTP verb methods used in rest apps

from flask_restful import Resource # convert class as resource
from flask import request
import logging as logger
from .calcs import days_between, weeks_between, months_between


class count(Resource): # Task class is inheriting from Resource class

    def get(self,choice):
        logger.debug("Inside get method of count")

        return {"Type" : "Calculating {}".format(choice)},200

    def post(self,choice):
        logger.debug("Inside post method of count")

        # retrieve data from postman body
        req_data = request.get_json()
        date1 = req_data['Date 1']
        date2 = req_data['Date 2']

        # calculation functions
        days = days_between(date1, date2)
        weeks = weeks_between(date1, date2)
        months = months_between(date1, date2)

        if choice=='days':
            return {"Days" : "{}".format(days)},200
        if choice=='weeks':
            return {"Weeks" : "{}".format(weeks)},200
        if choice=='months':
            return {"Months" : "{}".format(months)},200
        if choice=='all':
            return {
                "Days" : "{} days".format(days),
                "Weeks" : "{} weeks".format(weeks),
                "Months" : "{} months".format(months)
            },200


    def put(self,choice):
        logger.debug("Inside put method of count")
        return {"message" : "inside put method of count. CHOICE-
... [truncated; full doc has 1,805 chars]
```

### Sample 5

_(valevo/Thesis / /src/subsampling/heap.py (score 2.56))_

```
# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles
from data.corpus import Words, Articles, Sentences

from stats.stat_functions import compute_vocab_size

from stats.mle import Heap

from jackknife.plotting import hexbin_plot

import numpy as np
import numpy.random as rand

import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import os
        
def heap(corp, rng):
    vocab_sizes = []
    for i, ntoks in enumerate(rng):
        if i % 10 == 0:
            print(i, ntoks)
        subsample = Sentences.subsample(corp, ntoks)
        vocab_size = compute_vocab_size(subsample)
        vocab_sizes.append(vocab_size)
        
    return vocab_sizes

def heap_from_file(save_dir, rng_params):
    rng_params = map(str, rng_params)
    required_file_name = "vocab_growth_" + "_".join(rng_params) + ".pkl"
    print(required_file_name)
    if required_file_name in os.listdir(save_dir):
        with open(save_dir + required_file_name, "rb") as handle:
            return pickle.load(handle)
    else:
        raise FileNotFoundError


def do_mles(rng, vocab_sizes, save_dir):
    with open(save_dir + "mle_heap_point_estimates.txt", "w") as handle:
        for vs in vocab_sizes:
            heap = Heap(vs, rng)
            heap_fit = heap.fit(start_params=np.asarray([100000.0, 1.0]), 
                                        method="powell", full_output=True)    
            heap.register_fit(heap_fit)

            handle.write(heap.print_result(string=True))
 
... [truncated; full doc has 3,206 chars]
```

### Sample 6

_(DenBlacky808/--Python / /3. Cells.py (score 3.86))_

```
class Cell:
    def __init__(self, population):
        self.population = population

    def __add__(self, other):
        return self.population + other.population

    def __sub__(self, other):
        if self.population >= other.population:
            return self.population - other.population
        else:
            return 'Ячеек в первой клетке меньше чем во второй!'

    def __mul__(self, other):
        return Cell(self.population * other.population)

    def __truediv__(self, other):
        return Cell(self.population // other.population)

    def make_order(self, raw):
        num = self.population // raw
        return print('\n'.join([(''.join(['*' for _ in range(raw)])) for _ in range(num)]) + '\n' + ''.join(
            ['*' for _ in range(self.population - num * raw)]))


cells_1 = Cell(13)
cells_2 = Cell(25)
print(cells_1 + cells_2)
print(cells_1 - cells_2)
print(cells_2 - cells_1)
cells_3 = cells_1 * cells_2
cells_4 = cells_2 / cells_1
print(cells_3.population)
print(cells_4.population)
cells_2.make_order(7)

```

### Sample 7

_(anuckp/calculator / /calculator.py (score 4.09))_

```
x=int(input("Enter the first num: "))
y=int(input("Enter the second num: "))
while True:
    z=input("Enter the operator: ")
    if z=='+':
        result=x+y
        print(result)
        break

    elif z=='-':
        result=x-y
        print(result)
        break
    
    elif z=='*':
        result=x*y
        print(result)
        break
    
    elif z=='/':
        result=x/y
        print(result)
        break
    elif z=='%':
        result=x%y
        print(result)
        break
    else:
        print("choose in between + - * /")
    

```

### Sample 8

_(Alexanderklau/Algorithm / /Everyday_alg/2021/01/2021_01_22/number-of-rectangles-that-can-form-the-largest-square.py (score 3.52))_

```
# coding: utf-8

__author__ = 'Yemilice_lau'

"""
给你一个数组 rectangles ，其中 rectangles[i] = [li, wi] 表示第 i 个矩形的长度为 li 、宽度为 wi 。

如果存在 k 同时满足 k <= li 和 k <= wi ，就可以将第 i 个矩形切成边长为 k 的正方形。例如，矩形 [4,6] 可以切成边长最大为 4 的正方形。

设 maxLen 为可以从矩形数组 rectangles 切分得到的 最大正方形 的边长。

返回可以切出边长为 maxLen 的正方形的矩形 数目 。

 

示例 1：

输入：rectangles = [[5,8],[3,9],[5,12],[16,5]]
输出：3
解释：能从每个矩形中切出的最大正方形边长分别是 [5,3,5,5] 。
最大正方形的边长为 5 ，可以由 3 个矩形切分得到。
示例 2：

输入：rectangles = [[2,3],[3,7],[4,3],[3,7]]
输出：3
"""

rectangles = [[5,8],[3,9],[5,12],[16,5]]


z = []

for i in rectangles:
    z.append(min(i))



v = []
for i in list(set(z)):
    v.append(z.count(i))


print(sum(1 for i in z if i == max(v)))

```

### Sample 9

_(CapstoneProject18/Twitter-sentiment-analysis / /m7.py (score 4.00))_

```
'''
---------------------------------------------------------------------------------------------------
#pseudocode
#1 importing csv module
#2 file = open ("file path")
#3 data of file which is read is Stored in a variable csv_file
#4  for loop :
     #making count of row
     #if > 1
     #masking the first 4 numbers with x
     #printing row
#5 else printing number without masking
----------------------------------------------------------------------------------------------------
'''
#importing csv
import csv
#open file
file = open('C:\\Users\\764024\\Desktop\\exercises\\Worksheet in Problems.csv')
#reading csv file variable name(file) 
csv_file = csv.reader(file)
count = -1
#for every row in csv file we run the loop
for row in csv_file:
            
    count = count+1
    #storing that row in variable a
    if count > 0:
        a = row[0]
    #printing the row details and replacing first 3 digits with x and displaying last four digits
        row[0] = (a[-4:].rjust(len(a),'x'))
        print(row)
    else:
        print(row)
file.close()

```

### Sample 10

_(RuiboZhang1/test_one / /10001st prime.py (score 3.42))_

```
import time
start=time.clock()
i=int(input('please enter  an integer:'))
#创建一个空list

r=list()
NumPrime = 0
#添加元素2
r.append(2)
#从3开始挨个筛选
for a in range(3,i):
    b=False

#用a除以小于a的质数b
    for b in r:
        if a%b==0:
            b=False
            break
        else:
            b=True
    if b==True:
        r.append(a)
        NumPrime +=1
        if NumPrime == 10000:
            break

print (r)
print(NumPrime)
print(r[10000])
t=(time.clock()-start)
print (t)
```

---

## Nemotron-Pretraining-Specialized-v1.1 / Code-Concepts (synthetic Python from concept-taxonomy prompts, gpt-oss-20b/120b)

Source: ``/fsx/users/dongweij/marin/outputs/raw/nemotron_specialized_v11/Nemotron-Pretraining-Code-Concepts/*.parquet``

### Sample 1

_(shard part_000021.parquet, row 89338, category=Nemotron-Pretraining-Code-Concepts)_

```
def max_subarray_from_parentheses(expr: str) -> int:
    """
    Parse a string of comma‑separated integers optionally enclosed in parentheses,
    produce a flat list, then treat it as a line graph of consecutive nodes
    and search for the contiguous subsequence with maximum sum (Kadane style).
    >>> max_subarray_from_parentheses("(1,2,-3)(4,5)(-6,7)")
    9
    >>> max_subarray_from_parentheses("(5)(-1,2)(3)(-4,1)")
    9
    """
    # ------------------------------------------------------------
    # 1. Extract all integer tokens from the expression.
    #    Parentheses are only delimiters; they can be replaced by commas.
    # ------------------------------------------------------------
    tokens = expr.replace('(', ',').replace(')', ',').split(',')
    # keep non‑empty trimmed parts and convert them to ints
    nums = [int(tok) for tok in tokens if tok.strip()]

    # ------------------------------------------------------------
    # 2. Kadane's algorithm – maximum sub‑array sum in O(n).
    #    If the list is empty we return 0 (nothing to sum).
    # ------------------------------------------------------------
    if not nums:
        return 0

    max_ending = max_sofar = nums[0]
    for x in nums[1:]:
        # best sum of a sub‑array that ends at the current element
        max_ending = max(x, max_ending + x)
        # best sum seen so far
        max_sofar = max(max_sofar, max_ending)

    return max_sofar
```

### Sample 2

_(shard part_000054.parquet, row 13605, category=Nemotron-Pretraining-Code-Concepts)_

```
from typing import List

def prime_substring_count(digits: str) -> int:
    """
    Count every contiguous substring of a digit string that, when interpreted as an
    integer (ignoring leading zeros), is a prime number.  The function uses a
    sieve of Eratosthenes to generate primes only up to the largest possible
    integer represented by the string, and then searches all substrings.

    >>> prime_substring_count("1012")
    2
    >>> prime_substring_count("13")
    2
    >>> prime_substring_count("123")
    3
    """
    # Edge case: empty string
    if not digits:
        return 0

    # Largest integer that can appear among substrings
    max_val = int(digits)
    if max_val < 2:
        return 0

    # --------- Sieve of Eratosthenes ----------
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    p = 2
    while p * p <= max_val:
        if is_prime[p]:
            step = p
            start = p * p
            for multiple in range(start, max_val + 1, step):
                is_prime[multiple] = False
        p += 1
    # ------------------------------------------

    count = 0
    n = len(digits)
    # Enumerate all contiguous substrings
    for i in range(n):
        # Build the number incrementally to avoid repeated int() on the same prefix,
        # but the straightforward conversion is still O(1) per substring given small sizes.
        for j in range(i + 1, n + 1):
            val = int(digits[i:j])          # leading zeros are ign
... [truncated; full doc has 1,587 chars]
```

### Sample 3

_(shard part_000010.parquet, row 35404, category=Nemotron-Pretraining-Code-Concepts)_

```
from typing import List

def has_prime_pair_with_sum(prime_count: int, target: int) -> bool:
    """
    Generate the first `prime_count` primes, push them onto a stack,
    then ask whether any two primes from that stack add up to `target`.
    The algorithm uses the classic two‑sum technique on the prime list.
    It returns True if such a pair exists, otherwise False.

    >>> has_prime_pair_with_sum(10, 20)
    True   # 13 + 7
    >>> has_prime_pair_with_sum(5, 9)
    False  # primes are 2,3,5,7,11; none sum to 9
    >>> has_prime_pair_with_sum(15, 50)
    True   # 43 + 7 (within the first 15 primes)
    """
    # ---------- helper to test primality ----------
    def is_prime(n: int) -> bool:
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        i = 3
        while i * i <= n:
            if n % i == 0:
                return False
            i += 2
        return True

    # ---------- generate first `prime_count` primes ----------
    primes: List[int] = []
    candidate = 2
    while len(primes) < prime_count:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1

    # Need at least two numbers for a pair
    if len(primes) < 2:
        return False

    # ---------- classic two‑sum on the sorted list ----------
    left, right = 0, len(primes) - 1
    while left < right:
        s = primes[left] + primes[right]
        if s == target:
      
... [truncated; full doc has 1,616 chars]
```

### Sample 4

_(shard part_000021.parquet, row 147052, category=Nemotron-Pretraining-Code-Concepts)_

```
from typing import List, Tuple

def csv_contiguous_sums(csv: str) -> List[Tuple[int, int, int]]:
    """Return a list of contiguous ranges and their sums from a comma‑separated string.

    The input string contains integers separated by commas.
    Consecutive integers in the sequence that form a continuous block
    (each element +1 from the previous) are grouped as a range.
    For each range you must produce a tuple (start, end, sum) where
    * start  – first value of the range
    * end    – last value of the range
    * sum    – arithmetic sum of all values inside the range

    Example 1:
    >>> csv_contiguous_sums("1,2,3,5,6,10")
    [(1, 3, 6), (5, 6, 11), (10, 10, 10)]

    Example 2:
    >>> csv_contiguous_sums("4,5,6,7,8")
    [(4, 8, 34)]

    Example 3:
    >>> csv_contiguous_sums("9")
    [(9, 9, 9)]
    """
    # Edge case: empty string -> no ranges
    if not csv.strip():
        return []

    # Convert the CSV string to a list of integers
    numbers = [int(item) for item in csv.split(",")]

    result: List[Tuple[int, int, int]] = []
    start = numbers[0]
    prev = numbers[0]

    for n in numbers[1:]:
        # If the current number continues the contiguous block
        if n == prev + 1:
            prev = n
            continue

        # Block ended – compute the range and its sum
        end = prev
        # arithmetic series sum: count * (first + last) // 2
        count = end - start + 1
        total = count * (start + end) // 2
        result.
... [truncated; full doc has 1,761 chars]
```

### Sample 5

_(shard part_000021.parquet, row 183943, category=Nemotron-Pretraining-Code-Concepts)_

```
from typing import List

def unique_or_sum(arrays: List[List[int]]) -> int:
    """Compute the sum of unique bitwise OR values from a list of integer sub‑lists. For each sub‑list, compute the bitwise OR of its elements, collect these OR results into a set to remove duplicates, and sum the unique values.

    >>> unique_or_sum([[1, 2], [3], [1, 2]])
    3
    >>> unique_or_sum([[5, 2], [15], [8, 1]])
    31
    >>> unique_or_sum([[0], [0, 1], [1]])
    1
    """
    # Set to keep distinct OR results
    seen = set()

    for sub in arrays:
        # Bitwise OR of an empty list is defined as 0 (neutral element)
        cur = 0
        for num in sub:
            cur |= num
        seen.add(cur)

    # Sum of the unique OR values
    return sum(seen)
```

### Sample 6

_(shard part_000015.parquet, row 42933, category=Nemotron-Pretraining-Code-Concepts)_

```
from typing import List, Tuple

def max_xor_path(adj: List[List[Tuple[int, int]]]) -> int:
    """
    Compute the maximum XOR‑sum of edge weights along any simple path
    from vertex 0 to vertex n‑1 in an undirected graph.

    The graph is given as an adjacency list where adj[u] is a list of
    (v, w) pairs meaning an edge between u and v with integer weight w.
    Vertices are numbered from 0.

    The algorithm performs a depth‑first search using an explicit stack
    (potentially very large recursion depth) and tracks the XOR value
    of the current path.  Whenever the destination vertex is reached,
    the XOR value of that path is compared to the current best result,
    and the maximum is returned.

    Examples
    -------
    >>> # Graph: 0--1--2  (weights 5 and 3)
    >>> adj1 = [[(1, 5)], [(0, 5), (2, 3)], [(1, 3)]]
    >>> max_xor_path(adj1)
    6
    >>> # Graph:
    >>> # 0--1 (1), 1--2 (2), 0--2 (4), 2--3 (8)
    >>> adj2 = [[(1, 1), (2, 4)], [(0, 1), (2, 2)], [(1, 2), (0, 4), (3, 8)], [(2, 8)]]
    >>> max_xor_path(adj2)
    12
    """
    n = len(adj)
    if n == 0:
        return 0
    # best XOR found so far
    best = 0

    # visited[i] == True while vertex i is on the current DFS stack
    visited = [False] * n

    # stack holds frames (node, xor_to_node, next_index)
    # next_index is the position of the next neighbour to explore
    stack: List[Tuple[int, int, int]] = [(0, 0, 0)]
    visited[0] = True

    while stack:
        node, cur_xor, nxt 
... [truncated; full doc has 2,223 chars]
```

### Sample 7

_(shard part_000000.parquet, row 237477, category=Nemotron-Pretraining-Code-Concepts)_

```
from typing import Dict

def filter_dict_by_value_range(dictionary: Dict[str, int], low: int, high: int) -> Dict[str, int]:
    """Return a new dictionary containing only the entries from the input dictionary whose integer values lie within the inclusive range [low, high].

    The original dictionary is left unchanged.  
    The function uses a simple brute‑force scan of all keys to select the qualifying entries.

    >>> filter_dict_by_value_range({'a': 1, 'b': 4, 'c': 7}, 3, 6)
    {'b': 4}
    >>> filter_dict_by_value_range({'x': 10, 'y': 20}, 5, 15)
    {'x': 10}
    >>> filter_dict_by_value_range({'p': 5}, 6, 10)
    {}
    """
    # Use a dictionary comprehension to copy only those key‑value pairs
    # whose values fall within the inclusive range [low, high].
    return {k: v for k, v in dictionary.items() if low <= v <= high}
```

### Sample 8

_(shard part_000027.parquet, row 202791, category=Nemotron-Pretraining-Code-Concepts)_

```
from typing import List

def shortest_path_weight(edge_list: List[str], start: str, end: str) -> int:
    """
    Given a directed graph encoded as a list of strings in the form
    "node_a,node_b,weight", convert the string representation to a graph
    and return the total weight of the shortest path from **start** to
    **end**. If no path exists, return -1.

    >>> shortest_path_weight(
    ...     ["A,B,3", "B,C,2", "A,C,5"], "A", "C")
    5
    >>> shortest_path_weight(
    ...     ["A,B,1", "B,C,4", "A,C,10"], "A", "C")
    5
    """
    # ------------------------------------------------------------------
    # Build adjacency list: node -> list of (neighbor, weight)
    # ------------------------------------------------------------------
    graph = {}
    for token in edge_list:
        a, b, w = token.split(',')
        w = int(w)
        graph.setdefault(a, []).append((b, w))
        # Ensure every node appears in the dict (even if it has no outgoing edges)
        graph.setdefault(b, [])

    # Trivial case: start equals end -> zero weight
    if start == end:
        return 0

    # If start node does not exist in the graph, there is no path
    if start not in graph:
        return -1

    # ------------------------------------------------------------------
    # Dijkstra's algorithm (using a min‑heap) to find the shortest distance
    # ------------------------------------------------------------------
    import heapq

    INF = float('inf')
    dist = {node
... [truncated; full doc has 2,075 chars]
```

### Sample 9

_(shard part_000005.parquet, row 230327, category=Nemotron-Pretraining-Code-Concepts)_

```
from typing import Dict, List

def find_encrypted_palindromic_path(
    graph: Dict[int, List[int]],
    labels: Dict[int, str],
    shift: int,
    start: int,
    goal: int,
    max_depth: int = 10
) -> bool:
    """Return True if there is a simple path from `start` to `goal` such that the
    concatenated node labels along the path, when each character is shifted
    by `shift` (a Caesar‑cipher encryption), form a palindrome string.

    The search must not revisit nodes and cannot exceed `max_depth` edges.

    >>> find_encrypted_palindromic_path(
    ...     {0:[1], 1:[2], 2:[]},
    ...     {0:"ab", 1:"c", 2:"ba"},
    ...     shift=1,
    ...     start=0,
    ...     goal=2
    ... )
    True

    >>> find_encrypted_palindromic_path(
    ...     {0:[1,2], 1:[3], 2:[3], 3:[]},
    ...     {0:"x", 1:"y", 2:"z", 3:"x"},
    ...     shift=2,
    ...     start=0,
    ...     goal=3
    ... )
    False

    >>> find_encrypted_palindromic_path(
    ...     {0:[1], 1:[2,3], 2:[4], 3:[4], 4:[]},
    ...     {0:"a", 1:"b", 2:"c", 3:"b", 4:"a"},
    ...     shift=0,
    ...     start=0,
    ...     goal=4,
    ...     max_depth=5
    ... )
    True
    """
    # Helper: apply Caesar shift to a string (wrap for letters, leave others)
    def caesar(s: str, k: int) -> str:
        res = []
        for ch in s:
            if 'a' <= ch <= 'z':
                res.append(chr(((ord(ch) - 97 + k) % 26) + 97))
            elif 'A' <= ch <= 'Z':
                res.append(chr(((ord(ch) -
... [truncated; full doc has 2,736 chars]
```

### Sample 10

_(shard part_000038.parquet, row 98984, category=Nemotron-Pretraining-Code-Concepts)_

```
from typing import List

def has_two_sum_modulo(nums: List[int], mod: int, target: int) -> bool:
    """Check if any two distinct elements of *nums* satisfy
    (nums[i] + nums[j]) % mod == target.

    The task combines the classic two‑sum problem, efficient bit
    manipulation for lookup, and modular arithmetic properties.
    Using a bit set (bit vector) helps achieve linear time.

    >>> has_two_sum_modulo([1, 4, 5, 6], 7, 6)
    True
    >>> has_two_sum_modulo([2, 2, 2], 4, 0)
    True
    >>> has_two_sum_modulo([1, 2, 3], 5, 4)
    False
    """
    # Normalise the target to the same modular range.
    target %= mod

    # `seen` will store residues we have already processed.
    # Using a set gives O(1) average‑time membership tests,
    # which is effectively the same as a bit‑vector for this problem.
    seen = set()

    for value in nums:
        # Only the residue modulo `mod` matters for the sum.
        residue = value % mod

        # We need a complement `c` such that (residue + c) % mod == target.
        # Rearranging gives c ≡ target - residue (mod mod).
        complement = (target - residue) % mod

        # If we have already seen the needed complement, we have a valid pair.
        if complement in seen:
            return True

        # Record the current residue for future checks.
        seen.add(residue)

    # No suitable pair was found.
    return False
```

---

## Nemotron-Pretraining-Specialized-v1.1 / Unconditional-Algorithmic (synthetic Python from minimal 'write a function' prompts, gpt-oss-120b + Qwen3-235B)

Source: ``/fsx/users/dongweij/marin/outputs/raw/nemotron_specialized_v11/Nemotron-Pretraining-Unconditional-Algorithmic/*.parquet``

### Sample 1

_(shard part_000000.parquet, row 1728, category=Nemotron-Pretraining-Unconditional-Algorithmic)_

```
Problem:
Shortest Path with Exactly K Edges

You are given a directed weighted graph with N vertices (numbered 1 … N) and M edges. All edge weights are non‑negative integers.

For each query you are given three integers u, v, K. You have to output the length of the shortest path that starts at vertex u, ends at vertex v and uses exactly K edges. If no such path exists output -1.

Input
N M
a1 b1 w1
...
aM bM wM
Q
u1 v1 K1
...
uQ vQ KQ

Constraints
1 ≤ N ≤ 100
0 ≤ M ≤ N·(N‑1)
1 ≤ ai, bi ≤ N
0 ≤ wi ≤ 10^9
1 ≤ Q ≤ 2000
0 ≤ Ki ≤ 10^9

If there are multiple edges between the same pair of vertices keep only the smallest weight.

Output
For each query output a single integer – the length of the shortest path that uses exactly K edges, or -1 if such a path does not exist.

Sample
Input
4 5
1 2 3
2 3 4
3 4 5
1 3 10
2 4 1
5
1 4 2
1 4 3
1 4 4
2 3 1
3 1 0

Output
4
12
-1
4
-1

Explanation
* K = 2: 1 → 2 (3) → 4 (1) → total 4.
* K = 3: 1 → 2 (3) → 3 (4) → 4 (5) → total 12.
* K = 4: no path with exactly 4 edges exists.
* K = 1 from 2 to 3 is the direct edge weight 4.
* K = 0 from 3 to 1 is possible only if u == v; here it isn’t, so -1.

Solution:
```python
import sys

INF = 10 ** 18

def min_plus_mat_mul(A, B, n):
    """C = A ⊗ B (min‑plus multiplication) for n×n matrices."""
    C = [[INF] * n for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for k in range(n):
            aik = Ai[k]
            if aik == INF:
                continue
            Bk =
... [truncated; full doc has 3,274 chars]
```

### Sample 2

_(shard part_000000.parquet, row 117524, category=Nemotron-Pretraining-Unconditional-Algorithmic)_

```
1. Task:

Write a python function `minimumDeletionSize(strs: List[str]) -> int` that, given a list of equal‑length strings, returns the minimum number of columns that must be deleted so that each remaining column is sorted in non‑decreasing order when read top‑to‑bottom. The solution should run in O(m·n) time where `m` is the number of strings and `n` is their length.

2. Hint:
Iterate column by column. Keep a boolean array `sorted[i]` indicating whether the pair of rows `i` and `i+1` is already confirmed to be in correct order by a previous column. If a column causes any unsorted pair to become decreasing, that column must be deleted. Otherwise, mark newly sorted pairs. Count deleted columns.

3. Starter Code:
def minimumDeletionSize(strs):
    

4. Answer:
def minimumDeletionSize(strs):
    if not strs:
        return 0

    m, n = len(strs), len(strs[0])
    # sorted[i] == True means rows i and i+1 are already ordered
    sorted_pair = [False] * (m - 1)
    deletions = 0

    for col in range(n):
        delete_this = False
        for i in range(m - 1):
            if not sorted_pair[i] and strs[i][col] > strs[i + 1][col]:
                delete_this = True
                break

        if delete_this:
            deletions += 1
            continue

        # Update sorted pairs with this column
        for i in range(m - 1):
            if strs[i][col] < strs[i + 1][col]:
                sorted_pair[i] = True

        # Early exit: if all pairs are sorted, we can stop

... [truncated; full doc has 2,395 chars]
```

### Sample 3

_(shard part_000000.parquet, row 11504, category=Nemotron-Pretraining-Unconditional-Algorithmic)_

```
**Problem**

> **Path K‑th Smallest (Hard)**
> 
> You are given a tree with `N` vertices (`1 ≤ N ≤ 2·10⁵`).  
> The vertices are numbered from `1` to `N`.  
> Each vertex `i` has an integer value `a[i]` (`0 ≤ a[i] ≤ 10⁹`).
> 
> The tree is rooted at vertex `1`.  
> You have to answer `Q` queries (`1 ≤ Q ≤ 2·10⁵`).  
> Queries are given online and are of two types:
> 
> 1. **Update** – `1  u  x`  
>    Change the value of vertex `u` to `x` (`0 ≤ x ≤ 10⁹`).
> 
> 2. **Path‑kth** – `2  u  v  k`  
>    Consider the simple path from `u` to `v` (both inclusive).  
>    Let `S` be the multiset of values of the vertices on this path.  
>    Output the `k`‑th smallest element of `S` (1‑based).  
>    If `k` is larger than the number of vertices on the path, output `-1`.
> 
> All queries must be processed in the order they appear.
> 
> **Input**
> ```
> N Q
> a[1] a[2] … a[N]
> N‑1 lines:  u v          (edges of the tree)
> Q lines:   type …        (queries as described)
> ```
> 
> **Output**
> For each query of type 2 print the answer on a separate line.
> 
> **Constraints**
> * `1 ≤ N, Q ≤ 2·10⁵`
> * The sum of `N` over all test files does not exceed `2·10⁵`.
> * The sum of `Q` over all test files does not exceed `2·10⁵`.
> * `0 ≤ a[i], x ≤ 10⁹`
> * `1 ≤ u, v ≤ N`
> * `1 ≤ k ≤ N`

--------------------------------------------------------------------

### Explanation / Reasoning  

The classic way to answer “k‑th smallest on a path” is to use a **persistent segment tree** (also called a
... [truncated; full doc has 14,554 chars]
```

### Sample 4

_(shard part_000000.parquet, row 174886, category=Nemotron-Pretraining-Unconditional-Algorithmic)_

```
# Question

Write a function to find the lexicographical order of two given strings without using any built-in comparison functions. The function should return 0 if the strings are equal, a negative value if the first string is lexicographically smaller, and a positive value if it is greater. Constraints: no built‑in comparison operators and avoid explicit loops or recursion. Assume ASCII input (bonus for Unicode).

# Explanation
1. Use zip to pair characters from both strings and map with a lambda that computes the difference of their ASCII codes. 2. Find the first non‑zero difference; if such a difference exists, return it. 3. If all paired characters are equal, return the difference in lengths of the two strings. This approach satisfies the lexicographical ordering rules while avoiding explicit comparison operators and loops.

# Starter Code:
def string_compare(str1, str2):
    

# Implementation
def string_compare(str1, str2):
    diff = list(map(lambda x: ord(x[0]) - ord(x[1]), zip(str1, str2)))
    first_non_zero = next((d for d in diff if d != 0), 0)
    if first_non_zero != 0:
        return first_non_zero
    return len(str1) - len(str2)

# Testing
```python
print(string_compare("apple", "banana"))  # -1
print(string_compare("hello", "hello"))   # 0
print(string_compare("zebra", "apple"))   # 25
print(string_compare("abc", "abcd"))      # -3
print(string_compare("xyz", "XYZ"))       # 32
```
```

### Sample 5

_(shard part_000000.parquet, row 33095, category=Nemotron-Pretraining-Unconditional-Algorithmic)_

```
**Problem – “K‑Turn Unique Paths”**

You are given an `N × M` grid.  
Some cells contain obstacles (marked with `#`), all other cells are free (marked with `.`).  
You start in the top‑left cell `(0, 0)` and want to reach the bottom‑right cell `(N‑1, M‑1)`.  
From any free cell you may move **only** to the cell directly to the right or directly below it (i.e. directions **right** or **down**) and you may never step on an obstacle.

A **turn** is a change of moving direction:  
* moving right after having moved down, or  
* moving down after having moved right.

The first step does **not** count as a turn because there is no previous direction.

Given an integer `K (0 ≤ K ≤ 10)`, count how many different paths from `(0, 0)` to `(N‑1, M‑1)` use **at most** `K` turns.

The answer can be large; output it modulo `10⁹+7`.

---

### Input
```
N M K
grid line 1
grid line 2
...
grid line N
```
* `1 ≤ N, M ≤ 100`
* `0 ≤ K ≤ 10`
* Each of the next `N` lines contains exactly `M` characters, each either `.` (free) or `#` (obstacle).  
  The start `(0,0)` and the target `(N‑1,M‑1)` are always free.

### Output
A single integer – the number of valid paths modulo `1 000 000 007`.

### Sample
**Input**
```
3 3 1
...
.#.
...
```
**Output**
```
2
```
**Explanation**

The two admissible paths with at most one turn are  

* Right → Right → Down → Down (0 turns)  
* Down → Down → Right → Right (0 turns)  

All other paths need at least two turns because they must go around the obstacle at `(1,1)`.
... [truncated; full doc has 4,539 chars]
```

### Sample 6

_(shard part_000000.parquet, row 76494, category=Nemotron-Pretraining-Unconditional-Algorithmic)_

```
**Problem**

You are given an array `A` of `N` integers (`1 ≤ N ≤ 2·10⁵`).  
You have to process `Q` operations (`1 ≤ Q ≤ 2·10⁵`).  
Each operation is one of the two types  

* **Update** `1  i  x` – set `A[i] = x` (`1‑based index`).  
* **Query**  `2  L  R` – consider the sub‑array `A[L … R]`.  
  From this sub‑array you may choose any subset of positions such that **no two chosen positions are adjacent**.  
  Output the maximum possible sum of the chosen elements.

All array values and updates satisfy `|A[i]| ≤ 10⁹`.

The answer for each query fits into a signed 64‑bit integer.

---

### Input
```
N Q
A1 A2 … AN
op1
op2
…
opQ
```
Each operation line has the format described above.

### Output
For each query (`op = 2`) output a single line containing the answer.



---

## Explanation  

For a fixed interval `[L,R]` the problem is exactly the **Maximum‑Weight Independent Set on a path**.  
For a path we can solve it with a simple DP:

```
dp0 = 0                         # best sum ending at previous position, previous not taken
dp1 = -∞                        # best sum ending at previous position, previous taken
for each value v in the interval:
        new_dp0 = max(dp0, dp1)                 # we do NOT take current
        new_dp1 = dp0 + v                       # we take current, previous must be not taken
        dp0, dp1 = new_dp0, new_dp1
answer = max(dp0, dp1)
```

Running this DP for every query would be `O(N·Q)` – far too slow.

### Segment‑tree formulation  

A se
... [truncated; full doc has 7,167 chars]
```

### Sample 7

_(shard part_000000.parquet, row 150315, category=Nemotron-Pretraining-Unconditional-Algorithmic)_

```
"""
Write a function square_and_sort that takes a list of integers and returns a new list of the squares of those integers sorted in ascending order.
assert square_and_sort([-4, -2, 0, 1, 3]) == [0, 1, 4, 9, 16]
"""
from typing import List

def square_and_sort(arr: List[int]) -> List[int]:
    
      """
      Return a new list containing the squares of the integers in *arr* sorted in ascending order.
      """
      squared = [x * x for x in arr]
      squared.sort()
      return squared

"""
Write a function map that takes a list and a callable and returns a new list with each element transformed by the callable.
assert map([True, False], lambda x: not x) == [False, True]
"""
from typing import Callable, List, Any

def map(array: List[Any], func: Callable[[Any], Any]) -> List[Any]:
    
      """Return a new list where each element of *array* is transformed by *func*.
  
      The function handles empty or ``None`` inputs by returning an empty list.
      """
      if not array:
          return []
      result: List[Any] = []
      for item in array:
          result.append(func(item))
      return result

"""
Write a function max_sum_no_adjacent(arr) that returns the maximum sum of a subsequence of arr where no two chosen elements are adjacent. Return 0 for empty or all-negative inputs.
assert max_sum_no_adjacent([0,0,0]) == 0
"""
def max_sum_no_adjacent(arr: list[int]) -> int:
    
      '''Return the maximum sum of a subsequence of *arr* with no adjacent elements.
     
... [truncated; full doc has 2,966 chars]
```

### Sample 8

_(shard part_000000.parquet, row 69552, category=Nemotron-Pretraining-Unconditional-Algorithmic)_

```
Problem: Minimum‑Time Travel with a Cost Budget

You are given a directed graph with `N` vertices (numbered `1 … N`) and `M` edges.  
Each edge `i` is described by four integers  

* `u_i` – start vertex  
* `v_i` – end vertex  
* `t_i` – time required to traverse the edge ( `1 ≤ t_i ≤ 10⁹`  )  
* `c_i` – cost to use the edge ( `0 ≤ c_i ≤ 10⁹` )

You have a budget `C` ( `0 ≤ C ≤ 10⁹` ).  
Starting from vertex `1`, you want to reach vertex `N`.  

**Goal** – find the minimum total time needed to travel from `1` to `N` while the sum of costs of the used edges does **not exceed** `C`.  
If it is impossible, output `-1`.

---

### Constraints
| Symbol | Value |
|--------|-------|
| `1 ≤ N ≤ 10⁵` | number of vertices |
| `1 ≤ M ≤ 2·10⁵` | number of edges |
| `0 ≤ C ≤ 10⁹` | total cost budget |
| `1 ≤ u_i, v_i ≤ N` | edge endpoints |
| `1 ≤ t_i ≤ 10⁹` | edge time |
| `0 ≤ c_i ≤ 10⁹` | edge cost |

The graph may contain multiple edges and self‑loops.  
All edges are directed.

---

### Input
```
N M C
u₁ v₁ t₁ c₁
u₂ v₂ t₂ c₂
...
u_M v_M t_M c_M
```

### Output
A single integer – the minimum possible total time, or `-1` if no feasible route exists.

---

## Explanation / Reasoning  

The problem is a classic *constrained shortest‑path* problem: we must minimise one metric (time) while a second metric (cost) is bounded.  
A naïve DP `dp[node][cost]` is impossible because `C` can be as large as `10⁹`.  
Instead we keep, for each vertex, **only the Pareto‑optimal states** that have been
... [truncated; full doc has 7,264 chars]
```

### Sample 9

_(shard part_000000.parquet, row 9632, category=Nemotron-Pretraining-Unconditional-Algorithmic)_

```
**Problem – Color‑Constrained Shortest Path**

You are given a directed weighted graph with `N` vertices (numbered `1 … N`) and `M` edges.  
Each edge `i` is described by four integers  

```
ui  vi  wi  ci
```

* `ui → vi` is a directed edge,
* `wi` ( `1 ≤ wi ≤ 10^6` ) is its length,
* `ci` ( `1 ≤ ci ≤ C` ) is its colour.

You have to travel from vertex `1` to vertex `N`.  
During the whole trip the following restrictions must hold:

1. **No two consecutive edges may have the same colour.**  
   (If you have just used an edge of colour `c`, the next edge you traverse must have a different colour.)

2. **You may use at most `K` different colours in total** (including the colour of the first edge).  

Find the minimum possible total length of a path that satisfies both restrictions.  
If no such path exists, output `-1`.

---

### Input  

```
N M C K
u1 v1 w1 c1
u2 v2 w2 c2
...
uM vM wM cM
```

* `1 ≤ N ≤ 5 000`
* `1 ≤ M ≤ 20 000`
* `1 ≤ C ≤ 10`
* `1 ≤ K ≤ C`
* `1 ≤ ui, vi ≤ N`
* `1 ≤ wi ≤ 10^6`
* `1 ≤ ci ≤ C`

The graph may contain parallel edges and self‑loops.

### Output  

A single integer – the length of the shortest feasible path, or `-1` if it does not exist.

---

### Explanation of the required algorithm  

The problem is a classic shortest‑path problem, but the state of the traveller is not only the current vertex.  
Two additional pieces of information influence which edges can be taken next:

* **The colour of the last traversed edge** – we must forbid taking ano
... [truncated; full doc has 5,496 chars]
```

### Sample 10

_(shard part_000000.parquet, row 90036, category=Nemotron-Pretraining-Unconditional-Algorithmic)_

```
**Problem: Smallest Subarray to Sort**

You are given an array `nums` of `n` integers.  
Your task is to find the length of the shortest (contiguous) subarray that, if you sort **only** this subarray in non‑decreasing order, the whole array becomes sorted in non‑decreasing order.

If the array is already sorted, return `0`.

**Input Format**

```
n                     // 1 ≤ n ≤ 10^5
a1 a2 a3 … an         // -10^9 ≤ ai ≤ 10^9
```

**Output Format**

```
len                  // length of the required subarray (0 ≤ len ≤ n)
```

**Example**

```
Input
7
1 3 2 2 2 4 5

Output
4
```

*Explanation*: Sorting the subarray `nums[2..5] = [3,2,2,2]` yields `[1,2,2,2,3,4,5]`, which is fully sorted. Its length is `4`.

---

**Explanation / Reasoning**

The problem can be solved in linear time with a single pass from both ends:

1. **Find the left boundary**  
   Scan from the left while the array is non‑decreasing. The first index `l` where `nums[l] > nums[l+1]` is the leftmost position that is out of order. If no such index exists the array is already sorted → answer `0`.

2. **Find the right boundary**  
   Scan from the right while the array is non‑increasing when read backwards (i.e. `nums[r-1] ≤ nums[r]`). The first index `r` where `nums[r-1] > nums[r]` marks the rightmost out‑of‑order element.

3. **Determine the needed extension**  
   The subarray `nums[l..r]` certainly needs to be sorted, but elements outside it may still be out of place after sorting because they are larger tha
... [truncated; full doc has 3,023 chars]
```

---
