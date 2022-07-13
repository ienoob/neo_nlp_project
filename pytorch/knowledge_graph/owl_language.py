#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/3/31 21:34
    @Author  : jack.li
    @Site    : 
    @File    : owl_language.py

"""
import os
from owlready2 import default_world, get_ontology

import io, bz2

DBPEDIA_DIR = "d:\\_DBPedia"
TMP_DIR = "d:\\sqlite_tmp_dir"
QUADSTORE = "dbpedia.sqlite3"
default_world.set_backend(
    filename=QUADSTORE,
    sqlite_tmp_dir=TMP_DIR)

dbpedia = get_ontology("http://wikidata.dbpedia.org/ontology/")
contenu = open(os.path.join(DBPEDIA_DIR, "ontology_type=parsed.owl"), encoding="utf8").read()
contenu = contenu.replace("http://dbpedia.org/ontology", "http://wikidata.dbpedia.org/ontology/")
contenu = contenu.replace("http://www.wikidata.org/entity", "http://wikidata.dbpedia.org/ontology/")
dbpedia.load(fileobj=io.BytesIO(contenu.encode("utf8")))
