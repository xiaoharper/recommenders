# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
from unittest.mock import patch
from requests import Session
from reco_utils.dataset.wikidata import (
    search_wikidata,
    find_wikidata_id,
    query_entity_links,
    read_linked_entities,
    query_entity_description,
    requests,
)

class MockResponse:
    # Class that mocks requests.models.Response
    def __init__(self, content, error):
        self._content = content
        self._error = error

    def json(self, params):
        if "list" in params:
            return {"query": {"search": [ {"pageid":}]}}
        elif "pageids" in params:
            return {}

@pytest.fixture(scope="module")
def q():
    return {
        "correct": "the lord of the rings",
        "not_correct": "000000aaaaa",
        "entity_id": "Q15228",
    }


def test_find_wikidata_id(q):
    #assert find_wikidata_id(q["correct"]) == "Q15228"
    #assert find_wikidata_id(q["not_correct"]) == "entityNotFound"
    with patch("requests.get", side_effect)


def test_query_entity_links(q):
    resp = query_entity_links(q["entity_id"])
    assert "head" in resp
    assert "results" in resp


def test_read_linked_entities(q):
    resp = query_entity_links(q["entity_id"])
    related_links = read_linked_entities(resp)
    assert len(related_links) > 5


def test_query_entity_description(q):
    desc = query_entity_description(q["entity_id"])
    assert desc == "1954–1955 fantasy novel by J. R. R. Tolkien"


def test_search_wikidata():
    # TODO
    pass
