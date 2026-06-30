import pytest
from oracle_flexcube_copilot.retrieval.fusion import RRFFuser
from oracle_flexcube_copilot.indexing.models import SearchResult

def test_rrf_fuser_basic():
    fuser = RRFFuser(k=60)
    
    # List 1: chunk A (rank 1), chunk B (rank 2)
    list1 = [
        SearchResult(chunk_id="A", score=0.9, source_document="doc1", page=1, heading="h", oracle_entities=[], text="textA", retrieval_method="vector"),
        SearchResult(chunk_id="B", score=0.8, source_document="doc1", page=1, heading="h", oracle_entities=[], text="textB", retrieval_method="vector")
    ]
    
    # List 2: chunk B (rank 1), chunk C (rank 2)
    list2 = [
        SearchResult(chunk_id="B", score=10.0, source_document="doc1", page=1, heading="h", oracle_entities=[], text="textB", retrieval_method="bm25"),
        SearchResult(chunk_id="C", score=5.0, source_document="doc1", page=1, heading="h", oracle_entities=[], text="textC", retrieval_method="bm25")
    ]
    
    fused = fuser.fuse([list1, list2], top_k=3)
    
    # RRF scores:
    # A = 1/61 = 0.01639
    # B = 1/62 + 1/61 = 0.01612 + 0.01639 = 0.0325
    # C = 1/62 = 0.01612
    # Order should be B, A, C
    
    assert len(fused) == 3
    assert fused[0].chunk_id == "B"
    assert fused[1].chunk_id == "A"
    assert fused[2].chunk_id == "C"
    
    # Check method fusion
    assert fused[0].retrieval_method == "vector+bm25"
