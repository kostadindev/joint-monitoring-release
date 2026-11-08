from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from record_exact_grouping import partitions


def test_exact_unlabeled_partition_counts_and_shape():
    for n,sizes,count in [(7,[2,2,2,1],105),(8,[2,2,2,2],105),(8,[2,2,2,1,1],420)]:
        es=[('a',str(i)) for i in range(n)];ps=partitions(es,sizes)
        assert len(ps)==count and len(set(ps))==count
        for groups in ps:
            assert sorted(e for g in groups for e in g)==sorted(es)
            assert sorted(map(len,groups))==sorted(sizes)
