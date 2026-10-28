from pathlib import Path

from cxr_flows.versioning import dataset_rev, dvc_file, dvc_md5, pinned_md5s


def test_dataset_rev_is_order_independent_and_content_addressed() -> None:
    a = dataset_rev({"data/nih": "x.dir", "data/test_locked": "y.dir"})
    b = dataset_rev({"data/test_locked": "y.dir", "data/nih": "x.dir"})
    assert a == b and len(a) == 12
    assert a != dataset_rev({"data/nih": "z.dir", "data/test_locked": "y.dir"})


def test_dvc_md5_reads_the_pointer_file(tmp_path: Path) -> None:
    target = tmp_path / "nih"
    dvc_file(target).write_text(
        "outs:\n- md5: 36a7112e01e72b6cffb24e7676353acf.dir\n  size: 10\n  nfiles: 2\n"
        "  hash: md5\n  path: nih\n"
    )
    assert dvc_md5(target) == "36a7112e01e72b6cffb24e7676353acf.dir"


def test_pinned_md5s_keys_by_path_relative_to_root(tmp_path: Path) -> None:
    targets = [tmp_path / "data" / "nih", tmp_path / "data" / "test_locked"]
    targets[0].parent.mkdir()
    for target, md5 in zip(targets, ("a", "b"), strict=True):
        dvc_file(target).write_text(f"outs:\n- md5: {md5}.dir\n  path: {target.name}\n")
    assert pinned_md5s(tmp_path, targets) == {"data/nih": "a.dir", "data/test_locked": "b.dir"}
