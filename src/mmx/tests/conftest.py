import pytest
from filelock import FileLock

from mmx.tests.utils import download_all_tf_models


@pytest.fixture(scope="session")
def download_models(tmp_path_factory, worker_id):
    if not worker_id:
        # not executing in with multiple workers, just produce the data and let
        # pytest's fixture caching do its job
        download_all_tf_models()

    # get the temp directory shared by all workers
    root_tmp_dir = tmp_path_factory.getbasetemp().parent

    fn = root_tmp_dir / "download_models.json"
    with FileLock(str(fn) + ".lock"):
        if not fn.is_file():
            download_all_tf_models()
            fn.write_text("Done")
