from unittest.mock import MagicMock, patch

import pytest

from nemo_run.core.packaging.configmap import ConfigMapPackager


@pytest.fixture
def temp_py_files(tmp_path):
    """Create test files for packaging."""
    # Create some test files
    file1 = tmp_path / "a.py"
    file2 = tmp_path / "b.py"
    file3 = tmp_path / "subdir" / "c.py"
    file3.parent.mkdir()

    file1.write_text("print('A')\n")
    file2.write_text("print('B')\n")
    file3.write_text("print('C')\n")

    return tmp_path, [file1, file2, file3]


@pytest.mark.parametrize(
    "job_dir,expected_prefix",
    [
        ("test-job", "test-job/"),
        ("", ""),
    ],
)
def test_package_creates_configmap_with_job_dir(temp_py_files, job_dir, expected_prefix):
    """Test that package creates a ConfigMap with the correct data for different job_dir values."""
    tmp_path, files = temp_py_files
    mock_v1 = MagicMock()

    with patch(
        "nemo_run.core.packaging.configmap.ConfigMapPackager.__post_init__",
        lambda self: setattr(self, "v1", mock_v1),
    ):
        packager = ConfigMapPackager(include_pattern="*.py", relative_path=".", namespace="test-ns")
        configmap_name = packager.package(tmp_path, job_dir, "testjob")

        assert configmap_name == "nemo-workspace-testjob"
        assert mock_v1.create_namespaced_config_map.called

        _, kwargs = mock_v1.create_namespaced_config_map.call_args
        assert kwargs["namespace"] == "test-ns"

        data = kwargs["body"].data
        for file_path in files:
            rel_path = file_path.relative_to(tmp_path)
            configmap_key = f"{expected_prefix}{rel_path}" if expected_prefix else str(rel_path)
            assert configmap_key in data
            assert data[configmap_key] == file_path.read_text()


def test_cleanup_deletes_configmap():
    """Test that cleanup deletes the ConfigMap."""
    mock_v1 = MagicMock()

    with patch(
        "nemo_run.core.packaging.configmap.ConfigMapPackager.__post_init__",
        lambda self: setattr(self, "v1", mock_v1),
    ):
        packager = ConfigMapPackager()
        packager.cleanup("testjob")

        assert mock_v1.delete_namespaced_config_map.called
        _, kwargs = mock_v1.delete_namespaced_config_map.call_args
        assert kwargs["name"] == "nemo-workspace-testjob"
        assert kwargs["namespace"] == "default"


def test_find_files_to_package(temp_py_files):
    """Test file finding logic."""
    tmp_path, files = temp_py_files

    # Add a non-Python file to test filtering
    txt_file = tmp_path / "b.txt"
    txt_file.write_text("text file")

    packager = ConfigMapPackager(include_pattern="*.py", relative_path=".")
    found_files = packager._find_files_to_package(tmp_path)

    # Use files from fixture to make test maintainable
    assert len(found_files) == len(files)  # Should find all Python files from fixture

    # Check that all fixture files are found
    for file_path in files:
        assert file_path in found_files

    # Check that the non-Python file is NOT found
    assert txt_file not in found_files


def test_package_no_files_found(temp_py_files):
    """Test behavior when no files match the pattern."""
    tmp_path, _ = temp_py_files
    mock_v1 = MagicMock()

    with patch(
        "nemo_run.core.packaging.configmap.ConfigMapPackager.__post_init__",
        lambda self: setattr(self, "v1", mock_v1),
    ):
        packager = ConfigMapPackager(include_pattern="*.nonexistent", relative_path=".")
        configmap_name = packager.package(tmp_path, "test-job", "testjob")

        assert configmap_name == "nemo-workspace-testjob"
        # Should not call create_namespaced_config_map
        assert not mock_v1.create_namespaced_config_map.called


def test_package_kubernetes_client_unavailable(temp_py_files):
    """Test behavior when Kubernetes client is not available."""
    tmp_path, _ = temp_py_files

    with patch(
        "nemo_run.core.packaging.configmap.ConfigMapPackager.__post_init__",
        lambda self: setattr(self, "v1", None),
    ):
        packager = ConfigMapPackager()
        configmap_name = packager.package(tmp_path, "test-job", "testjob")

        assert configmap_name == "nemo-workspace-testjob"


def test_cleanup_kubernetes_client_unavailable():
    """Test cleanup behavior when Kubernetes client is not available."""
    with patch(
        "nemo_run.core.packaging.configmap.ConfigMapPackager.__post_init__",
        lambda self: setattr(self, "v1", None),
    ):
        packager = ConfigMapPackager()
        # Should not raise any exception
        packager.cleanup("testjob")


def test_package_with_large_files(temp_py_files):
    """Test that package handles large files appropriately."""
    tmp_path, files = temp_py_files
    mock_v1 = MagicMock()

    # Create a large file that would exceed the 1MB limit
    large_file = tmp_path / "large_file.py"
    large_content = "print('x')\n" * 200000  # Create a large file (~1.2MB)
    large_file.write_text(large_content)

    with patch(
        "nemo_run.core.packaging.configmap.ConfigMapPackager.__post_init__",
        lambda self: setattr(self, "v1", mock_v1),
    ):
        packager = ConfigMapPackager(include_pattern="*.py", relative_path=".", debug=True)
        configmap_name = packager.package(tmp_path, "test-job", "testjob")

        # Should return the configmap name but not create it due to size limit
        assert configmap_name == "nemo-workspace-testjob"
        # Should not call create_namespaced_config_map due to size limit
        assert not mock_v1.create_namespaced_config_map.called
