"""
Unit tests for file copying between versions.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from storage.datasets import (
    DatasetStorageLocal, generate_version_id, build_file_mapping,
    build_dataset_config
)


class TestFileCopying:
    """Test file copying between versions."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage for testing."""
        temp_dir = tempfile.mkdtemp()
        storage = DatasetStorageLocal(base_path=temp_dir)
        yield storage
        shutil.rmtree(temp_dir)
    
    def test_copy_file_from_version(self, temp_storage):
        """Test that files are copied correctly between versions."""
        dataset_id = "test_dataset"
        version1 = generate_version_id()
        version2 = generate_version_id()
        filename = "test_file.csv"
        content = b"test file content"
        
        # Upload file to version1
        path1 = temp_storage.upload_file(dataset_id, version1, filename, content)
        assert Path(path1).exists()
        assert Path(path1).read_bytes() == content
        
        # Copy to version2
        path2 = temp_storage.copy_file_from_version(
            dataset_id, version1, version2, filename
        )
        
        # Verify file exists in version2
        assert Path(path2).exists()
        assert Path(path2).read_bytes() == content
        
        # Verify both files exist
        assert Path(path1).exists()
        assert Path(path2).exists()
    
    def test_unchanged_source_file_available(self, temp_storage):
        """Test that unchanged source files are accessible in new version."""
        dataset_id = "test_dataset"
        version1 = generate_version_id()
        version2 = generate_version_id()
        filename = "unchanged_file.csv"
        content = b"unchanged content"
        
        # Upload to version1
        temp_storage.upload_file(dataset_id, version1, filename, content)
        
        # Copy to version2
        temp_storage.copy_file_from_version(
            dataset_id, version1, version2, filename
        )
        
        # Verify file can be downloaded from version2
        downloaded = temp_storage.download_file(dataset_id, version2, filename)
        assert downloaded == content
    
    def test_copy_nonexistent_file_raises_error(self, temp_storage):
        """Test that copying nonexistent file raises FileNotFoundError."""
        dataset_id = "test_dataset"
        version1 = generate_version_id()
        version2 = generate_version_id()
        
        with pytest.raises(FileNotFoundError):
            temp_storage.copy_file_from_version(
                dataset_id, version1, version2, "nonexistent.csv"
            )
    
    def test_copy_multiple_files(self, temp_storage):
        """Test copying multiple files between versions."""
        dataset_id = "test_dataset"
        version1 = generate_version_id()
        version2 = generate_version_id()
        
        files = {
            "file1.csv": b"content1",
            "file2.csv": b"content2",
            "file3.csv": b"content3"
        }
        
        # Upload all files to version1
        for filename, content in files.items():
            temp_storage.upload_file(dataset_id, version1, filename, content)
        
        # Copy all files to version2
        for filename in files.keys():
            temp_storage.copy_file_from_version(
                dataset_id, version1, version2, filename
            )
        
        # Verify all files exist in version2
        for filename, expected_content in files.items():
            downloaded = temp_storage.download_file(dataset_id, version2, filename)
            assert downloaded == expected_content
