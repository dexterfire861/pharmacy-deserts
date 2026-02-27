"""
Unit tests for source accumulation logic in dataset versioning.
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
    DatasetStorageLocal, generate_source_id, build_file_mapping,
    build_dataset_config, generate_version_id
)


class TestSourceIdGeneration:
    """Test source ID generation."""
    
    def test_basic_filename(self):
        """Test source_id generation for basic filename."""
        source_id = generate_source_id("health_data.csv")
        assert source_id == "health_data.csv"
    
    def test_filename_with_sheet(self):
        """Test source_id generation with Excel sheet name."""
        source_id = generate_source_id("data.xlsx", "Sheet1")
        assert source_id == "data.xlsx_sheet1"
    
    def test_filename_normalization(self):
        """Test that filenames are normalized (lowercase)."""
        source_id = generate_source_id("HEALTH_DATA.CSV")
        assert source_id == "health_data.csv"
    
    def test_sheet_name_normalization(self):
        """Test that sheet names are normalized."""
        source_id = generate_source_id("data.xlsx", "My Sheet")
        assert source_id == "data.xlsx_my_sheet"
    
    def test_stable_ids(self):
        """Test that same filename+sheet produces same source_id."""
        id1 = generate_source_id("test.csv", "Sheet1")
        id2 = generate_source_id("test.csv", "Sheet1")
        assert id1 == id2


class TestSourceAccumulation:
    """Test source accumulation across versions."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage for testing."""
        temp_dir = tempfile.mkdtemp()
        storage = DatasetStorageLocal(base_path=temp_dir)
        yield storage
        shutil.rmtree(temp_dir)
    
    def test_new_source_added(self, temp_storage):
        """Test that new source_id adds to existing list."""
        dataset_id = "test_dataset"
        version1 = generate_version_id()
        
        # Create first version with one source
        source1 = build_file_mapping(
            filename="file1.csv",
            file_type="csv",
            zip_column="zip",
            normalization_mode="already_5_digit",
            feature_columns=["col1"],
            column_renames={},
            cleaning_rules={}
        )
        
        # Upload file and config
        temp_storage.upload_file(dataset_id, version1, "file1.csv", b"test content")
        config1 = build_dataset_config(dataset_id, version1, [source1])
        temp_storage.upload_config(dataset_id, version1, config1)
        temp_storage.update_latest(dataset_id, version1)
        
        # Create second version with new source
        version2 = generate_version_id()
        source2 = build_file_mapping(
            filename="file2.csv",
            file_type="csv",
            zip_column="zip",
            normalization_mode="already_5_digit",
            feature_columns=["col2"],
            column_renames={},
            cleaning_rules={}
        )
        
        # Load existing sources
        existing_config = temp_storage.get_config(dataset_id, version1)
        existing_sources = existing_config.get('sources', [])
        
        # Build accumulated sources
        existing_by_id = {s.get('source_id', generate_source_id(s['filename'])): s for s in existing_sources}
        new_source_id = source2['source_id']
        
        # New source should not exist
        assert new_source_id not in existing_by_id
        
        # Accumulated should include both
        accumulated = existing_sources + [source2]
        assert len(accumulated) == 2
        assert accumulated[0]['source_id'] == source1['source_id']
        assert accumulated[1]['source_id'] == source2['source_id']
    
    def test_existing_source_replaced(self, temp_storage):
        """Test that same source_id replaces old source."""
        dataset_id = "test_dataset"
        version1 = generate_version_id()
        
        # Create first version
        source1 = build_file_mapping(
            filename="file1.csv",
            file_type="csv",
            zip_column="zip",
            normalization_mode="already_5_digit",
            feature_columns=["col1"],
            column_renames={},
            cleaning_rules={}
        )
        
        temp_storage.upload_file(dataset_id, version1, "file1.csv", b"old content")
        config1 = build_dataset_config(dataset_id, version1, [source1])
        temp_storage.upload_config(dataset_id, version1, config1)
        temp_storage.update_latest(dataset_id, version1)
        
        # Create second version with same filename (replacement)
        version2 = generate_version_id()
        source2 = build_file_mapping(
            filename="file1.csv",  # Same filename = same source_id
            file_type="csv",
            zip_column="zip",
            normalization_mode="already_5_digit",
            feature_columns=["col1", "col2"],  # Different features
            column_renames={},
            cleaning_rules={}
        )
        
        # Load existing sources
        existing_config = temp_storage.get_config(dataset_id, version1)
        existing_sources = existing_config.get('sources', [])
        existing_by_id = {s.get('source_id', generate_source_id(s['filename'])): s for s in existing_sources}
        
        # Build accumulated sources (replacement logic)
        replaced_source_ids = {source2['source_id']}
        accumulated = []
        for src in existing_sources:
            source_id = src.get('source_id', generate_source_id(src['filename']))
            if source_id not in replaced_source_ids:
                accumulated.append(src)
        accumulated.append(source2)
        
        # Should have only one source (replaced)
        assert len(accumulated) == 1
        assert accumulated[0]['source_id'] == source2['source_id']
        assert len(accumulated[0]['feature_columns']) == 2  # New features
    
    def test_mixed_add_replace(self, temp_storage):
        """Test multiple new sources + replacements in one upload."""
        dataset_id = "test_dataset"
        version1 = generate_version_id()
        
        # Create first version with 2 sources
        source1a = build_file_mapping(
            filename="file1.csv",
            file_type="csv",
            zip_column="zip",
            normalization_mode="already_5_digit",
            feature_columns=["col1"],
            column_renames={},
            cleaning_rules={}
        )
        source1b = build_file_mapping(
            filename="file2.csv",
            file_type="csv",
            zip_column="zip",
            normalization_mode="already_5_digit",
            feature_columns=["col2"],
            column_renames={},
            cleaning_rules={}
        )
        
        temp_storage.upload_file(dataset_id, version1, "file1.csv", b"content1")
        temp_storage.upload_file(dataset_id, version1, "file2.csv", b"content2")
        config1 = build_dataset_config(dataset_id, version1, [source1a, source1b])
        temp_storage.upload_config(dataset_id, version1, config1)
        temp_storage.update_latest(dataset_id, version1)
        
        # Create second version: replace file1, add file3
        version2 = generate_version_id()
        source2a = build_file_mapping(
            filename="file1.csv",  # Replacement
            file_type="csv",
            zip_column="zip",
            normalization_mode="already_5_digit",
            feature_columns=["col1", "col1b"],
            column_renames={},
            cleaning_rules={}
        )
        source2b = build_file_mapping(
            filename="file3.csv",  # New
            file_type="csv",
            zip_column="zip",
            normalization_mode="already_5_digit",
            feature_columns=["col3"],
            column_renames={},
            cleaning_rules={}
        )
        
        # Load existing and build accumulated
        existing_config = temp_storage.get_config(dataset_id, version1)
        existing_sources = existing_config.get('sources', [])
        existing_by_id = {s.get('source_id', generate_source_id(s['filename'])): s for s in existing_sources}
        
        new_sources = [source2a, source2b]
        replaced_source_ids = {s['source_id'] for s in new_sources}
        
        accumulated = []
        for src in existing_sources:
            source_id = src.get('source_id', generate_source_id(src['filename']))
            if source_id not in replaced_source_ids:
                accumulated.append(src)
        accumulated.extend(new_sources)
        
        # Should have: file2 (unchanged), file1 (replaced), file3 (new) = 3 sources
        assert len(accumulated) == 3
        source_ids = {s['source_id'] for s in accumulated}
        assert generate_source_id("file1.csv") in source_ids
        assert generate_source_id("file2.csv") in source_ids
        assert generate_source_id("file3.csv") in source_ids
    
    def test_empty_existing(self, temp_storage):
        """Test first upload (no existing sources) works."""
        dataset_id = "test_dataset"
        version1 = generate_version_id()
        
        # No existing version
        latest = temp_storage.get_latest_version(dataset_id)
        assert latest is None
        
        # First upload
        source1 = build_file_mapping(
            filename="file1.csv",
            file_type="csv",
            zip_column="zip",
            normalization_mode="already_5_digit",
            feature_columns=["col1"],
            column_renames={},
            cleaning_rules={}
        )
        
        accumulated = [source1]  # No existing to merge with
        assert len(accumulated) == 1
    
    def test_unchanged_sources_preserved(self, temp_storage):
        """Test that unchanged sources are included in new version."""
        dataset_id = "test_dataset"
        version1 = generate_version_id()
        
        # Create first version with 2 sources
        source1a = build_file_mapping(
            filename="file1.csv",
            file_type="csv",
            zip_column="zip",
            normalization_mode="already_5_digit",
            feature_columns=["col1"],
            column_renames={},
            cleaning_rules={}
        )
        source1b = build_file_mapping(
            filename="file2.csv",
            file_type="csv",
            zip_column="zip",
            normalization_mode="already_5_digit",
            feature_columns=["col2"],
            column_renames={},
            cleaning_rules={}
        )
        
        temp_storage.upload_file(dataset_id, version1, "file1.csv", b"content1")
        temp_storage.upload_file(dataset_id, version1, "file2.csv", b"content2")
        config1 = build_dataset_config(dataset_id, version1, [source1a, source1b])
        temp_storage.upload_config(dataset_id, version1, config1)
        temp_storage.update_latest(dataset_id, version1)
        
        # Create second version with only new source (file1 and file2 unchanged)
        version2 = generate_version_id()
        source2 = build_file_mapping(
            filename="file3.csv",  # New
            file_type="csv",
            zip_column="zip",
            normalization_mode="already_5_digit",
            feature_columns=["col3"],
            column_renames={},
            cleaning_rules={}
        )
        
        # Build accumulated
        existing_config = temp_storage.get_config(dataset_id, version1)
        existing_sources = existing_config.get('sources', [])
        replaced_source_ids = {source2['source_id']}
        
        accumulated = []
        for src in existing_sources:
            source_id = src.get('source_id', generate_source_id(src['filename']))
            if source_id not in replaced_source_ids:
                accumulated.append(src)
        accumulated.append(source2)
        
        # Should have all 3 sources
        assert len(accumulated) == 3
        source_ids = {s['source_id'] for s in accumulated}
        assert generate_source_id("file1.csv") in source_ids
        assert generate_source_id("file2.csv") in source_ids
        assert generate_source_id("file3.csv") in source_ids
