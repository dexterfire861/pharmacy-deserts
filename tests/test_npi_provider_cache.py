"""
Unit tests for NPPES provider cache building and ZIP lookups.
"""
from pathlib import Path
import tempfile
import shutil
import sys

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.loaders import (
    _compute_short_zip_series,
    build_nppes_zip_detail_cache,
    load_npi_pharmacist_data,
    load_npi_pharmacy_data,
    get_pharmacies_for_zip,
)


class TestNppesZipNormalization:
    def test_short_zip_rules(self):
        primary = pd.Series(["8701", "11368", "770840", "1234567", "12345678", "123456789", "", None])
        fallback = pd.Series(["", "", "", "", "", "", "90210-9999", "00601"])

        out = _compute_short_zip_series(primary, fallback).tolist()

        assert out[0] == "08701"   # 4-digit -> zfill
        assert out[1] == "11368"   # 5-digit keep
        assert out[2] == "00077"   # 6-digit -> first2 + zfill
        assert out[3] == "00123"   # 7-digit -> first3 + zfill
        assert out[4] == "1234".zfill(5)  # 8-digit -> first4 + zfill
        assert out[5] == "12345"   # 9-digit -> first5
        assert out[6] == "90210"   # fallback from mailing postal
        assert out[7] == "00601"   # fallback from mailing postal


class TestNppesCacheBuild:
    def test_build_and_lookup(self):
        temp_dir = tempfile.mkdtemp()
        try:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "npidata_pfile_test.csv"
            cache_dir = temp_path / "npi_cache"

            rows = [
                {
                    "NPI": "1000000001",
                    "Provider First Name": "Jane",
                    "Provider Last Name (Legal Name)": "Doe",
                    "Provider Organization Name (Legal Business Name)": "Independent Pharmacist One",
                    "Provider First Line Business Practice Location Address": "101 Main St",
                    "Provider Business Practice Location Address City Name": "Townsville",
                    "Provider Business Practice Location Address State Name": "NJ",
                    "Provider Business Practice Location Address Postal Code": "8701",
                    "Provider Business Mailing Address State Name": "NJ",
                    "Provider Business Mailing Address Postal Code": "08701",
                    "Provider Business Practice Location Address Country Code (If outside U.S.)": "US",
                    "Healthcare Provider Taxonomy Code_1": "183500000X",
                    "Provider Business Practice Location Address Telephone Number": "7321112222",
                },
                {
                    "NPI": "1000000002",
                    "Provider Organization Name (Legal Business Name)": "CVS Pharmacy 1234",
                    "Provider First Line Business Practice Location Address": "202 Broad St",
                    "Provider Business Practice Location Address City Name": "Queens",
                    "Provider Business Practice Location Address State Name": "NY",
                    "Provider Business Practice Location Address Postal Code": "11368-1234",
                    "Provider Business Mailing Address State Name": "NY",
                    "Provider Business Mailing Address Postal Code": "11368",
                    "Provider Business Practice Location Address Country Code (If outside U.S.)": "US",
                    "Healthcare Provider Taxonomy Code_1": "333600000X",
                    "Provider Business Practice Location Address Telephone Number": "(718) 555-4444",
                },
                {
                    "NPI": "1000000003",
                    "Provider Organization Name (Legal Business Name)": "Walgreen Community",
                    "Provider First Line Business Practice Location Address": "303 Oak Ave",
                    "Provider Business Practice Location Address City Name": "Houston",
                    "Provider Business Practice Location Address State Name": "TX",
                    "Provider Business Practice Location Address Postal Code": "770840000",
                    "Provider Business Mailing Address State Name": "TX",
                    "Provider Business Mailing Address Postal Code": "77084",
                    "Provider Business Practice Location Address Country Code (If outside U.S.)": "US",
                    "Healthcare Provider Taxonomy Code_1": "3336C0003X",
                    "Provider Business Practice Location Address Telephone Number": "2813339999",
                },
                {
                    # should be filtered out (PR)
                    "NPI": "1000000004",
                    "Provider Organization Name (Legal Business Name)": "PR Pharmacy",
                    "Provider First Line Business Practice Location Address": "404 Island Rd",
                    "Provider Business Practice Location Address City Name": "San Juan",
                    "Provider Business Practice Location Address State Name": "PR",
                    "Provider Business Practice Location Address Postal Code": "00901",
                    "Provider Business Mailing Address State Name": "PR",
                    "Provider Business Mailing Address Postal Code": "00901",
                    "Provider Business Practice Location Address Country Code (If outside U.S.)": "US",
                    "Healthcare Provider Taxonomy Code_1": "333600000X",
                    "Provider Business Practice Location Address Telephone Number": "7870000000",
                },
                {
                    # should be filtered out (non-US)
                    "NPI": "1000000005",
                    "Provider Organization Name (Legal Business Name)": "Toronto Pharmacy",
                    "Provider First Line Business Practice Location Address": "505 North St",
                    "Provider Business Practice Location Address City Name": "Toronto",
                    "Provider Business Practice Location Address State Name": "ON",
                    "Provider Business Practice Location Address Postal Code": "M5V3L9",
                    "Provider Business Mailing Address State Name": "ON",
                    "Provider Business Mailing Address Postal Code": "M5V3L9",
                    "Provider Business Practice Location Address Country Code (If outside U.S.)": "CA",
                    "Healthcare Provider Taxonomy Code_1": "333600000X",
                    "Provider Business Practice Location Address Telephone Number": "4160000000",
                },
            ]
            pd.DataFrame(rows).to_csv(csv_path, index=False)

            result = build_nppes_zip_detail_cache(
                csv_path=csv_path,
                cache_dir=cache_dir,
                chunksize=2,
                force=True,
            )
            assert result["pharmacist_rows"] == 1
            assert result["pharmacy_rows"] == 2

            pharmacist_df = load_npi_pharmacist_data(
                data_dir=str(temp_path),
                auto_build=False,
            )
            pharmacy_df = load_npi_pharmacy_data(
                data_dir=str(temp_path),
                auto_build=False,
            )

            assert len(pharmacist_df) == 1
            assert len(pharmacy_df) == 2
            assert pharmacist_df["Short_ZIP"].iloc[0] == "08701"
            assert pharmacist_df["Combined"].iloc[0] == "Jane Doe"
            assert sorted(pharmacy_df["Short_ZIP"].tolist()) == ["11368", "77084"]

            # ZIP lookups should return chain pharmacies first (alphabetical within class).
            lookup_rows = pd.DataFrame(
                [
                    {"Short_ZIP": "10001", "pharmacy_name": "Indie Care", "Chain": "Independent", "Phone": "", "Address": ""},
                    {"Short_ZIP": "10001", "pharmacy_name": "CVS Midtown", "Chain": "CVS", "Phone": "", "Address": ""},
                    {"Short_ZIP": "10001", "pharmacy_name": "Walgreens 8th", "Chain": "Walgreens", "Phone": "", "Address": ""},
                ]
            )
            pharmacies = get_pharmacies_for_zip("10001", lookup_rows)
            assert [p[0] for p in pharmacies] == ["CVS Midtown", "Walgreens 8th", "Indie Care"]
        finally:
            shutil.rmtree(temp_dir)
